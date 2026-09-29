/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The vk_sync type behind VkFence and binary VkSemaphore. There is one
 * queue per device, so a sync is fully described by "signalled from the
 * host", or "pending on the done values the queue's subqueues will reach"
 * (mali_queue.c), or neither. Host waits read the done slots in GPU memory
 * and sleep on the device's condition variable, which the event thread
 * broadcasts on every kernel event. Timeline semaphores are not supported:
 * ARMSX2 does not use them and the device does not report the feature.
 *
 * A third kind of payload is an imported sync file (the Android acquire
 * fence): the sync owns a dup of the fd. Host waits poll it; GPU waits go
 * through the kcpu queue (mali_sync_file.c, mali_queue.c). Export makes a
 * sync file for whatever the payload is.
 */

#include "mali_queue.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "util/os_time.h"

#include "mali_vk.h"

static struct mali_sync *
to_sync(struct vk_sync *s)
{
   return container_of(s, struct mali_sync, vk);
}

static struct mali_csf_device *
csf_of(struct vk_device *vkdev)
{
   return container_of(vkdev, struct mali_device, vk)->csf;
}

void
mali_sync_clear(struct mali_sync *s)
{
   s->host_signaled = false;
   s->submitted = false;
   if (s->fd >= 0) {
      close(s->fd);
      s->fd = -1;
   }
}

static VkResult
sync_init(struct vk_device *vkdev, struct vk_sync *vs, uint64_t initial_value)
{
   struct mali_sync *s = to_sync(vs);
   s->host_signaled = initial_value != 0;
   s->submitted = false;
   s->fd = -1;
   memset(s->req, 0, sizeof(s->req));
   return VK_SUCCESS;
}

static void
sync_finish(struct vk_device *vkdev, struct vk_sync *vs)
{
   struct mali_sync *s = to_sync(vs);
   if (s->fd >= 0)
      close(s->fd);
   s->fd = -1;
}

static VkResult
sync_signal(struct vk_device *vkdev, struct vk_sync *vs, uint64_t value)
{
   struct mali_csf_device *csf = csf_of(vkdev);
   struct mali_sync *s = to_sync(vs);
   pthread_mutex_lock(&csf->lock);
   mali_sync_clear(s);
   s->host_signaled = true;
   pthread_cond_broadcast(&csf->cond);
   pthread_mutex_unlock(&csf->lock);
   return VK_SUCCESS;
}

static VkResult
sync_reset(struct vk_device *vkdev, struct vk_sync *vs)
{
   struct mali_csf_device *csf = csf_of(vkdev);
   struct mali_sync *s = to_sync(vs);
   pthread_mutex_lock(&csf->lock);
   mali_sync_clear(s);
   pthread_mutex_unlock(&csf->lock);
   return VK_SUCCESS;
}

static VkResult
sync_move(struct vk_device *vkdev, struct vk_sync *dst, struct vk_sync *src)
{
   struct mali_csf_device *csf = csf_of(vkdev);
   struct mali_sync *d = to_sync(dst), *s = to_sync(src);
   pthread_mutex_lock(&csf->lock);
   mali_sync_clear(d);
   d->host_signaled = s->host_signaled;
   d->submitted = s->submitted;
   memcpy(d->req, s->req, sizeof(d->req));
   d->fd = s->fd;
   s->fd = -1;
   mali_sync_clear(s);
   pthread_mutex_unlock(&csf->lock);
   return VK_SUCCESS;
}

/* With csf->lock held. A sync file found signalled becomes a host
 * signal (its fd is closed). */
static bool
sync_is_signaled(struct mali_csf_device *csf, struct mali_sync *s, bool pending)
{
   if (s->host_signaled)
      return true;
   if (s->fd >= 0) {
      if (pending)
         return true;
      if (!mali_sync_file_signaled(s->fd))
         return false;
      mali_sync_clear(s);
      s->host_signaled = true;
      return true;
   }
   if (!s->submitted)
      return false;
   /* In immediate submit mode a submitted signal is already "pending". */
   return pending || mali_csf_reached(csf, s->req);
}

/* Without kernel events a sleeping wait re-reads the done slots this
 * often; with them, this only bounds the damage of a missed event. */
#define WAIT_SLICE_NO_EVENTS_NS (1ll * 1000 * 1000)
#define WAIT_SLICE_EVENTS_NS (20ll * 1000 * 1000)

static VkResult
sync_wait_many(struct vk_device *vkdev, uint32_t count,
               const struct vk_sync_wait *waits, enum vk_sync_wait_flags flags,
               uint64_t abs_timeout_ns)
{
   struct mali_csf_device *csf = csf_of(vkdev);
   const bool any = flags & VK_SYNC_WAIT_ANY;
   const bool pending = flags & VK_SYNC_WAIT_PENDING;
   const int64_t slice = csf->thread_running ? WAIT_SLICE_EVENTS_NS : WAIT_SLICE_NO_EVENTS_NS;
   VkResult result;
   bool slept = false;

   pthread_mutex_lock(&csf->lock);
   for (;;) {
      if (csf->lost) {
         result = VK_ERROR_DEVICE_LOST;
         break;
      }

      /* A faulted stream shows up in the done slots' error words. */
      struct mali_csf_queue *q = csf->queue;
      if (q) {
         volatile struct mali_cs_sync64 *done = mali_queue_done(q);
         bool error = false;
         for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++)
            error |= done[i].error != 0;
         if (error) {
            pthread_mutex_unlock(&csf->lock);
            mali_csf_set_lost(csf, "a command stream reported an error in its done slot");
            return VK_ERROR_DEVICE_LOST;
         }
      }

      /* Sync files are waited for with poll(), everything else on the
       * condition variable. */
      struct pollfd fds[16];
      uint32_t nfds = 0, signaled = 0;
      bool gpu_pending = false;
      for (uint32_t i = 0; i < count; i++) {
         struct mali_sync *s = to_sync(waits[i].sync);
         if (sync_is_signaled(csf, s, pending)) {
            signaled++;
         } else if (s->fd >= 0 && nfds < ARRAY_SIZE(fds)) {
            fds[nfds++] = (struct pollfd){.fd = s->fd, .events = POLLIN};
         } else {
            gpu_pending = true;
         }
      }
      if ((any && signaled) || signaled == count) {
         result = VK_SUCCESS;
         break;
      }

      int64_t now = os_time_get_nano();
      if ((uint64_t)now >= abs_timeout_ns) {
         result = VK_TIMEOUT;
         break;
      }

      int64_t until = MIN2((int64_t)MIN2(abs_timeout_ns, (uint64_t)INT64_MAX), now + slice);
      if (!slept) {
         csf->stats.waits++;
         slept = true;
      }
      if (nfds) {
         /* A sync file cannot wake the condition variable. With GPU work
          * to watch as well, poll in short slices. The fds stay valid:
          * only a reset, signal or import of these syncs closes them, and
          * the application may not do that during the wait. */
         int64_t len = (gpu_pending ? MIN2(until, now + WAIT_SLICE_NO_EVENTS_NS) : until) - now;
         struct timespec ts = {
            .tv_sec = len / 1000000000ll,
            .tv_nsec = len % 1000000000ll,
         };
         pthread_mutex_unlock(&csf->lock);
         ppoll(fds, nfds, &ts, NULL);
         pthread_mutex_lock(&csf->lock);
      } else {
         struct timespec ts = {
            .tv_sec = until / 1000000000ll,
            .tv_nsec = until % 1000000000ll,
         };
         pthread_cond_timedwait(&csf->cond, &csf->lock, &ts);
      }
      csf->stats.wakeups++;
   }
   pthread_mutex_unlock(&csf->lock);
   return result;
}

/*
 * Import: the sync takes a dup of the fd (the runtime closes the caller's
 * after a successful import), after the blob's check that it is a sync
 * file (KBASE_IOCTL_FENCE_VALIDATE). A sync file that has signalled
 * already is kept as a host signal. fd -1 never gets here (the runtime
 * signals the sync instead).
 */
static VkResult
sync_import_sync_file(struct vk_device *vkdev, struct vk_sync *vs, int fd)
{
   struct mali_csf_device *csf = csf_of(vkdev);
   struct mali_sync *s = to_sync(vs);

   int dup_fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
   if (dup_fd < 0)
      return vk_errorf(vkdev, errno == EMFILE ? VK_ERROR_TOO_MANY_OBJECTS :
                                                VK_ERROR_OUT_OF_HOST_MEMORY,
                       "cannot duplicate sync file %d: %s", fd, strerror(errno));
   if (mali_kbase_fence_validate(csf->kb, dup_fd) != MALI_KBASE_SUCCESS) {
      close(dup_fd);
      return vk_errorf(vkdev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "fd %d is not a sync file", fd);
   }

   pthread_mutex_lock(&csf->lock);
   mali_sync_clear(s);
   if (mali_sync_file_signaled(dup_fd)) {
      close(dup_fd);
      s->host_signaled = true;
   } else {
      s->fd = dup_fd;
   }
   pthread_cond_broadcast(&csf->cond);
   pthread_mutex_unlock(&csf->lock);
   return VK_SUCCESS;
}

/*
 * Export: a dup of an imported sync file, -1 for a payload that has
 * signalled (the Android and Vulkan conventions for "already signalled"),
 * or a new sync file from the kcpu queue for pending GPU work. The runtime
 * resets a binary semaphore afterwards (copy transference).
 */
static VkResult
sync_export_sync_file(struct vk_device *vkdev, struct vk_sync *vs, int *pfd)
{
   struct mali_csf_device *csf = csf_of(vkdev);
   struct mali_sync *s = to_sync(vs);
   VkResult result = VK_SUCCESS;

   pthread_mutex_lock(&csf->lock);
   if (csf->lost) {
      result = VK_ERROR_DEVICE_LOST;
   } else if (sync_is_signaled(csf, s, false)) {
      *pfd = -1;
   } else if (s->fd >= 0) {
      *pfd = fcntl(s->fd, F_DUPFD_CLOEXEC, 0);
      if (*pfd < 0)
         result = vk_errorf(vkdev, errno == EMFILE ? VK_ERROR_TOO_MANY_OBJECTS :
                                                     VK_ERROR_OUT_OF_HOST_MEMORY,
                            "cannot duplicate a sync file: %s", strerror(errno));
   } else if (s->submitted) {
      result = mali_sync_file_create(csf, 0, NULL, s->req, pfd);
   } else {
      /* Nothing will signal it: invalid usage. The blob answers the same. */
      result = vk_errorf(vkdev, VK_ERROR_OUT_OF_HOST_MEMORY,
                         "exporting a sync file from a sync nothing will signal");
   }
   pthread_mutex_unlock(&csf->lock);
   return result;
}

const struct vk_sync_type mali_sync_type = {
   .size = sizeof(struct mali_sync),
   .features = VK_SYNC_FEATURE_BINARY | VK_SYNC_FEATURE_GPU_WAIT |
               VK_SYNC_FEATURE_CPU_WAIT |
               VK_SYNC_FEATURE_CPU_RESET | VK_SYNC_FEATURE_CPU_SIGNAL |
               VK_SYNC_FEATURE_WAIT_ANY | VK_SYNC_FEATURE_WAIT_PENDING,
   .init = sync_init,
   .finish = sync_finish,
   .signal = sync_signal,
   .reset = sync_reset,
   .move = sync_move,
   .wait_many = sync_wait_many,
   .import_sync_file = sync_import_sync_file,
   .export_sync_file = sync_export_sync_file,
};
