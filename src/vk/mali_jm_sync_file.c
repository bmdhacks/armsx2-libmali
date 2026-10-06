/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Sync files (Linux sync_file fds, Android's native fences) on the job
 * manager, through soft atoms (the T820 blob's mechanism, JMS §6.1):
 *
 *  - GPU wait on a sync file: a FENCE_WAIT atom whose jc points at a
 *    base_fence holding the fd; the kernel takes its own reference at
 *    JOB_SUBMIT. mali_jm_queue.c puts the ones a submit waits for into a
 *    slot's order; the ones here (a sync file made from other sync files)
 *    stand alone.
 *  - A sync file for GPU work: a FENCE_TRIGGER atom whose base_fence names
 *    the device's timeline stream (STREAM_CREATE). The kernel creates the
 *    sync file during JOB_SUBMIT and writes its fd into the base_fence; it
 *    signals when the atom's dependencies have completed, with no help
 *    from user space (so no event thread is needed for release fences).
 *
 * All of it under mali_device::lock.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "mali_jm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "vk_log.h"

static VkResult
stream_get(struct mali_device *dev, int *out)
{
   struct mali_jm_device *jd = dev->jm;
   if (jd->stream_fd < 0) {
      char name[32];
      snprintf(name, sizeof(name), "libmali_%d", (int)getpid());
      enum mali_kbase_result r = mali_kbase_stream_create(jd->kb, name, &jd->stream_fd);
      if (r != MALI_KBASE_SUCCESS) {
         jd->stream_fd = -1;
         return vk_errorf(dev, VK_ERROR_OUT_OF_HOST_MEMORY,
                          "cannot create the sync-file timeline: %s",
                          mali_kbase_result_str(r));
      }
   }
   *out = jd->stream_fd;
   return VK_SUCCESS;
}

VkResult
MALI_PER_ARCH(sync_file_create)(struct mali_device *dev, uint32_t fd_count, const int *fds,
                                const uint64_t *req, int *out)
{
   struct mali_jm_device *jd = dev->jm;
   *out = -1;
   if (!jd || !jd->queue)
      return vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED, "sync files need the queue");
   if (dev->lost || jd->fault)
      return VK_ERROR_DEVICE_LOST;

   /* What is left to wait for: the submission's tracker, while it is in
    * flight, and the sync files that have not signalled. */
   const struct mali_jm_queue *q = jd->queue;
   const mali_jm_ref tracker =
      req && req[0] > q->completed_seq && mali_jm_ref_live(jd, req[1]) ? req[1] : 0;
   uint32_t pending = 0;
   for (uint32_t i = 0; i < fd_count; i++)
      pending += fds[i] >= 0 && !mali_sync_file_signaled(fds[i]);
   if (!tracker && !pending)
      return VK_SUCCESS;

   int stream_fd = -1;
   VkResult result = stream_get(dev, &stream_fd);
   if (result != VK_SUCCESS)
      return result;

   mali_jm_ref *deps = malloc((1 + fd_count) * sizeof(*deps));
   if (!deps)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   unsigned nd = 0;
   if (tracker)
      deps[nd++] = tracker;

   mali_jm_build_begin(dev, 0);
   for (uint32_t i = 0; i < fd_count && result == VK_SUCCESS; i++) {
      if (fds[i] < 0 || mali_sync_file_signaled(fds[i]))
         continue;
      const struct kb_fence f = {.fd = fds[i], .stream_fd = -1};
      mali_jm_ref w = mali_jm_build_atom(dev, MALI_JM_ATOM_FENCE_WAIT, MALI_JM_SLOT_NONE,
                                         KB_JM_REQ_SOFT_FENCE_WAIT, 0, &f, 0, 0, 0, &result);
      if (w) {
         deps[nd++] = w;
         jd->stats.fence_waits++;
      }
   }
   mali_jm_ref dep[2] = {0, 0};
   if (result == VK_SUCCESS)
      mali_jm_build_join_all(dev, deps, nd, dep, &result);
   if (result == VK_SUCCESS) {
      /* The kernel writes the new sync file's fd here during JOB_SUBMIT.
       * Its dependency on the submission's tracker is DATA (directly or
       * through join atoms): if the GPU work faulted, the kernel fails the
       * trigger with the same code and the sync file signals with an
       * error. On the sync files it waits for it is ORDER: an error there
       * is the other side's and is not passed on. */
      const struct kb_fence f = {.fd = -1, .stream_fd = stream_fd};
      if (mali_jm_build_atom(dev, MALI_JM_ATOM_FENCE_TRIGGER, MALI_JM_SLOT_NONE,
                             KB_JM_REQ_SOFT_FENCE_TRIGGER, 0, &f, 0, dep[0], dep[1], &result))
         jd->stats.fence_triggers++;
   }
   free(deps);

   int fd = -1;
   if (result != VK_SUCCESS) {
      mali_jm_build_abort(dev);
      return result;
   }
   result = mali_jm_build_end(dev, false, NULL, NULL, &fd);
   if (result != VK_SUCCESS) {
      if (fd >= 0)
         close(fd);
      return result;
   }
   if (fd < 0) {
      /* The kernel refused the trigger at submit (its event will say so,
       * as device loss). */
      return vk_errorf(dev, VK_ERROR_OUT_OF_HOST_MEMORY,
                       "the kernel did not create a sync file for the fence trigger");
   }
   *out = fd;
   return VK_SUCCESS;
}

void
mali_jm_sync_file_finish(struct mali_jm_device *jd)
{
   if (jd->stream_fd >= 0)
      close(jd->stream_fd);
   jd->stream_fd = -1;
}
