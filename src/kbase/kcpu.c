/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The kcpu queue and sync-file validation. The blob creates one kcpu queue
 * per queue context and enqueues one command per KCPU_QUEUE_ENQUEUE, with
 * a retry loop around "busy"; so do we.
 */

#include <string.h>
#include <time.h>

#include "kbase_priv.h"

enum mali_kbase_result
mali_kbase_kcpu_queue_create(struct mali_kbase *kb, uint8_t *id)
{
   struct kb_ioctl_kcpu_queue_new q;
   memset(&q, 0, sizeof(q));
   long ret = kb_ioctl(kb, KB_IOCTL_KCPU_QUEUE_CREATE, &q);
   if (ret < 0) {
      KB_LOG(kb, "KCPU_QUEUE_CREATE failed: %s", strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }
   *id = q.id;
   return MALI_KBASE_SUCCESS;
}

void
mali_kbase_kcpu_queue_destroy(struct mali_kbase *kb, uint8_t id)
{
   struct kb_ioctl_kcpu_queue_delete q;
   memset(&q, 0, sizeof(q));
   q.id = id;
   long ret = kb_ioctl(kb, KB_IOCTL_KCPU_QUEUE_DELETE, &q);
   if (ret < 0)
      KB_LOG(kb, "KCPU_QUEUE_DELETE %u failed: %s", id, strerror((int)-ret));
}

enum mali_kbase_result
mali_kbase_kcpu_enqueue(struct mali_kbase *kb, uint8_t id, const struct kb_kcpu_command *cmd)
{
   struct kb_ioctl_kcpu_queue_enqueue e;
   memset(&e, 0, sizeof(e));
   e.addr = (uint64_t)(uintptr_t)cmd;
   e.nr_commands = 1;
   e.id = id;

   /* 256 commands fit; the queue drains as the GPU and the sync files it
    * waits for make progress. */
   for (unsigned tries = 0;; tries++) {
      long ret = kb_ioctl(kb, KB_IOCTL_KCPU_QUEUE_ENQUEUE, &e);
      if (ret >= 0)
         return MALI_KBASE_SUCCESS;
      if (ret != -EBUSY || tries >= 20000) {
         KB_LOG(kb, "KCPU_QUEUE_ENQUEUE (queue %u, command %u) failed: %s", id,
                cmd->type, strerror((int)-ret));
         return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
      }
      const struct timespec ts = {.tv_nsec = 100 * 1000};
      nanosleep(&ts, NULL);
   }
}

enum mali_kbase_result
mali_kbase_fence_validate(struct mali_kbase *kb, int fd)
{
   struct kb_ioctl_fence_validate v = {.fd = fd};
   long ret = kb_ioctl(kb, KB_IOCTL_FENCE_VALIDATE, &v);
   if (ret < 0)
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   return MALI_KBASE_SUCCESS;
}
