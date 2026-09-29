/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The CSF side of a VkDevice and its VkQueue:
 *
 *  - struct mali_csf_device: per device. Scoreboard facts, the cache of
 *    command-buffer memory slabs, the sync-object lock and condition that
 *    host waits sleep on, the event thread that reads the kbase event
 *    channel, and device loss.
 *  - struct mali_csf_queue: per VkQueue. One kbase queue group with one CS
 *    queue (ring) per subqueue, the tiler heap, the GPU sync objects, and
 *    the submit counter.
 *  - struct mali_sync: the vk_sync type behind VkFence and binary
 *    VkSemaphore. A signal operation of submit N makes the sync "pending
 *    on N"; it is signalled once every subqueue has finished submit N's
 *    work, which each subqueue reports by writing N into its own "done"
 *    sync object in GPU memory.
 */

#ifndef MALI_QUEUE_H
#define MALI_QUEUE_H

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

#include "util/list.h"
#include "util/simple_mtx.h"
#include "vk_queue.h"
#include "vk_sync.h"

#include "kbase/kbase.h"

#include "mali_cs.h"

struct mali_device;
struct mali_queue;
struct mali_shader;

/* Internal compute shaders for copies and fills (mali_cmd_copy.c). */
enum mali_meta_shader {
   MALI_META_COPY_16,     /* 16 bytes per invocation */
   MALI_META_COPY_4,
   MALI_META_COPY_1,
   MALI_META_FILL_16,
   MALI_META_FILL_4,
   MALI_META_COUNT,
};

struct mali_csf_device {
   struct mali_device *dev;
   struct mali_kbase *kb;
   struct mali_cs_sb_info sb;
   uint8_t nr_registers;          /* CS work registers (96) */
   /* The vertex/tiler -> fragment pass dependency goes through a shared
    * scoreboard entry (mali_cs.h) instead of the vertex/tiler barrier
    * counter: gpu_features bit 3, as the blob decides. */
   bool shared_sb;

   /* Command-buffer memory: free 64 KiB slabs, shared by every command
    * pool of the device. */
   simple_mtx_t slab_lock;
   struct list_head free_slabs;
   unsigned free_slab_count;

   /* Guards every mali_sync's state and the device's loss state; waiters
    * sleep on cond, which submits, host signals and kernel events
    * broadcast. */
   pthread_mutex_t lock;
   pthread_cond_t cond;
   bool lost;
   uint32_t lost_reported;        /* vk_device_set_lost called (atomic) */

   /* The event thread: reads the kbase event channel, wakes waiters,
    * turns a queue-group error into device loss. */
   pthread_t thread;
   bool thread_running;
   int wake_fd;                   /* eventfd that stops the thread */
   uint8_t group_handle;          /* the group whose errors mean loss */
   bool group_valid;

   /* Internal shaders, compiled on first use. */
   simple_mtx_t meta_lock;
   struct mali_shader *meta[MALI_META_COUNT];
   /* Internal fragment shaders (mali_cmd_meta_gfx.c), by key. */
   void *meta_gfx;
   /* Blend shaders (mali_blend.c), by key. Under meta_lock. */
   void *blend_shaders;

   /* The device's one queue (MALI_QUEUE_COUNT is 1). */
   struct mali_csf_queue *queue;

   /* Sync-file import and export (mali_sync_file.c): the kcpu queue,
    * created on first use. Under lock. */
   struct mali_kcpu *kcpu;

   struct {
      uint64_t kernel_events;       /* EVENT notifications read */
      uint64_t group_errors;
      uint64_t waits;               /* host waits that had to sleep */
      uint64_t wakeups;             /* condition wake-ups while waiting */
      uint64_t submits;
   } stats;
};

/* Where the queue's sync memory keeps what (one page, CSF event memory). */
#define MALI_QUEUE_SYNCOBJS_OFFSET 0x000
#define MALI_QUEUE_DONE_OFFSET     0x040
/* The kcpu queue sets this sync64 to N once the N-th sync file a submit
 * waits for has signalled (mali_sync_file.c). */
#define MALI_QUEUE_FENCE_OFFSET    0x080
/* Per subqueue, a sync64 its ring adds one to after a command buffer when
 * a later command buffer of the same submit must follow it (mali_queue.c,
 * "Compute-subqueue ordering across command buffers"). */
#define MALI_QUEUE_HANDOFF_OFFSET  0x0a0
#define MALI_QUEUE_CTX_OFFSET(sq)  (0x100 + (sq) * sizeof(struct mali_cs_subqueue_ctx))

/* The tiler heap block: a 64 KiB geometry buffer, then the Tiler Heap
 * descriptor. */
#define MALI_TILER_GEOM_BUF_SIZE   (64 * 1024)
#define MALI_TILER_HEAP_DESC_OFFSET MALI_TILER_GEOM_BUF_SIZE
#define MALI_TILER_HEAP_CHUNK_SIZE (2 * 1024 * 1024)
#define MALI_TILER_HEAP_MAX_CHUNKS 552

#define MALI_RING_SIZE (64 * 1024)
#define MALI_RING_PRIORITY 8

struct mali_csf_queue {
   struct mali_device *dev;

   uint8_t group;
   bool group_created;

   struct mali_kbase_bo heap_block;
   uint64_t heap_ctx;               /* from CS_TILER_HEAP_INIT */
   bool heap_created;

   struct mali_kbase_bo sync_mem;
   struct mali_kbase_queue sq[MALI_SUBQUEUE_COUNT];

   /* Submits handed to the rings so far; submit N writes N into done[]. */
   uint64_t seqno;
   /* The last submit number each subqueue will write into its done slot;
    * a subqueue that had no work since keeps its old value. */
   uint64_t last_signal[MALI_SUBQUEUE_COUNT];

   /* Compute-subqueue ordering across command buffers (mali_queue.c):
    * done values of each subqueue that a stream relying on the compute
    * subqueue's earlier work (prior) or the compute subqueue's next work
    * (after_lazy) must wait for; ring_waited[k][j], the highest done value
    * of j ring k already waited for; handoff[j], signals ring j made on
    * its handoff counter so far. */
   uint64_t compute_prior[MALI_SUBQUEUE_COUNT];
   uint64_t compute_after_lazy[MALI_SUBQUEUE_COUNT];
   uint64_t ring_waited[MALI_SUBQUEUE_COUNT][MALI_SUBQUEUE_COUNT];
   uint64_t handoff[MALI_SUBQUEUE_COUNT];
};

static inline volatile struct mali_cs_sync64 *
mali_queue_done(const struct mali_csf_queue *q)
{
   return (volatile struct mali_cs_sync64 *)((uint8_t *)q->sync_mem.cpu +
                                             MALI_QUEUE_DONE_OFFSET);
}

/* The barrier counter of subqueue sq (struct mali_cs_sync64). The device
 * has one queue (MALI_QUEUE_COUNT), so command buffers use its address as
 * an immediate instead of loading it from the subqueue context. */
static inline uint64_t
mali_queue_syncobj_va(const struct mali_csf_queue *q, unsigned sq)
{
   return q->sync_mem.gpu_va + MALI_QUEUE_SYNCOBJS_OFFSET +
          sq * sizeof(struct mali_cs_sync64);
}

/* The tiler heap's Tiler Heap descriptor and the geometry buffer. */
static inline uint64_t
mali_queue_heap_desc_va(const struct mali_csf_queue *q)
{
   return q->heap_block.gpu_va + MALI_TILER_HEAP_DESC_OFFSET;
}

static inline uint64_t
mali_queue_geom_buf_va(const struct mali_csf_queue *q)
{
   return q->heap_block.gpu_va;
}

static inline uint64_t
mali_queue_done_va(const struct mali_csf_queue *q, unsigned sq)
{
   return q->sync_mem.gpu_va + MALI_QUEUE_DONE_OFFSET +
          sq * sizeof(struct mali_cs_sync64);
}

VkResult mali_csf_device_init(struct mali_device *dev);
void mali_csf_device_finish(struct mali_device *dev);

VkResult mali_csf_queue_init(struct mali_device *dev, struct mali_queue *queue);
void mali_csf_queue_finish(struct mali_device *dev, struct mali_queue *queue);

/* vk_queue::driver_submit. */
VkResult mali_queue_submit(struct vk_queue *vkq, struct vk_queue_submit *submit);

/* vk_device::check_status: DEVICE_LOST after a queue-group error or a
 * faulted sync object. */
VkResult mali_device_check_status(struct vk_device *vkdev);

/* Report device loss to the runtime and the log (once), then mark it and
 * wake every waiter. Takes csf->lock; call without it. */
void mali_csf_set_lost(struct mali_csf_device *csf, const char *fmt, ...)
   __attribute__((format(printf, 2, 3)));

/* ---------------------------------------------------------------------- */
/* Sync objects                                                            */

struct mali_sync {
   struct vk_sync vk;
   /* Under mali_csf_device::lock. */
   bool host_signaled;
   bool submitted;                  /* a submit will signal it */
   uint64_t req[MALI_SUBQUEUE_COUNT];   /* ... when done[i] >= req[i] for all i */
   int fd;                          /* an imported sync file, or -1 */
};

extern const struct vk_sync_type mali_sync_type;

/* Drop a sync's payload (close its sync file). With csf->lock held. */
void mali_sync_clear(struct mali_sync *s);

/* ---------------------------------------------------------------------- */
/* Sync files (mali_sync_file.c)                                           */

/*
 * GPU wait on a sync file: the kcpu queue waits for fd, then sets the
 * queue's fence sync64 to *value; a ring waits for "fence > *value - 1".
 * The kernel takes its own reference, so fd may be closed afterwards.
 * With csf->lock held.
 */
VkResult mali_sync_file_gpu_wait(struct mali_csf_device *csf, int fd, uint64_t *value);

/*
 * A new sync file that signals once every sync file in fds has signalled
 * and every subqueue has reached req (req may be NULL). *out = -1 when
 * nothing is left to wait for. With csf->lock held.
 */
VkResult mali_sync_file_create(struct mali_csf_device *csf, uint32_t fd_count,
                               const int *fds, const uint64_t *req, int *out);

/* Delete the kcpu queue (before the queue memory it points at goes). */
void mali_sync_file_finish(struct mali_csf_device *csf);

/* Is the sync file signalled (poll without waiting)? */
bool mali_sync_file_signaled(int fd);

/* The queue's fence sync64 GPU address. */
static inline uint64_t
mali_queue_handoff_va(const struct mali_csf_queue *q, unsigned sq)
{
   return q->sync_mem.gpu_va + MALI_QUEUE_HANDOFF_OFFSET + sq * sizeof(struct mali_cs_sync64);
}

static inline uint64_t
mali_queue_fence_va(const struct mali_csf_queue *q)
{
   return q->sync_mem.gpu_va + MALI_QUEUE_FENCE_OFFSET;
}

/* With csf->lock held: has every subqueue reached req? Checks the done
 * slots' error words too (a fault there is device loss). */
bool mali_csf_reached(struct mali_csf_device *csf, const uint64_t req[MALI_SUBQUEUE_COUNT]);

#endif
