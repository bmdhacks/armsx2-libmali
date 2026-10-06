/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The job-manager (JM) back half of a VkDevice, for the Mali-G57 (arch v9).
 * Compiled only at PAN_ARCH 9; the CSF counterpart is mali_queue.h.
 *
 * What the command-buffer side and the queue agree on:
 *
 *  - A command buffer is a list of batches (struct mali_jm_batch). A batch
 *    is one vertex/tiler/compute ("vtc") job chain and zero or more
 *    fragment segments, each a fragment job chain. Each non-empty chain
 *    becomes one kbase atom at submit: the vtc chain on job slot 1
 *    (core_req CS|T|CF|COHERENT_GROUP), a fragment segment on slot 0
 *    (core_req FS). The chains are finished when recorded; submit only
 *    numbers atoms and fills in their dependencies.
 *  - Ordering the recorder asks for, per batch: vtc_after_frag (the vtc
 *    chain must wait for fragment work recorded earlier: in an earlier
 *    batch of the same command buffer, or anywhere before), frag_after_vtc
 *    (a batch with no vtc chain whose fragment work must wait for earlier
 *    vtc work), and heap_slot (the tiler heap region the batch tiles into;
 *    its vtc chain waits for the region's previous user's fragment work).
 *
 * The queue's rules (mali_jm_queue.c): each atom ORDER-depends on the
 * previous atom of its slot, so each slot runs in submission order, as a
 * CSF ring does; a fragment atom DATA-depends on its batch's vtc atom; a
 * vtc atom's one other dependency is the newest fragment-slot atom it must
 * wait for. Atoms in a slot order carry EVENT_COALESCE; each JOB_SUBMIT
 * ends in one that does not and that waits for the others, and the last
 * one of a submission (its "tracker") says "submission N and everything
 * before it is done". Completion is read on demand by whichever thread
 * needs it, with no event thread.
 *
 * Locking: everything here is under mali_device::lock.
 */

#ifndef MALI_JM_H
#define MALI_JM_H

#include <stdbool.h>
#include <stdint.h>

#include "util/u_dynarray.h"
#include "vk_command_buffer.h"
#include "vk_queue.h"
#include "vk_sync.h"

#include "kbase/kbase.h"

#if !defined(PAN_ARCH) || PAN_ARCH != 9
#error "mali_jm.h is the v9 (job manager) back half: build with PAN_ARCH=9"
#endif
#include "mali_arch.h"
#include "mali_vk.h"

/* Job slots (kbase_js_get_slot): fragment work runs on slot 0, everything
 * else we submit on slot 1. */
#define MALI_JM_SLOT_FRAG  0
#define MALI_JM_SLOT_VTC   1
#define MALI_JM_SLOT_COUNT 2
#define MALI_JM_SLOT_NONE  0xff

/* core_req of the two hardware atom kinds (the T820 blob's values, JC §7.3;
 * device check D6). */
#define MALI_JM_REQ_VTC  (KB_JM_REQ_CS | KB_JM_REQ_T | KB_JM_REQ_CF | KB_JM_REQ_COHERENT_GROUP)
#define MALI_JM_REQ_FRAG KB_JM_REQ_FS

/* ---------------------------------------------------------------------- */
/* Command buffers: batches and job chains                                 */

/* One job chain = one atom. Built by the command-buffer side (G8). */
struct mali_jm_chain {
   uint64_t first;          /* GPU VA of the first job; 0: empty */
   uint32_t *prev_next;     /* CPU pointer to the last header's Next (written only) */
   uint16_t index;          /* last job index (Mesa's scheme: unique, increasing) */
   uint16_t tiler_dep;      /* index of the last tiler-side job */
   uint32_t jobs;
};

/* A run of fragment jobs submitted as one fragment atom. */
struct mali_jm_frag_seg {
   struct mali_jm_chain chain;
};

/* mali_jm_batch::vtc_after_frag: 0 = no requirement; this value = every
 * fragment-slot atom submitted before this batch (the barrier's first
 * scope includes earlier submissions); otherwise 1 + the index of a batch
 * of the same command buffer whose fragment work must finish first. */
#define MALI_JM_AFTER_NONE    0
#define MALI_JM_AFTER_EARLIER 0xffff

struct mali_jm_batch {
   struct mali_jm_chain vtc;
   /* Fragment segments: frag_count entries of mali_jm_cmd::frags from
    * frag_first (indices, not a pointer: the array grows while recording). */
   uint32_t frag_first;
   uint32_t frag_count;
   int16_t heap_slot;       /* -1: the batch tiles nothing */
   uint16_t vtc_after_frag; /* MALI_JM_AFTER_* or 1 + batch index */
   bool frag_after_vtc;     /* no vtc chain of its own, but its fragment work
                               must follow the vtc work submitted before it */
   uint32_t passes, draws;
   uint64_t est_heap_bytes; /* batch closing (G9) */
};

/* Inside the v9 command buffer: what submit reads. */
struct mali_jm_cmd {
   struct util_dynarray batches; /* struct mali_jm_batch, closed batches */
   struct util_dynarray frags;   /* struct mali_jm_frag_seg */
};

/* The v9 command buffer. G7 holds only what submit needs; the recording
 * state (slabs, descriptor and draw state, the draw template) comes with
 * the command-buffer unit (G8). */
struct mali_cmd_buffer {
   struct vk_command_buffer vk;
   struct mali_device *dev;
   struct mali_jm_cmd jm;
};

extern const struct vk_command_buffer_ops MALI_PER_ARCH(cmd_buffer_ops);

static inline struct mali_jm_batch *
mali_jm_cmd_batches(struct mali_cmd_buffer *cmd, uint32_t *count)
{
   *count = util_dynarray_num_elements(&cmd->jm.batches, struct mali_jm_batch);
   return util_dynarray_begin(&cmd->jm.batches);
}

/* Append a closed batch (and its fragment segments). For the recorder and
 * for tests that build chains by hand. Returns false on allocation
 * failure. */
bool mali_jm_cmd_add_batch(struct mali_cmd_buffer *cmd, const struct mali_jm_batch *batch,
                           const struct mali_jm_frag_seg *frags, uint32_t frag_count);

/* ---------------------------------------------------------------------- */
/* The device and queue                                                    */

/* A reference to a submitted atom that stays meaningful after its number
 * is reused: (serial << 8) | number. Serials increase with every atom. 0
 * names nothing. */
typedef uint64_t mali_jm_ref;

static inline uint8_t
mali_jm_ref_num(mali_jm_ref r)
{
   return (uint8_t)r;
}

static inline uint64_t
mali_jm_ref_serial(mali_jm_ref r)
{
   return r >> 8;
}

enum mali_jm_atom_kind {
   MALI_JM_ATOM_VTC = 1,
   MALI_JM_ATOM_FRAG,
   MALI_JM_ATOM_JOIN,           /* dependency-only (core_req 0) */
   MALI_JM_ATOM_FENCE_WAIT,     /* soft: waits for an imported sync file */
   MALI_JM_ATOM_FENCE_TRIGGER,  /* soft: signals a sync file we hand out */
};

/* udata[1] of every atom: its kind, flags, and where it came from (for
 * the device-loss message). udata[0] is the submission number. */
#define MALI_JM_UDATA_KIND(u)   ((unsigned)((u) & 0xff))
#define MALI_JM_UDATA_TRACKER   (1ull << 8)   /* completes submission udata[0] */
#define MALI_JM_UDATA_TERMINAL  (1ull << 9)   /* last atom of a JOB_SUBMIT call */
#define MALI_JM_UDATA_BATCH(u)  ((unsigned)(((u) >> 16) & 0xffff))
#define MALI_JM_UDATA_CMD(u)    ((unsigned)(((u) >> 32) & 0xffff))

/* What the driver knows about one atom number. */
struct mali_jm_atom_info {
   mali_jm_ref ref;         /* the atom holding the number now (or last) */
   bool in_flight;          /* numbered and its event not read yet */
   uint8_t kind;            /* enum mali_jm_atom_kind */
   uint8_t slot;            /* MALI_JM_SLOT_*: in that slot's order, or NONE */
   /* Per slot, the newest atom in that slot's order this one waits for,
    * directly or through other atoms (itself, for its own slot). */
   mali_jm_ref cov[MALI_JM_SLOT_COUNT];
};

/* Tiler heap slots (g57-backend.md §8): the ring itself is G9's; the
 * queue keeps, per slot, the last fragment atom that read it, and the vtc
 * atom of the next batch using the slot waits for it. */
#define MALI_JM_HEAP_SLOTS_MAX 32

#define MALI_JM_EVENTS_PER_READ 32

struct mali_jm_queue;
struct mali_jm_build;

struct mali_jm_device {
   struct mali_device *dev;
   struct mali_kbase *kb;

   /* Atom numbers (1..255) and what each one is. A number is freed when
    * its event is read, never earlier, so a dependency on an atom whose
    * event has not been read names that atom. */
   struct mali_kbase_jm_atom_ids ids;
   unsigned free_ids;
   uint64_t next_serial;
   struct mali_jm_atom_info atoms[256];
   unsigned in_flight;

   /* One thread at a time blocks in poll() on the kbase fd (the
    * "reader"); the others sleep on mali_device::cond, which the reader
    * broadcasts. Non-blocking reads happen under the lock, when no reader
    * is polling. */
   bool reader;
   /* A thread is building and submitting atoms. Set while it may drop
    * the lock to wait for atom numbers, so no other thread numbers atoms
    * in between. */
   bool building;

   /* A fault read from an event that is not reported yet (the message
    * for mali_device_set_lost, which is called without the lock). */
   bool fault;
   char fault_msg[192];

   /* STREAM_CREATE's timeline fd for FENCE_TRIGGER (first use), or -1. */
   int stream_fd;

   /* The device's one queue (MALI_QUEUE_COUNT is 1). */
   struct mali_jm_queue *queue;

   /* Atoms of the JOB_SUBMIT being built (only while building). */
   struct mali_jm_build *build;

   struct {
      uint64_t atoms, job_submits, events, joins, partial_flushes, number_waits;
      uint64_t fence_waits, fence_triggers, fence_wait_errors;
   } stats;
};

struct mali_jm_queue {
   struct mali_device *dev;
   /* The newest atom in each slot's order. */
   mali_jm_ref last[MALI_JM_SLOT_COUNT];
   /* Per tiler heap slot, the last fragment atom that used it. */
   mali_jm_ref heap_last_frag[MALI_JM_HEAP_SLOTS_MAX];
   /* Submissions with atoms so far, the last one's tracker, and the
    * newest submission whose tracker event has been read. */
   uint64_t seq;
   mali_jm_ref tracker;
   uint64_t completed_seq;
};

VkResult MALI_PER_ARCH(device_init)(struct mali_device *dev);
void MALI_PER_ARCH(device_finish)(struct mali_device *dev);
VkResult MALI_PER_ARCH(queue_init)(struct mali_device *dev, struct mali_queue *queue);
void MALI_PER_ARCH(queue_finish)(struct mali_device *dev, struct mali_queue *queue);

/* vk_queue::driver_submit. */
VkResult MALI_PER_ARCH(queue_submit)(struct vk_queue *vkq, struct vk_queue_submit *submit);

/* vk_device::check_status. */
VkResult MALI_PER_ARCH(device_check_status)(struct vk_device *vkdev);

/* Shared with the CSF side (mali_queue.c): report device loss once, then
 * mark it and wake every waiter. Takes dev->lock; call without it. */
void mali_device_set_lost(struct mali_device *dev, const char *fmt, ...)
   __attribute__((format(printf, 2, 3)));

/* ---------------------------------------------------------------------- */
/* Atom building (mali_jm_queue.c), also used by mali_jm_sync_file.c       */

/* Is the atom r names still in flight (numbered, event not read)? */
static inline bool
mali_jm_ref_live(const struct mali_jm_device *jd, mali_jm_ref r)
{
   if (!r)
      return false;
   const struct mali_jm_atom_info *a = &jd->atoms[mali_jm_ref_num(r)];
   return a->in_flight && a->ref == r;
}

/* Start building atoms for one JOB_SUBMIT (more past the atom-number
 * limit), as submission seq (0 for atoms outside any submission). Waits
 * while another thread builds. With dev->lock held. */
void mali_jm_build_begin(struct mali_device *dev, uint64_t seq);

/*
 * Add one atom. slot: the slot order it joins (MALI_JM_SLOT_*: it
 * ORDER-follows nothing by itself, the caller passes the slot's last atom
 * as a dependency) or MALI_JM_SLOT_NONE. Atoms in a slot order carry
 * EVENT_COALESCE; others report their own event. deps (up to two, 0 for
 * none) are dropped when no longer in flight. For soft fence atoms, fence
 * is the base_fence the atom's jc points at (copied). Returns the atom's
 * reference, or 0 with *result set (device lost, or the kernel refused a
 * submit). May drop dev->lock to wait for atom numbers.
 */
mali_jm_ref mali_jm_build_atom(struct mali_device *dev, enum mali_jm_atom_kind kind,
                               uint8_t slot, uint32_t core_req, uint64_t jc,
                               const struct kb_fence *fence, uint64_t udata1,
                               mali_jm_ref dep0, uint8_t type0,
                               mali_jm_ref dep1, uint8_t type1, VkResult *result);

/* At most two dependencies standing for all of deps[0..n) (in flight or
 * not), joining pairs with dependency-only atoms while more than two are
 * in flight. Returns how many of out[] are set (0 with *result set on
 * failure, or when none is in flight). */
unsigned mali_jm_build_join_all(struct mali_device *dev, const mali_jm_ref *deps, unsigned n,
                                mali_jm_ref out[2], VkResult *result);

/*
 * Finish: the last atom built, or a join atom after it, becomes the
 * non-coalesced terminal of the JOB_SUBMIT (with tracker set: covering
 * every atom of both slot orders, flagged as the tracker of the build's
 * seq), and what is left is submitted. *terminal: its reference (0 when
 * nothing needs one); *built: whether the build numbered any atom;
 * *trigger_fd (may be NULL): the sync file of a FENCE_TRIGGER in the
 * build, -1 if none. Ends the build, also on failure.
 */
VkResult mali_jm_build_end(struct mali_device *dev, bool tracker, mali_jm_ref *terminal,
                           bool *built, int *trigger_fd);

/* End a build that failed: atoms not submitted yet are dropped. */
void mali_jm_build_abort(struct mali_device *dev);

/* With dev->lock held: read and process pending events without
 * blocking, unless another thread is blocked reading them. */
void mali_jm_drain_locked(struct mali_device *dev);

/* With dev->lock held: wait for events until abs_ns (CLOCK_MONOTONIC),
 * as the reader or on the condition variable. Returns early on any
 * event. */
void mali_jm_wait_locked(struct mali_device *dev, int64_t abs_ns);

/* ---------------------------------------------------------------------- */
/* Sync objects (mali_sync.c, built per arch)                              */

/*
 * The vk_sync type behind VkFence and binary VkSemaphore on v9. The
 * payload of a GPU signal is req[0], the submission number (reached when
 * the queue's completed_seq is at least that), and req[1], that
 * submission's tracker atom, which a same-queue wait or a sync-file export
 * depends on while it is in flight.
 */
#define MALI_SYNC_REQ_COUNT 2

struct mali_sync {
   struct vk_sync vk;
   /* Under mali_device::lock. */
   bool host_signaled;
   bool submitted;
   uint64_t req[MALI_SYNC_REQ_COUNT];
   int fd;                          /* an imported sync file, or -1 */
};

extern const struct vk_sync_type MALI_PER_ARCH(sync_type);

void MALI_PER_ARCH(sync_clear)(struct mali_sync *s);

/* Hooks mali_sync.c calls (see mali_queue.h for the CSF bodies). */
const char *MALI_PER_ARCH(queue_wait)(struct mali_device *dev);
bool MALI_PER_ARCH(queue_reached)(struct mali_device *dev, const uint64_t req[MALI_SYNC_REQ_COUNT]);

/* Host waits re-check every 20 ms even without an event, as on the
 * G615: the reader's poll() covers GPU completion, the slice covers host
 * signals that cannot interrupt it. */
static inline bool
MALI_PER_ARCH(queue_has_notifier)(struct mali_device *dev)
{
   return true;
}

static inline void
MALI_PER_ARCH(queue_sleep)(struct mali_device *dev, int64_t until_ns)
{
   mali_jm_wait_locked(dev, until_ns);
}

/* ---------------------------------------------------------------------- */
/* Sync files (mali_jm_sync_file.c)                                        */

/* Is the sync file signalled (poll without waiting)? Shared with the CSF
 * side (mali_sync_file.c); frontend-neutral. */
bool mali_sync_file_signaled(int fd);

/*
 * A new sync file that signals once every sync file in fds has signalled
 * and the submission req names has completed (req may be NULL): FENCE_WAIT
 * atoms for the fds, a FENCE_TRIGGER after them and the submission's
 * tracker. *out = -1 when nothing is left to wait for. With dev->lock
 * held.
 */
VkResult MALI_PER_ARCH(sync_file_create)(struct mali_device *dev, uint32_t fd_count,
                                         const int *fds, const uint64_t *req, int *out);

/* Close the device's sync-file stream. */
void mali_jm_sync_file_finish(struct mali_jm_device *jd);

#endif
