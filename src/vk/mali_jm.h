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

#include "util/simple_mtx.h"
#include "util/u_dynarray.h"
#include "vk_command_buffer.h"
#include "vk_queue.h"
#include "vk_sync.h"

#include "kbase/kbase.h"

#if !defined(PAN_ARCH) || PAN_ARCH != 9
#error "mali_jm.h is the v9 (job manager) back half: build with PAN_ARCH=9"
#endif
#include "genxml/gen_macros.h"

#include "mali_arch.h"
#include "mali_cmd_gfx.h"
#include "mali_shader.h"
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

/* mali_jm_chain::pending: what the chain's next job has to wait for (an
 * in-chain barrier, g57-backend.md §5.2). BARRIER: the next job sets the
 * header's Barrier bit (it starts after every earlier job of the chain).
 * FLUSH: a Cache Flush job with Barrier and "Invalidate Shader Core Other"
 * goes first, for reads through a shader core's read-only caches (texture,
 * attribute) of what earlier jobs wrote. */
#define MALI_JM_PENDING_BARRIER (1u << 0)
#define MALI_JM_PENDING_FLUSH   (1u << 1)

/* One job chain = one atom. */
struct mali_jm_chain {
   uint64_t first;          /* GPU VA of the first job; 0: empty */
   uint32_t *prev_next;     /* CPU pointer to the last header's Next (written only) */
   uint16_t index;          /* last job index (Mesa's scheme: unique, increasing) */
   uint16_t tiler_dep;      /* index of the last tiler-side job */
   uint32_t jobs;
   uint8_t pending;         /* MALI_JM_PENDING_* */
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
/* Every earlier fragment-slot atom of a batch with fragment-side transfer
 * work (mali_jm_batch::frag_xfer): what a barrier whose fragment-slot
 * source stages are only transfer stages waits for. */
#define MALI_JM_AFTER_EARLIER_XFER 0xfffe

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
   bool frag_xfer;          /* its fragment segments carry transfer work
                               (MALI_JM_AFTER_EARLIER_XFER) */
   uint32_t passes, draws;
   uint64_t est_heap_bytes; /* estimated tiler heap use, for batch closing */
   uint64_t heap_desc;      /* the Tiler Heap descriptor every pass of the
                               batch tiles through (0 until one tiles) */
};

/*
 * Batch limits (g57-backend.md §5.3). A batch is closed before a job that
 * would take its vtc chain past MALI_JM_BATCH_MAX_JOBS jobs, outside a
 * render pass (a pass's tiling stays in one batch; the render pass side
 * checks the limit when a pass begins). The limit keeps a chain's GPU time
 * far below the T820 kernel's 1 s hard-stop (soft-stops resume a chain at
 * a job boundary, so only one long job is at risk, design §9.5) and the
 * 16-bit job index far from wrapping. 1024 vtc jobs are 3-6 ms of tiling
 * at ARMSX2's draw sizes (device check: D13 tunes it).
 */
#define MALI_JM_BATCH_MAX_JOBS   1024
#define MALI_JM_BATCH_MAX_PASSES 16

/* mali_jm_cmd::end_req: ordering a barrier recorded in this command buffer
 * asks of work after it, in later command buffers (the queue carries it to
 * the next atom of the slot): the next vtc atom waits for the newest
 * fragment atom, the next fragment atom of a batch without a vtc chain
 * waits for the newest vtc atom. */
#define MALI_JM_REQ_VTC_AFTER_FRAG      (1u << 0)
#define MALI_JM_REQ_FRAG_AFTER_VTC      (1u << 1)
#define MALI_JM_REQ_VTC_AFTER_FRAG_XFER (1u << 2)

/* A GPU-written range to restore before a command buffer runs again
 * (mali_jm_cmd::resets): size bytes at dst, from data_off in
 * mali_jm_cmd::reset_data, or zero when data_off is UINT32_MAX. */
struct mali_jm_reset {
   void *dst;
   uint32_t data_off;
   uint32_t size;
};

/* Inside the v9 command buffer: what submit reads, and the open batch. */
struct mali_jm_cmd {
   struct util_dynarray batches; /* struct mali_jm_batch, closed batches */
   struct util_dynarray frags;   /* struct mali_jm_frag_seg */

   /* The batch being recorded (when `open`); its fragment segments are
    * already in frags, from cur.frag_first. */
   struct mali_jm_batch cur;
   bool open;
   /* Barrier requirements recorded and not yet applied to a job of the
    * slot they order (mali_jm_cmd_buffer.c), and what is left of them at
    * the end of the command buffer (MALI_JM_REQ_*), for the queue. */
   uint16_t req;
   uint8_t end_req;

   /* Re-submission (g57-backend.md §5.2): command buffers recorded
    * without ONE_TIME_SUBMIT note every GPU-written word, and submit
    * restores them before every run after the first. */
   bool resubmit;
   bool submitted;
   uint64_t last_seq;            /* the submission that ran it last */
   struct util_dynarray resets;  /* struct mali_jm_reset */
   struct util_dynarray reset_data;
};

/*
 * The v9 command buffer: what submit reads (jm), and the recording state
 * the builders shared with the v11 back half read, under the same member
 * names as the v11 struct (mali_cmd_state.h lists them).
 */
struct mali_cmd_buffer {
   struct vk_command_buffer vk;
   struct mali_device *dev;
   struct mali_jm_cmd jm;

   /* GPU memory: slabs in use, the one being filled (mali_cmd_alloc). */
   struct list_head slabs;
   struct mali_cmd_slab *cur;
   uint64_t cur_offset;
   uint8_t *cur_cpu;
   uint64_t cur_gpu;

   uint8_t push_constants[MALI_MAX_PUSH_CONSTANTS_SIZE];

   /* Storage behind vk.dynamic_graphics_state.vi / .ms.sample_locations. */
   struct vk_vertex_input_state dyn_vi;
   struct vk_sample_locations_state dyn_sl;

   struct {
      const struct mali_shader *shader;
      struct mali_desc_state desc;
   } compute;

   struct {
      struct mali_desc_state desc;
      /* regs/regs_valid are the v11 register record; unused here. */
      struct mali_gfx_draw_state draw;
      struct mali_render_state render;
   } gfx;

   /* Thread-local storage shared by every job of the command buffer. */
   struct {
      uint32_t size;
      uint64_t gpu;
   } tls;

   uint32_t dispatches;
   uint32_t draws;
   uint32_t passes;
};

VK_DEFINE_HANDLE_CASTS(mali_cmd_buffer, vk.base, VkCommandBuffer,
                       VK_OBJECT_TYPE_COMMAND_BUFFER)

/* size bytes of CPU-mapped, CPU-uncached GPU memory at a multiple of align
 * (a power of two), alive until the command buffer is reset; {0} with the
 * command buffer's error set on failure. The same as the v11 one
 * (mali_cmd_buffer.h). */
static inline struct mali_ptr
mali_cmd_alloc(struct mali_cmd_buffer *cmd, uint64_t size, uint64_t align)
{
   const uint64_t off = (cmd->cur_offset + align - 1) & ~(align - 1);
   if (likely(cmd->cur_cpu && size && size <= MALI_CMD_SLAB_SIZE / 2 &&
              off + size <= MALI_CMD_SLAB_SIZE)) {
      cmd->cur_offset = off + size;
      return (struct mali_ptr){cmd->cur_cpu + off, cmd->cur_gpu + off};
   }
   return MALI_PER_ARCH(cmd_alloc_slow)(cmd, size, align);
}

/* vkCmdBindPipeline, and the pipeline ops' bind (mali_cmd_state.c). */
void MALI_PER_ARCH(cmd_bind_pipeline)(struct mali_cmd_buffer *cmd, struct vk_pipeline *pipeline);

extern const struct vk_command_buffer_ops MALI_PER_ARCH(cmd_buffer_ops);

static inline struct mali_jm_batch *
mali_jm_cmd_batches(struct mali_cmd_buffer *cmd, uint32_t *count)
{
   *count = util_dynarray_num_elements(&cmd->jm.batches, struct mali_jm_batch);
   return util_dynarray_begin(&cmd->jm.batches);
}

/* Append a closed batch (and its fragment segments) behind the batches
 * recorded so far. For tests that build chains by hand; the open batch, if
 * any, is closed first. Returns false on allocation failure. */
bool mali_jm_cmd_add_batch(struct mali_cmd_buffer *cmd, const struct mali_jm_batch *batch,
                           const struct mali_jm_frag_seg *frags, uint32_t frag_count);

/* ---------------------------------------------------------------------- */
/* Recording job chains (mali_jm_cmd_buffer.c)                             */

/* The open batch, opened if there is none. NULL on allocation failure
 * (the command buffer has the error). */
struct mali_jm_batch *mali_jm_cmd_batch(struct mali_cmd_buffer *cmd);

/* Close the open batch: it joins the closed batches if it has any job,
 * with the barrier requirements recorded for its work. */
void mali_jm_cmd_batch_close(struct mali_cmd_buffer *cmd);

/* The open batch's vtc chain, for a job outside a render pass: closes a
 * full batch first (MALI_JM_BATCH_MAX_JOBS). NULL on failure. */
struct mali_jm_chain *mali_jm_cmd_vtc(struct mali_cmd_buffer *cmd);

/* The same for a transfer done by compute (buffer and image copies, fills,
 * updates): it also waits for what barriers whose destination stages are
 * only transfer stages asked for. Plain vtc jobs (dispatches, draws) are
 * not in those barriers' second scope, so such a barrier does not close a
 * batch at the next draw. */
struct mali_jm_chain *mali_jm_cmd_vtc_transfer(struct mali_cmd_buffer *cmd);

/* Would a transfer job on the vtc chain close the open batch now (a
 * recorded barrier makes it wait for this batch's fragment work)? An
 * image copy then goes to the fragment side instead (mali_cmd_image.c). */
bool mali_jm_cmd_transfer_needs_frag(struct mali_cmd_buffer *cmd);

/* The open batch's current fragment segment, started if the batch has
 * none; with new_segment, always a new one (a new fragment atom). Applies
 * the barriers recorded for fragment work. NULL on failure. The vtc
 * variant above applies those for vtc work: call it before the first job
 * of a render pass's tiling (a batch can still close there, since nothing
 * of the pass is in it yet), and before each job outside a pass. */
struct mali_jm_chain *mali_jm_cmd_frag(struct mali_cmd_buffer *cmd, bool new_segment);

/* The open batch's fragment work includes a transfer (a fragment-side
 * copy, blit, resolve or image clear): barriers from transfer stages wait
 * for it. */
void mali_jm_cmd_mark_frag_transfer(struct mali_cmd_buffer *cmd);

/*
 * Append a job to chain c: size bytes (a v9 job, its header included) at
 * 128-byte alignment from command memory, with the header written (type,
 * the next index, Barrier from the chain's pending barrier or `barrier`,
 * Dependency 1 = dep1, Dependency 2 = the previous tiler-side job for a
 * tiler-side job, Next = 0) and linked behind the chain's last job. A
 * pending Cache Flush job goes in first. The caller writes the payload,
 * every byte of it (slabs are recycled without clearing). {0} on failure,
 * with the command buffer's error set.
 */
struct mali_ptr mali_jm_cmd_add_job(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c,
                                    enum mali_job_type type, unsigned size, bool barrier,
                                    uint16_t dep1);

/* A Write Value job on chain c (timestamps, availability, tests). */
bool mali_jm_cmd_write_value(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c,
                             enum mali_write_value_type type, uint64_t addr, uint64_t value,
                             bool barrier);

/* Note size bytes at dst as GPU-written: restored from tmpl (or zeroed,
 * tmpl NULL) before every run after the first. Free for ONE_TIME_SUBMIT
 * command buffers. */
void mali_jm_cmd_note_reset(struct mali_cmd_buffer *cmd, void *dst, const void *tmpl,
                            uint32_t size);

/*
 * The JM barrier: what VkPipelineStageFlags2/VkAccessFlags2 on each side
 * require of the job chains (g57-backend.md §5.3): an in-chain barrier on
 * the vtc or fragment chain, a closed batch with the next one waiting for
 * the fragment atom, or a requirement the queue resolves at submit.
 * Outside a render pass; vkCmdPipelineBarrier2 handles the in-pass case.
 */
void mali_jm_cmd_barrier(struct mali_cmd_buffer *cmd, VkPipelineStageFlags2 src_stages,
                         VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stages,
                         VkAccessFlags2 dst_access);

/* Before a submission runs cmd: restore what earlier runs changed. With
 * dev->lock held, the command buffer not in flight. */
void mali_jm_cmd_prepare_submit(struct mali_cmd_buffer *cmd);

/* ---------------------------------------------------------------------- */
/* Render passes (mali_jm_cmd_render.c) and the tiler heap (mali_jm_heap.c) */

/*
 * What a draw (mali_jm_cmd_draw.c) uses of the pass, after
 * MALI_PER_ARCH(cmd_render_tiler) has returned true for it:
 *  - the chain its jobs go into: mali_jm_cmd_pass_vtc (the open batch's
 *    vtc chain; the batch cannot close inside a pass, and the barriers
 *    owed to the pass's vtc work were applied by cmd_render_tiler, so a
 *    draw takes the chain as it is);
 *  - one Tiler Context per layer: mali_jm_pass_tiler(r, layer), layers
 *    0 .. r->td_count - 1 (MALI_LAYERS_PER_TILER_CTX is 1 on v9);
 *  - the pass's Local Storage descriptor: r->tsd (filled at the end of the
 *    pass from r->tls_size, which the draw raises to its shaders' need);
 *  - the batch's heap estimate: mali_jm_cmd_heap_use(cmd, bytes) per draw
 *    (vertices x instances x (packet stride + 16) plus the polygon list
 *    estimate); the next pass's begin closes the
 *    batch past half a heap slot.
 */
static inline struct mali_jm_chain *
mali_jm_cmd_pass_vtc(struct mali_cmd_buffer *cmd)
{
   assert(cmd->jm.open && cmd->gfx.render.tiler);
   return &cmd->jm.cur.vtc;
}

static inline uint64_t
mali_jm_pass_tiler(const struct mali_render_state *r, uint32_t layer)
{
   assert(layer < r->td_count);
   return r->tiler + (uint64_t)layer * pan_size(TILER_CONTEXT);
}

static inline void
mali_jm_cmd_heap_use(struct mali_cmd_buffer *cmd, uint64_t bytes)
{
   cmd->jm.cur.est_heap_bytes += bytes;
}

struct mali_jm_device;

/* The ring's lock; slots are allocated on first use. */
void mali_jm_heap_init(struct mali_jm_device *jd);
/* Frees every slot. The GPU must be idle. */
void mali_jm_heap_finish(struct mali_jm_device *jd);

/* Give the open batch b a heap slot (the next in the ring, allocated if it
 * is new) and its Tiler Heap descriptor in command memory (b->heap_slot,
 * b->heap_desc). False on failure, with the command buffer's error set. */
bool mali_jm_heap_take(struct mali_cmd_buffer *cmd, struct mali_jm_batch *b);

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

/*
 * The tiler heap ring (mali_jm_heap.c). On v9 the tiler
 * allocates polygon lists and vertex packets (varyings) from the heap, and
 * the kernel does not order atoms that share memory, so a heap region is
 * reused only after the fragment work that reads it. The ring has
 * MALI_JM_HEAP_SLOTS regions; a batch that tiles takes the next one when
 * its first pass starts tiling, and every pass and layer of the batch tiles
 * through one Tiler Heap descriptor in its command memory. The queue keeps,
 * per slot, the last fragment atom that read it, and the vtc atom of the
 * next batch using the slot waits for it.
 *
 * Each slot is a separate growable allocation (a GROW_ON_GPF fault commits
 * every page up to the faulting one, so windows inside one region would
 * commit the gaps between them): GPU read/write, no CPU access,
 * TILER_ALIGN_TOP, 2 MiB committed, 2 MiB growth steps, 256 MiB of VA (the
 * T820 blob's big JIT heaps are 360 MiB).
 */
#define MALI_JM_HEAP_SLOTS_MAX 32
#define MALI_JM_HEAP_SLOTS 4
#define MALI_JM_HEAP_CHUNK (2ull << 20)
#define MALI_JM_HEAP_SLOT_SIZE (256ull << 20)

struct mali_jm_heap {
   simple_mtx_t lock;
   uint32_t next;           /* the slot the next batch takes */
   struct mali_kbase_bo bo[MALI_JM_HEAP_SLOTS];
};

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

   /* Tiler heap regions; their own lock (recording threads take slots). */
   struct mali_jm_heap heap;

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
   /* The newest fragment atom of a batch with fragment-side transfer
    * work (MALI_JM_AFTER_EARLIER_XFER). */
   mali_jm_ref last_frag_xfer;
   /* Barrier requirements (MALI_JM_REQ_*) recorded at the end of a
    * command buffer and not yet met by a later atom. */
   uint8_t carry;
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
