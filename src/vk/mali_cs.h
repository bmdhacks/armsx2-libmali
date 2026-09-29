/*
 * Copyright 2026 The libmali authors
 * SPDX-License-Identifier: MIT
 */

/*
 * What the command buffers and the queue agree on about the command
 * streams: the subqueues, the scoreboard entries, the CS register
 * layout, and the GPU-visible sync objects.
 *
 * The layout is panvk's for v11 (Mesa src/panfrost/vulkan/csf/
 * panvk_cmd_buffer.h): panvk gives the subqueue-to-stage mapping and the
 * scoreboard use, and its v11 register layout stays inside the 96
 * registers the G615 has. What differs from panvk is only what panthor
 * did for it and kbase does not: the queue writes the submit-time words
 * into the kernel rings itself (mali_queue.c).
 */

#ifndef MALI_CS_H
#define MALI_CS_H

#ifndef PAN_ARCH
#define PAN_ARCH MALI_PAN_ARCH
#endif

#include <stdint.h>

#include "genxml/gen_macros.h"
#include "genxml/cs_builder.h"

/* The three subqueues, one kernel CS queue each, bound at CSI slots 0..2 of
 * the device's queue group (panvk's split; the blob uses five). */
enum mali_subqueue {
   MALI_SUBQUEUE_VERTEX_TILER = 0,
   MALI_SUBQUEUE_FRAGMENT,
   MALI_SUBQUEUE_COMPUTE,
   MALI_SUBQUEUE_COUNT,
};

#define MALI_SUBQUEUE_ALL ((1u << MALI_SUBQUEUE_COUNT) - 1)

/* Scoreboard entries (panvk v11). Entry 0 tracks loads and stores and the
 * immediate cache flushes, entry 1 the deferred sync operations, entry 2
 * deferred flushes, entries 3.. the iterators (dispatches, draws). */
enum mali_sb {
   MALI_SB_LS = 0,
   MALI_SB_IMM_FLUSH = 0,
   MALI_SB_DEFERRED_SYNC = 1,
   MALI_SB_DEFERRED_FLUSH = 2,
   MALI_SB_ITER_START = 3,
};

#define MALI_SB_MASK(x) (1u << (x))

/*
 * Shared scoreboard entries (v11 SHARED_SB_INC / SHARED_SB_DEC, opcodes
 * 30/31). A second set of counters, shared by the streams of one queue
 * group: SHARED_SB_INC on one stream adds one to an entry once the
 * scoreboard entries it waits on are done (immediately, or deferred like
 * any asynchronous instruction); SHARED_SB_DEC on another stream blocks
 * that stream until the entry is non-zero and takes one off. kbase counts
 * a stream blocked on a decrement as idle (CS_REQ_IDLE_SHARED_SB_DEC,
 * CS_STATUS_BLOCKED_ON_SB_WAIT), so the wait is handled by the front end,
 * not by a firmware sync-object check.
 *
 * The blob hands out one entry per (source, destination) subqueue pair
 * from 16, starting at 4 (2 and 3 are fixed for its transfer subqueue,
 * 0 and 1 unused). Only the vertex/tiler -> fragment pass dependency
 * uses one here.
 */
#define MALI_SHARED_SB_VT_TO_FRAG 4

static inline void
mali_cs_shared_sb_inc(struct cs_builder *b, unsigned entry, struct cs_async_op async)
{
   cs_emit(b, SHARED_SB_INC, I) {
      I.shared_entry = entry;
      if (async.indirect)
         I.defer_mode = MALI_CS_SHARED_SB_INCREMENT_DEFER_MODE_DEFER_INDIRECT;
      else
         I.sb_mask = async.wait_mask;
   }
}

static inline void
mali_cs_shared_sb_dec(struct cs_builder *b, unsigned entry)
{
   cs_emit(b, SHARED_SB_DEC, I) {
      I.shared_entry = entry;
   }
}

/*
 * Registers. 0..63 are the RUN_* staging registers (RUN_COMPUTE uses 0..39,
 * RUN_IDVS 0..63 on v11), 66..83 scratch for command-buffer code. The next
 * ones keep their value across command buffers and submits:
 *
 *   84..89  progress sequence numbers: for each subqueue j, a 64-bit count
 *           of the barrier signals subqueue j made in earlier command
 *           buffers. Every subqueue keeps the same view (panvk's scheme,
 *           cs_progress_seqno_reg).
 *   90..91  the subqueue context address (struct mali_cs_subqueue_ctx).
 *
 * 92..95 are what cs_builder calls kernel registers: its chunk-chaining
 * JUMPs use them inside command buffers, and the queue uses them for the
 * words it writes into the ring between command buffers.
 */
#define MALI_CS_REG_SCRATCH_START 66
#define MALI_CS_REG_SCRATCH_END   83
/* Scratch 16..17 (r82:83) belong to the measurement layer's timestamp
 * address (src/measurement/timing.c); command code uses 0..15. */
#define MALI_CS_SCRATCH_MEASURE 16
#define MALI_CS_REG_PROGRESS_SEQNO_START 84
#define MALI_CS_REG_SUBQUEUE_CTX 90
#define MALI_CS_KERNEL_REGS 4
#define MALI_CS_RING_REG0 92   /* 92..95: ring-level temporaries */

/* A 64-bit GPU sync object (panvk_cs_sync64). error is set by sync
 * operations with error propagation when the stream has faulted. */
struct mali_cs_sync64 {
   uint64_t seqno;
   uint32_t error;
   uint32_t pad;
};

/*
 * Per-subqueue context in GPU memory, its address in MALI_CS_REG_SUBQUEUE_CTX
 * from the subqueue's first words on. Render passes add the tiler
 * fields' users; the queue fills them now.
 */
struct mali_cs_subqueue_ctx {
   uint64_t syncobjs;     /* struct mali_cs_sync64[MALI_SUBQUEUE_COUNT]:
                             barrier counters, one per subqueue */
   uint64_t done;         /* struct mali_cs_sync64[MALI_SUBQUEUE_COUNT]:
                             last submit completed on each subqueue */
   uint64_t tiler_heap;   /* the heap's Tiler Heap descriptor */
   uint64_t geom_buf;     /* geometry buffer, size code in the low bits */
   uint32_t last_error;
   uint32_t pad[7];
} __attribute__((aligned(64)));

/* Scoreboard facts of the device's CS interface. */
struct mali_cs_sb_info {
   uint8_t count;           /* entries per CS, at most 16 */
   uint16_t all_mask;
   uint8_t iter_count;
   uint16_t all_iters_mask;
};

static inline struct cs_index
mali_cs_scratch_reg_tuple(struct cs_builder *b, unsigned start, unsigned count)
{
   assert(MALI_CS_REG_SCRATCH_START + start + count <= MALI_CS_REG_SCRATCH_END + 1);
   return cs_reg_tuple(b, MALI_CS_REG_SCRATCH_START + start, count);
}

static inline struct cs_index
mali_cs_scratch_reg32(struct cs_builder *b, unsigned reg)
{
   return mali_cs_scratch_reg_tuple(b, reg, 1);
}

static inline struct cs_index
mali_cs_scratch_reg64(struct cs_builder *b, unsigned reg)
{
   assert(reg % 2 == 0);
   return mali_cs_scratch_reg_tuple(b, reg, 2);
}

static inline struct cs_index
mali_cs_progress_seqno_reg(struct cs_builder *b, enum mali_subqueue sq)
{
   return cs_reg64(b, MALI_CS_REG_PROGRESS_SEQNO_START + 2 * sq);
}

static inline struct cs_index
mali_cs_subqueue_ctx_reg(struct cs_builder *b)
{
   return cs_reg64(b, MALI_CS_REG_SUBQUEUE_CTX);
}

/* The resources each subqueue asks the firmware for (panvk). */
static inline uint32_t
mali_subqueue_resources(enum mali_subqueue sq)
{
   switch (sq) {
   case MALI_SUBQUEUE_VERTEX_TILER: return CS_IDVS_RES | CS_TILER_RES;
   case MALI_SUBQUEUE_FRAGMENT: return CS_FRAG_RES;
   default: return CS_COMPUTE_RES;
   }
}

#endif
