/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The v9 (job manager) command buffer: command memory, batches and job
 * chains, job writing and linking, barriers, begin/end, and what a command
 * buffer that runs more than once needs restored.
 *
 * Every vkCmd* writes finished v9 jobs when it is recorded (D-3). A command
 * buffer is a list of batches (mali_jm.h); a batch is one vertex/tiler/
 * compute ("vtc") job chain, which runs as one atom on job slot 1, and
 * fragment segments, each a fragment chain that runs as one atom on slot 0
 * after the vtc atom. Submit (mali_jm_queue.c) only numbers the atoms and
 * fills in their dependencies.
 *
 * Linking follows Mesa (pan_jc.h pan_jc_add_job): every job of a chain
 * takes the next index (unique and increasing), a tiler-side job depends
 * on the previous tiler-side job through Dependency 2, and the previous
 * job's Next is written once the next job exists. All of it is stores to
 * write-combined command memory; nothing is read back. The queue's
 * "dsb st" before JOB_SUBMIT makes them visible.
 *
 * Barriers are applied lazily, per job slot (g57-backend.md §5.3): a
 * barrier records what the next vtc job and the next fragment job have to
 * wait for, and the requirement is resolved when such a job is recorded:
 *
 *  - vtc after vtc: the Barrier bit on the next job of the vtc chain, after
 *    a Cache Flush job ("Invalidate Shader Core Other") when the reads go
 *    through a shader core's read-only caches. Nothing when the chain is
 *    still empty: atoms of one slot run in order, and the kernel cleans
 *    and invalidates the caches between atoms;
 *  - vtc after fragment: a vtc chain runs before its batch's fragment
 *    chains, so the batch is closed if it has fragment work, and the next
 *    batch's vtc atom waits for the newest fragment atom; a batch without
 *    fragment work waits for the newest earlier fragment atom instead,
 *    which the queue resolves at submit. A barrier whose fragment-slot
 *    source stages are only transfer stages asks only for fragment atoms
 *    that carried transfer work (fragment-side copies, blits, clears), so
 *    an upload barrier does not hold the next frame's tiling behind the
 *    previous frame's fragment work. A barrier whose vtc-slot destination
 *    stages are only transfer stages asks this only of the next transfer
 *    done by compute (mali_jm_cmd_vtc_transfer), not of draws and
 *    dispatches, which are outside its second scope: a "colour output ->
 *    transfer" barrier for a fragment-side image copy then does not close
 *    the batch at the next pass's tiling;
 *  - fragment after vtc: free inside a batch (its fragment atoms depend on
 *    its vtc atom); a batch without a vtc chain waits for the newest vtc
 *    atom;
 *  - fragment after fragment: the Barrier bit (and Cache Flush job) on the
 *    next job of the current fragment segment.
 *
 * What is still owed when the command buffer ends goes to the queue
 * (mali_jm_cmd::end_req), which applies it to the next atoms of later
 * command buffers, as a barrier's second scope requires.
 */

#include "mali_cmd_state.h"

#include <stdalign.h>
#include <stdlib.h>
#include <string.h>

#include "util/log.h"
#include "vk_alloc.h"
#include "vk_command_pool.h"
#include "vk_log.h"
#include "vk_synchronization.h"

#include "mali_measure.h"

/* ---------------------------------------------------------------------- */
/* Memory                                                                  */

/* Free slabs the device keeps for reuse, as on v11 (mali_cmd_buffer.c):
 * the same device list, freed at device destruction by
 * mali_cmd_slabs_finish. */
#define MALI_CMD_FREE_SLABS_MAX 512

/* A slab of command memory: CPU and GPU read/write (the GPU writes job
 * headers back), CPU-uncached, written and never read by the CPU. */
static struct mali_cmd_slab *
slab_new(struct mali_device *dev, uint64_t size, bool recycle)
{
   if (recycle) {
      simple_mtx_lock(&dev->slab_lock);
      struct mali_cmd_slab *s = NULL;
      if (!list_is_empty(&dev->free_slabs)) {
         s = list_first_entry(&dev->free_slabs, struct mali_cmd_slab, link);
         list_del(&s->link);
         dev->free_slab_count--;
      }
      simple_mtx_unlock(&dev->slab_lock);
      if (s)
         return s;
   }

   struct mali_cmd_slab *s = calloc(1, sizeof(*s));
   if (!s)
      return NULL;
   const struct mali_kbase_alloc_info ai = {
      .size = size,
      .flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_DEVICE_UNCACHED),
      .mem_class = MALI_KBASE_MEM_CLASS_COMMAND_ALLOCATOR,
   };
   if (mali_kbase_alloc(dev->kbase, &ai, &s->bo) != MALI_KBASE_SUCCESS) {
      free(s);
      return NULL;
   }
   s->recycle = recycle;
   return s;
}

static void
slab_release(struct mali_device *dev, struct mali_cmd_slab *s)
{
   if (s->recycle) {
      simple_mtx_lock(&dev->slab_lock);
      if (dev->free_slab_count < MALI_CMD_FREE_SLABS_MAX) {
         list_add(&s->link, &dev->free_slabs);
         dev->free_slab_count++;
         s = NULL;
      }
      simple_mtx_unlock(&dev->slab_lock);
      if (!s)
         return;
   }
   mali_kbase_free(dev->kbase, &s->bo);
   free(s);
}

/* Command memory, and the TLS/WLS buffers the builders put on the same
 * list (recycle false). */
static void
release_memory(struct mali_cmd_buffer *cmd)
{
   list_for_each_entry_safe(struct mali_cmd_slab, s, &cmd->slabs, link) {
      list_del(&s->link);
      slab_release(cmd->dev, s);
   }
   cmd->cur = NULL;
   cmd->cur_offset = 0;
   cmd->cur_cpu = NULL;
   cmd->cur_gpu = 0;
   cmd->tls.size = 0;
   cmd->tls.gpu = 0;
}

struct mali_ptr
MALI_PER_ARCH(cmd_alloc_slow)(struct mali_cmd_buffer *cmd, uint64_t size, uint64_t align)
{
   assert(util_is_power_of_two_nonzero64(align) && align <= 4096);
   size = MAX2(size, 1);

   if (size > MALI_CMD_SLAB_SIZE / 2) {
      struct mali_cmd_slab *s = slab_new(cmd->dev, size, false);
      if (!s)
         goto oom;
      list_addtail(&s->link, &cmd->slabs);
      return (struct mali_ptr){s->bo.cpu, s->bo.gpu_va};
   }

   uint64_t off = ALIGN_POT(cmd->cur_offset, align);
   if (!cmd->cur || off + size > MALI_CMD_SLAB_SIZE) {
      struct mali_cmd_slab *s = slab_new(cmd->dev, MALI_CMD_SLAB_SIZE, true);
      if (!s)
         goto oom;
      list_addtail(&s->link, &cmd->slabs);
      cmd->cur = s;
      cmd->cur_cpu = s->bo.cpu;
      cmd->cur_gpu = s->bo.gpu_va;
      off = 0;
   }
   cmd->cur_offset = off + size;
   return (struct mali_ptr){cmd->cur_cpu + off, cmd->cur_gpu + off};

oom:
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return (struct mali_ptr){0};
}

/* ---------------------------------------------------------------------- */
/* Re-submission                                                           */

void
mali_jm_cmd_note_reset(struct mali_cmd_buffer *cmd, void *dst, const void *tmpl, uint32_t size)
{
   if (!cmd->jm.resubmit)
      return;
   struct mali_jm_reset r = {.dst = dst, .data_off = UINT32_MAX, .size = size};
   if (tmpl) {
      r.data_off = util_dynarray_num_elements(&cmd->jm.reset_data, uint8_t);
      void *d = util_dynarray_grow_bytes(&cmd->jm.reset_data, 1, size);
      if (!d) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return;
      }
      memcpy(d, tmpl, size);
   }
   struct mali_jm_reset *slot = util_dynarray_grow(&cmd->jm.resets, struct mali_jm_reset, 1);
   if (slot)
      *slot = r;
   else
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
}

/* The job manager writes a job's Exception Status, First Incomplete Task
 * and Fault Pointer (header words 0-3) when it runs the job, and refuses
 * to run a job whose status is not zero; the rest of the header is ours
 * (panvk JM and the T820 blob reset the same words, JC §7.4). Other
 * GPU-written words (tiler contexts, heap descriptors, the draw
 * descriptor's Vertex array words) are noted by the code that writes
 * them. */
void
mali_jm_cmd_prepare_submit(struct mali_cmd_buffer *cmd)
{
   if (!cmd->jm.resubmit || !cmd->jm.submitted)
      return;
   const uint8_t *data = util_dynarray_begin(&cmd->jm.reset_data);
   util_dynarray_foreach(&cmd->jm.resets, struct mali_jm_reset, r) {
      if (r->data_off == UINT32_MAX)
         memset(r->dst, 0, r->size);
      else
         memcpy(r->dst, data + r->data_off, r->size);
   }
}

/* ---------------------------------------------------------------------- */
/* Jobs                                                                    */

static inline bool
job_is_tiler_side(enum mali_job_type type)
{
   return type == MALI_JOB_TYPE_MALLOC_VERTEX || type == MALI_JOB_TYPE_FULLSCREEN ||
          type == MALI_JOB_TYPE_TILER;
}

/* Header and link of a job at job (size bytes, from command memory). */
static inline void
link_job(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c, struct mali_ptr job,
         enum mali_job_type type, bool barrier, uint16_t dep1)
{
   const bool tiler_side = job_is_tiler_side(type);
   const uint16_t index = ++c->index;

   pan_cast_and_pack(job.cpu, JOB_HEADER, h) {
      h.type = type;
      h.barrier = barrier;
      h.index = index;
      h.dependency_1 = dep1;
      h.dependency_2 = tiler_side ? c->tiler_dep : 0;
   }
   mali_jm_chain_link(c, job, index, tiler_side);
   mali_jm_cmd_note_reset(cmd, job.cpu, NULL, 16);
}

/* Room for one more job in the chain's 16-bit index space, with a Cache
 * Flush job in front of it. */
static inline bool
chain_has_room(struct mali_cmd_buffer *cmd, const struct mali_jm_chain *c)
{
   if (likely(c->index < UINT16_MAX - 2))
      return true;
   mesa_loge("libmali: a job chain ran out of job indices (%u jobs)", c->jobs);
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return false;
}

struct mali_ptr
mali_jm_cmd_add_job(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c,
                    enum mali_job_type type, unsigned size, bool barrier, uint16_t dep1)
{
   if (!chain_has_room(cmd, c))
      return (struct mali_ptr){0};

   if (unlikely(c->pending)) {
      if (c->pending & MALI_JM_PENDING_FLUSH) {
         struct mali_ptr f = mali_cmd_alloc(cmd, pan_size(CACHE_FLUSH_JOB), 128);
         if (!f.cpu)
            return f;
         link_job(cmd, c, f, MALI_JOB_TYPE_CACHE_FLUSH, true, 0);
         pan_section_pack(f.cpu, CACHE_FLUSH_JOB, PAYLOAD, p) {
            p.invalidate_shader_core_other = true;
         }
      }
      barrier = true;
      c->pending = 0;
   }

   struct mali_ptr job = mali_cmd_alloc(cmd, size, 128);
   if (!job.cpu)
      return job;
   link_job(cmd, c, job, type, barrier, dep1);
   return job;
}

bool
mali_jm_cmd_write_value(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c,
                        enum mali_write_value_type type, uint64_t addr, uint64_t value,
                        bool barrier)
{
   struct mali_ptr j =
      mali_jm_cmd_add_job(cmd, c, MALI_JOB_TYPE_WRITE_VALUE, pan_size(WRITE_VALUE_JOB),
                          barrier, 0);
   if (!j.cpu)
      return false;
   pan_section_pack(j.cpu, WRITE_VALUE_JOB, PAYLOAD, p) {
      p.address = addr;
      p.type = type;
      p.immediate_value = value;
   }
   return true;
}

/* ---------------------------------------------------------------------- */
/* Batches                                                                 */

/* Barrier requirements of mali_jm_cmd::req (lazily applied, file comment).
 * VTC_*: what the next vtc job waits for; FRAG_*: the next fragment job. */
#define REQ_VTC_AFTER_VTC       (1u << 0)
#define REQ_VTC_AFTER_FRAG      (1u << 1)   /* every earlier fragment atom */
#define REQ_VTC_AFTER_FRAG_XFER (1u << 2)   /* earlier fragment-side transfers */
#define REQ_VTC_FLUSH           (1u << 3)
#define REQ_FRAG_AFTER_VTC      (1u << 4)
#define REQ_FRAG_AFTER_FRAG     (1u << 5)
#define REQ_FRAG_FLUSH          (1u << 6)
/* The VTC_AFTER_FRAG requirements of barriers whose vtc-slot destination
 * stages are only transfer stages: only the next transfer job on the vtc
 * chain (mali_jm_cmd_vtc_transfer) waits for them. */
#define REQ_XVTC_AFTER_FRAG      (1u << 7)
#define REQ_XVTC_AFTER_FRAG_XFER (1u << 8)

#define REQ_VTC_ALL                                                            \
   (REQ_VTC_AFTER_VTC | REQ_VTC_AFTER_FRAG | REQ_VTC_AFTER_FRAG_XFER | REQ_VTC_FLUSH)
#define REQ_XVTC_ALL (REQ_XVTC_AFTER_FRAG | REQ_XVTC_AFTER_FRAG_XFER)

static inline bool
batch_has_frag_jobs(const struct mali_cmd_buffer *cmd)
{
   const struct mali_jm_frag_seg *segs = util_dynarray_begin(&cmd->jm.frags);
   for (uint32_t i = 0; i < cmd->jm.cur.frag_count; i++)
      if (segs[cmd->jm.cur.frag_first + i].chain.jobs)
         return true;
   return false;
}

struct mali_jm_batch *
mali_jm_cmd_batch(struct mali_cmd_buffer *cmd)
{
   if (likely(cmd->jm.open))
      return &cmd->jm.cur;
   cmd->jm.cur = (struct mali_jm_batch){
      .frag_first = util_dynarray_num_elements(&cmd->jm.frags, struct mali_jm_frag_seg),
      .heap_slot = -1,
   };
   cmd->jm.open = true;
   return &cmd->jm.cur;
}

void
mali_jm_cmd_batch_close(struct mali_cmd_buffer *cmd)
{
   if (!cmd->jm.open)
      return;
   cmd->jm.open = false;
   struct mali_jm_batch *b = &cmd->jm.cur;
   if (!b->vtc.jobs && !batch_has_frag_jobs(cmd)) {
      /* Nothing recorded: no atoms. Its (empty) segments go too. */
      cmd->jm.frags.size = b->frag_first * sizeof(struct mali_jm_frag_seg);
      return;
   }
   if (!b->vtc.jobs)
      b->vtc = (struct mali_jm_chain){0};
   struct mali_jm_batch *slot = util_dynarray_grow(&cmd->jm.batches, struct mali_jm_batch, 1);
   if (slot)
      *slot = *b;
   else
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
}

/* The requirements on the next vtc job: a transfer job also takes the
 * transfer-only ones (folded into the plain bits). */
static inline uint16_t
vtc_req(uint16_t req, bool xfer)
{
   uint16_t want = req & REQ_VTC_ALL;
   if (xfer) {
      if (req & REQ_XVTC_AFTER_FRAG)
         want |= REQ_VTC_AFTER_FRAG;
      if (req & REQ_XVTC_AFTER_FRAG_XFER)
         want |= REQ_VTC_AFTER_FRAG_XFER;
   }
   return want;
}

/* Does meeting req on the vtc chain close the open batch? */
static inline bool
vtc_req_closes(const struct mali_cmd_buffer *cmd, uint16_t req)
{
   if (req & REQ_VTC_AFTER_FRAG)
      return batch_has_frag_jobs(cmd);
   if (req & REQ_VTC_AFTER_FRAG_XFER)
      return cmd->jm.cur.frag_xfer;
   return false;
}

static struct mali_jm_chain *
vtc_chain(struct mali_cmd_buffer *cmd, bool xfer)
{
   struct mali_jm_batch *b = mali_jm_cmd_batch(cmd);
   const uint16_t req = vtc_req(cmd->jm.req, xfer);

   /* Outside a render pass, a full batch closes before the next job. */
   if (unlikely(b->vtc.jobs >= MALI_JM_BATCH_MAX_JOBS) && !cmd->gfx.render.active) {
      mali_jm_cmd_batch_close(cmd);
      b = mali_jm_cmd_batch(cmd);
   }

   if (unlikely(req)) {
      if (req & (REQ_VTC_AFTER_FRAG | REQ_VTC_AFTER_FRAG_XFER)) {
         /* Fragment work of this batch runs after its vtc chain: if the
          * barrier waits for some, the batch ends here. */
         if (vtc_req_closes(cmd, req)) {
            mali_jm_cmd_batch_close(cmd);
            b = mali_jm_cmd_batch(cmd);
            b->vtc_after_frag = MALI_JM_AFTER_EARLIER;
         } else if (req & REQ_VTC_AFTER_FRAG) {
            b->vtc_after_frag = MALI_JM_AFTER_EARLIER;
         } else if (b->vtc_after_frag != MALI_JM_AFTER_EARLIER) {
            b->vtc_after_frag = MALI_JM_AFTER_EARLIER_XFER;
         }
      }
      /* An empty chain is a new atom: the slot order and the kernel's
       * cache maintenance between atoms already do this. */
      if (b->vtc.jobs && (req & REQ_VTC_AFTER_VTC))
         b->vtc.pending |= MALI_JM_PENDING_BARRIER |
                           ((req & REQ_VTC_FLUSH) ? MALI_JM_PENDING_FLUSH : 0);
      /* The transfer-only requirements are met with the plain ones: the
       * batch's vtc atom now waits for every fragment atom they name. */
      uint16_t met = REQ_VTC_ALL;
      if (xfer || (req & REQ_VTC_AFTER_FRAG))
         met |= REQ_XVTC_ALL;
      else if (req & REQ_VTC_AFTER_FRAG_XFER)
         met |= REQ_XVTC_AFTER_FRAG_XFER;
      cmd->jm.req &= ~met;
   }
   return &b->vtc;
}

struct mali_jm_chain *
mali_jm_cmd_vtc(struct mali_cmd_buffer *cmd)
{
   return vtc_chain(cmd, false);
}

struct mali_jm_chain *
mali_jm_cmd_vtc_transfer(struct mali_cmd_buffer *cmd)
{
   return vtc_chain(cmd, true);
}

bool
mali_jm_cmd_transfer_needs_frag(struct mali_cmd_buffer *cmd)
{
   return cmd->jm.open && vtc_req_closes(cmd, vtc_req(cmd->jm.req, true));
}

struct mali_jm_chain *
mali_jm_cmd_frag(struct mali_cmd_buffer *cmd, bool new_segment)
{
   struct mali_jm_batch *b = mali_jm_cmd_batch(cmd);
   const uint16_t req = cmd->jm.req;

   if (new_segment || !b->frag_count) {
      struct mali_jm_frag_seg *seg =
         util_dynarray_grow(&cmd->jm.frags, struct mali_jm_frag_seg, 1);
      if (!seg) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return NULL;
      }
      *seg = (struct mali_jm_frag_seg){0};
      b->frag_count++;
   }
   struct mali_jm_chain *c =
      &util_dynarray_element(&cmd->jm.frags, struct mali_jm_frag_seg,
                             b->frag_first + b->frag_count - 1)->chain;

   if (unlikely(req & (REQ_FRAG_AFTER_VTC | REQ_FRAG_AFTER_FRAG | REQ_FRAG_FLUSH))) {
      if (req & REQ_FRAG_AFTER_VTC)
         b->frag_after_vtc = true;
      if (c->jobs && (req & REQ_FRAG_AFTER_FRAG))
         c->pending |= MALI_JM_PENDING_BARRIER |
                       ((req & REQ_FRAG_FLUSH) ? MALI_JM_PENDING_FLUSH : 0);
      cmd->jm.req &= ~(REQ_FRAG_AFTER_VTC | REQ_FRAG_AFTER_FRAG | REQ_FRAG_FLUSH);
   }
   return c;
}

void
mali_jm_cmd_mark_frag_transfer(struct mali_cmd_buffer *cmd)
{
   mali_jm_cmd_batch(cmd)->frag_xfer = true;
}

bool
mali_jm_cmd_add_batch(struct mali_cmd_buffer *cmd, const struct mali_jm_batch *batch,
                      const struct mali_jm_frag_seg *frags, uint32_t frag_count)
{
   mali_jm_cmd_batch_close(cmd);
   struct mali_jm_batch b = *batch;
   b.frag_first = util_dynarray_num_elements(&cmd->jm.frags, struct mali_jm_frag_seg);
   b.frag_count = frag_count;
   if (frag_count) {
      struct mali_jm_frag_seg *f =
         util_dynarray_grow(&cmd->jm.frags, struct mali_jm_frag_seg, frag_count);
      if (!f)
         return false;
      memcpy(f, frags, frag_count * sizeof(*f));
   }
   struct mali_jm_batch *slot = util_dynarray_grow(&cmd->jm.batches, struct mali_jm_batch, 1);
   if (!slot)
      return false;
   *slot = b;
   return true;
}

/* ---------------------------------------------------------------------- */
/* Barriers                                                                */

/* Stages whose work runs on the vtc slot: vertex and compute shading,
 * indirect and vertex input, and transfers done by compute (buffer copies,
 * fills, updates, buffer<->image copies). */
#define VTC_STAGES                                                             \
   (VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | \
    VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | \
    VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT |                      \
    VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT |                   \
    VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | \
    VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT)

/* The vtc-slot stages that are transfers. */
#define VTC_XFER_STAGES (VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT)

/* Stages of rendering on the fragment slot. */
#define FRAG_RENDER_STAGES                                                     \
   (VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |                             \
    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |                                  \
    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |                              \
    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)

/* Transfers that may run on the fragment slot: fragment-side image copies,
 * blits, resolves and image clears (render passes with a frame shader). */
#define FRAG_XFER_STAGES                                                       \
   (VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT |              \
    VK_PIPELINE_STAGE_2_RESOLVE_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT)

/* Destination accesses that read through a shader core's read-only caches
 * (texture and attachment reads), which a write elsewhere does not
 * update: the reads need an "Invalidate Shader Core Other" first. L2 is
 * shared and the load/store caches are coherent with it (panvk
 * add_memory_dependency). Host accesses need nothing here: the kernel
 * cleans and invalidates the caches at every atom's start and end. */
#define RO_L1_ACCESS                                                           \
   (VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |                                    \
    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT | \
    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT)

void
mali_jm_cmd_barrier(struct mali_cmd_buffer *cmd, VkPipelineStageFlags2 src_stages,
                    VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stages,
                    VkAccessFlags2 dst_access)
{
   src_stages = vk_expand_src_stage_flags2(src_stages);
   dst_stages = vk_expand_dst_stage_flags2(dst_stages);
   dst_access = vk_filter_dst_access_flags2(dst_stages, dst_access);
   (void)src_access;

   const bool src_vtc = src_stages & VTC_STAGES;
   const bool src_frag = src_stages & FRAG_RENDER_STAGES;
   const bool src_frag_xfer = src_stages & FRAG_XFER_STAGES;
   /* vtc-slot destination stages, and whether any of them is not a
    * transfer (then every vtc job is in the second scope, else only
    * transfer jobs are). */
   const bool dst_vtc = dst_stages & VTC_STAGES;
   const bool dst_vtc_all = dst_stages & VTC_STAGES & ~VTC_XFER_STAGES;
   const bool dst_frag = dst_stages & (FRAG_RENDER_STAGES | FRAG_XFER_STAGES);
   const bool flush = dst_access & RO_L1_ACCESS;

   uint16_t req = 0;
   if (dst_vtc) {
      if (src_vtc)
         req |= REQ_VTC_AFTER_VTC | (flush ? REQ_VTC_FLUSH : 0);
      if (src_frag)
         req |= dst_vtc_all ? REQ_VTC_AFTER_FRAG : REQ_XVTC_AFTER_FRAG;
      else if (src_frag_xfer)
         req |= dst_vtc_all ? REQ_VTC_AFTER_FRAG_XFER : REQ_XVTC_AFTER_FRAG_XFER;
   }
   if (dst_frag) {
      if (src_vtc)
         req |= REQ_FRAG_AFTER_VTC;
      if (src_frag || src_frag_xfer)
         req |= REQ_FRAG_AFTER_FRAG | (flush ? REQ_FRAG_FLUSH : 0);
   }
   cmd->jm.req |= req;
}

/*
 * Image barriers after which the image's CRCs may not describe its memory
 * (the same rule as on v11, mali_cmd_buffer.c): a transition from
 * UNDEFINED or PREINITIALIZED, or an acquire from an external or foreign
 * queue family, on level 0 of the colour aspect.
 */
static bool
image_barrier_invalidates_crc(const VkImageMemoryBarrier2 *m)
{
   if (m->subresourceRange.baseMipLevel != 0 ||
       !(m->subresourceRange.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT))
      return false;
   if (m->oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ||
       m->oldLayout == VK_IMAGE_LAYOUT_PREINITIALIZED)
      return true;
   return m->srcQueueFamilyIndex != m->dstQueueFamilyIndex &&
          (m->srcQueueFamilyIndex == VK_QUEUE_FAMILY_EXTERNAL ||
           m->srcQueueFamilyIndex == VK_QUEUE_FAMILY_FOREIGN_EXT);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdPipelineBarrier2)(VkCommandBuffer commandBuffer,
                                   const VkDependencyInfo *info)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);

   VkPipelineStageFlags2 src = 0, dst = 0;
   VkAccessFlags2 srca = 0, dsta = 0;
   for (uint32_t i = 0; i < info->memoryBarrierCount; i++) {
      const VkMemoryBarrier2 *m = &info->pMemoryBarriers[i];
      src |= m->srcStageMask;
      dst |= m->dstStageMask;
      srca |= m->srcAccessMask;
      dsta |= m->dstAccessMask;
   }
   /* Queue family transfers need nothing more: one queue family, and the
    * kernel's cache maintenance at atom boundaries covers the external
    * and foreign ones. */
   for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; i++) {
      const VkBufferMemoryBarrier2 *m = &info->pBufferMemoryBarriers[i];
      src |= m->srcStageMask;
      dst |= m->dstStageMask;
      srca |= m->srcAccessMask;
      dsta |= m->dstAccessMask;
   }
   for (uint32_t i = 0; i < info->imageMemoryBarrierCount; i++) {
      const VkImageMemoryBarrier2 *m = &info->pImageMemoryBarriers[i];
      src |= m->srcStageMask;
      dst |= m->dstStageMask;
      srca |= m->srcAccessMask;
      dsta |= m->dstAccessMask;
      if (image_barrier_invalidates_crc(m))
         MALI_PER_ARCH(cmd_crc_invalidate)(cmd, container_of(vk_image_from_handle(m->image),
                                                             struct mali_image, vk));
   }

   if (cmd->gfx.render.active) {
      /* Inside a render pass only framebuffer-local dependencies are
       * allowed; the pass's vtc work all runs before its fragment jobs. A
       * by-region fragment-to-fragment dependency is a draw that carries
       * a primitive barrier (JC §12.3); without BY_REGION the blob emits
       * nothing. */
      const VkPipelineStageFlags2 fs = FRAG_RENDER_STAGES;
      if ((info->dependencyFlags & VK_DEPENDENCY_BY_REGION_BIT) &&
          (vk_expand_src_stage_flags2(src) & fs) && (vk_expand_dst_stage_flags2(dst) & fs))
         MALI_PER_ARCH(cmd_fb_barrier)(cmd);
      return;
   }
   mali_jm_cmd_barrier(cmd, src, srca, dst, dsta);
}

/* ---------------------------------------------------------------------- */
/* Begin and end                                                           */

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(BeginCommandBuffer)(VkCommandBuffer commandBuffer,
                                  const VkCommandBufferBeginInfo *info)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);

   /* Resets a command buffer recorded before (implicit reset). */
   vk_command_buffer_begin(&cmd->vk, info);
   /* Only command buffers that may run more than once pay for noting
    * what the GPU writes (ARMSX2 records ONE_TIME_SUBMIT only). A command
    * buffer pending twice at once (SIMULTANEOUS_USE) runs its submissions
    * one after the other: one chain cannot run twice at the same time. */
   cmd->jm.resubmit = !(info->flags & VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(EndCommandBuffer)(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);

   mali_jm_cmd_batch_close(cmd);
   /* What later command buffers' atoms still have to wait for. Waits
    * within one slot are kept by the slot order and the cache maintenance
    * between atoms. */
   const uint16_t req = cmd->jm.req;
   cmd->jm.end_req = 0;
   /* Later command buffers' vtc atoms are not told apart by kind: the
    * transfer-only requirements apply to all of them. */
   if (req & (REQ_VTC_AFTER_FRAG | REQ_XVTC_AFTER_FRAG))
      cmd->jm.end_req |= MALI_JM_REQ_VTC_AFTER_FRAG;
   else if (req & (REQ_VTC_AFTER_FRAG_XFER | REQ_XVTC_AFTER_FRAG_XFER))
      cmd->jm.end_req |= MALI_JM_REQ_VTC_AFTER_FRAG_XFER;
   if (req & REQ_FRAG_AFTER_VTC)
      cmd->jm.end_req |= MALI_JM_REQ_FRAG_AFTER_VTC;
   cmd->jm.req = 0;
   return vk_command_buffer_end(&cmd->vk);
}

/* ---------------------------------------------------------------------- */
/* Command buffer objects                                                  */

static void
cmd_clear(struct mali_cmd_buffer *cmd)
{
   /* Timestamps live in command memory (below, release_memory): read them
    * before it goes. */
   mali_jm_measure_cmd_reset(cmd);
   util_dynarray_clear(&cmd->jm.batches);
   util_dynarray_clear(&cmd->jm.frags);
   util_dynarray_clear(&cmd->jm.resets);
   util_dynarray_clear(&cmd->jm.reset_data);
   cmd->jm.open = false;
   cmd->jm.req = cmd->jm.end_req = 0;
   cmd->jm.resubmit = false;
   cmd->jm.submitted = false;
   cmd->jm.last_seq = 0;
   release_memory(cmd);
   memset(cmd->push_constants, 0, sizeof(cmd->push_constants));
   memset(&cmd->compute, 0, sizeof(cmd->compute));
   memset(&cmd->gfx, 0, sizeof(cmd->gfx));
   cmd->gfx.draw.dirty = MALI_GFX_DIRTY_ALL;
   cmd->dispatches = cmd->draws = cmd->passes = 0;
}

static VkResult
cmd_create(struct vk_command_pool *pool, VkCommandBufferLevel level,
           struct vk_command_buffer **out)
{
   struct mali_device *dev = container_of(pool->base.device, struct mali_device, vk);
   struct mali_cmd_buffer *cmd =
      vk_zalloc(&pool->alloc, sizeof(*cmd), alignof(struct mali_cmd_buffer),
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!cmd)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result =
      vk_command_buffer_init(pool, &cmd->vk, &MALI_PER_ARCH(cmd_buffer_ops), level);
   if (result != VK_SUCCESS) {
      vk_free(&pool->alloc, cmd);
      return result;
   }
   cmd->dev = dev;
   util_dynarray_init(&cmd->jm.batches, NULL);
   util_dynarray_init(&cmd->jm.frags, NULL);
   util_dynarray_init(&cmd->jm.resets, NULL);
   util_dynarray_init(&cmd->jm.reset_data, NULL);
   list_inithead(&cmd->slabs);
   mali_jm_draw_tmpl_init(&cmd->jm.draw);
   cmd->vk.dynamic_graphics_state.vi = &cmd->dyn_vi;
   cmd->vk.dynamic_graphics_state.ms.sample_locations = &cmd->dyn_sl;
   cmd->gfx.draw.dirty = MALI_GFX_DIRTY_ALL;
   mali_jm_measure_cmd_create(cmd);
   *out = &cmd->vk;
   return VK_SUCCESS;
}

static void
cmd_reset(struct vk_command_buffer *vk_cmd, VkCommandBufferResetFlags flags)
{
   struct mali_cmd_buffer *cmd = container_of(vk_cmd, struct mali_cmd_buffer, vk);
   vk_command_buffer_reset(&cmd->vk);
   cmd_clear(cmd);
}

static void
cmd_destroy(struct vk_command_buffer *vk_cmd)
{
   struct mali_cmd_buffer *cmd = container_of(vk_cmd, struct mali_cmd_buffer, vk);
   mali_jm_measure_cmd_destroy(cmd);
   release_memory(cmd);
   util_dynarray_fini(&cmd->jm.batches);
   util_dynarray_fini(&cmd->jm.frags);
   util_dynarray_fini(&cmd->jm.resets);
   util_dynarray_fini(&cmd->jm.reset_data);
   vk_command_buffer_finish(&cmd->vk);
   vk_free(&cmd->vk.pool->alloc, cmd);
}

const struct vk_command_buffer_ops MALI_PER_ARCH(cmd_buffer_ops) = {
   .create = cmd_create,
   .reset = cmd_reset,
   .destroy = cmd_destroy,
};
