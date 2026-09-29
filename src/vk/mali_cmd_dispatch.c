/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Compute dispatch (vkCmdDispatch, vkCmdDispatchBase, and the internal
 * copy/fill shaders). Encoding follows panvk v11 (Mesa
 * csf/panvk_vX_cmd_dispatch.c), with what our compiled shaders need the
 * command buffer to provide:
 *
 *  - FAU: the used words of struct mali_compute_sysvals, then the used
 *    push-constant words, then the constants the compiler promoted
 *    (pan_fau_foreach_imm); pointer | word count << 56 in the FAU register
 *    (the v11 packing);
 *  - resource table: entry 0 the driver set (the dummy sampler, then the
 *    dynamic buffers in the shader's dyn_bufs order with their offsets),
 *    entry N + 1 descriptor set N;
 *  - TSD: a Local Storage descriptor with the command buffer's TLS buffer
 *    and, for shaders with shared memory, a per-dispatch WLS buffer;
 *  - WG_SIZE, JOB_OFFSET (0: the base workgroup is a system value), JOB_SIZE,
 *    and RUN_COMPUTE with the task axis/increment panvk computes, register
 *    group 0 (the blob uses group 3; either is valid).
 *
 * The iterator scoreboard entry moves on before each RUN_COMPUTE
 * (cs_iter_sb_update on v11: NEXT_SB_ENTRY), so consecutive dispatches can
 * overlap; barriers and the end of the command buffer wait for them.
 */

#include "mali_cmd_buffer.h"
#include "mali_measure.h"

#include <string.h>

#include "util/bitscan.h"
#include "util/u_math.h"

#include "mali_descriptor_set.h"
#include "mali_queue.h"
#include "mali_vk.h"

#define COMPUTE_SRT MALI_COMPUTE_SR_SRT_0
#define COMPUTE_FAU MALI_COMPUTE_SR_FAU_0
#define COMPUTE_SPD MALI_COMPUTE_SR_SPD_0
#define COMPUTE_TSD MALI_COMPUTE_SR_TSD_0

/* ---------------------------------------------------------------------- */
/* GPU facts the dispatch sizing needs (panthor's names, kbase's values)   */

struct thread_props {
   unsigned max_threads_per_wg;
   unsigned max_threads_per_core;
   unsigned max_tasks_per_core;
   unsigned registers_per_core;
   unsigned core_count;
   unsigned core_id_range;
};

static struct thread_props
thread_props(const struct mali_device *dev)
{
   const struct mali_kbase_gpu_props *p = &dev->kbase->props;
   struct thread_props t = {
      .max_threads_per_wg = p->thread_max_workgroup_size ? p->thread_max_workgroup_size
                                                         : p->max_workgroup_size,
      .max_threads_per_core = p->thread_max_threads ? p->thread_max_threads : p->max_threads,
      .max_tasks_per_core = p->thread_features >> 24,
      .registers_per_core = p->thread_features & 0x3fffff,
      .core_count = util_bitcount64(p->shader_present),
      .core_id_range = util_last_bit64(p->shader_present),
   };
   /* Fallbacks for a property stream without the THREAD_* registers (the
    * host tests' fake): the G615's values. */
   if (!t.max_threads_per_wg)
      t.max_threads_per_wg = 1024;
   if (!t.max_threads_per_core)
      t.max_threads_per_core = 2048;
   if (!t.max_tasks_per_core)
      t.max_tasks_per_core = 4;
   if (!t.registers_per_core)
      t.registers_per_core = 65536;
   if (!t.core_count)
      t.core_count = t.core_id_range = 1;
   return t;
}

/* panvk calculate_task_axis_and_increment: spread workgroups evenly over
 * the cores, with tasks small enough that other clients can share them. */
static void
task_axis_and_increment(const struct thread_props *tp, const struct mali_shader *cs,
                        const uint32_t groups[3], unsigned *axis, unsigned *inc)
{
   const unsigned regs = cs->info.work_reg_count <= 32 ? 32 : 64;
   const unsigned max_threads =
      MIN3(tp->max_threads_per_wg, tp->max_threads_per_core, tp->registers_per_core / regs);
   const unsigned threads_per_wg =
      cs->cs.local_size[0] * cs->cs.local_size[1] * cs->cs.local_size[2];
   const uint64_t total = (uint64_t)groups[0] * groups[1] * groups[2];

   *axis = MALI_TASK_AXIS_X;
   *inc = 1;
   if (!total || !threads_per_wg)
      return;

   const unsigned wgs_per_core = (unsigned)MIN2(DIV_ROUND_UP(total, tp->core_count), UINT32_MAX);
   const unsigned threads_per_task = DIV_ROUND_UP(max_threads, tp->max_tasks_per_core);
   unsigned wgs_per_task = CLAMP(threads_per_task / threads_per_wg, 1, wgs_per_core);

   *inc = wgs_per_task;
   for (unsigned i = 0; i < 2; i++) {
      if (*inc <= groups[i])
         break;
      (*axis)++;
      *inc /= groups[i];
   }
   *inc = MAX2(*inc, 1);
}

/* ---------------------------------------------------------------------- */
/* Per-dispatch memory                                                     */

static uint64_t
build_fau(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
          const void *push, uint32_t push_size, const uint32_t base[3],
          const uint32_t groups[3])
{
   if (!cs->fau.total_count)
      return 0;

   struct mali_ptr p = mali_cmd_alloc(cmd, cs->fau.total_count * 8, 16);
   if (!p.cpu)
      return 0;

   struct mali_compute_sysvals sv;
   memset(&sv, 0, sizeof(sv));
   sv.base.x = base[0];
   sv.base.y = base[1];
   sv.base.z = base[2];
   sv.num_work_groups.x = groups[0];
   sv.num_work_groups.y = groups[1];
   sv.num_work_groups.z = groups[2];
   sv.local_group_size.x = cs->cs.local_size[0];
   sv.local_group_size.y = cs->cs.local_size[1];
   sv.local_group_size.z = cs->cs.local_size[2];
   sv.common.push_uniforms = p.gpu;

   uint64_t pc[MALI_MAX_PUSH_CONST_FAUS];
   memset(pc, 0, sizeof(pc));
   memcpy(pc, push, MIN2(push_size, sizeof(pc)));

   uint64_t *out = p.cpu;
   uint64_t tmp[MALI_FAU_WORD_COUNT];
   unsigned n = 0, w;
   BITSET_FOREACH_SET(w, cs->fau.used_sysvals, MALI_MAX_SYSVAL_FAUS) {
      tmp[n++] = mali_sysval_word(&sv, sizeof(sv), w);
   }
   BITSET_FOREACH_SET(w, cs->fau.used_push_consts, MALI_MAX_PUSH_CONST_FAUS)
      tmp[n++] = pc[w];
   n += mali_shader_fau_consts(&cs->fau, &tmp[n]);
   for (unsigned i = n; i < cs->fau.total_count; i++)
      tmp[i] = 0;

   /* Constants the backend promoted into the FAU, in 32-bit halves. */
   pan_fau_foreach_imm(&cs->info.fau, i) {
      bool hi = i & 1;
      unsigned idx = i / 2;
      assert(idx < cs->fau.total_count);
      tmp[idx] = (tmp[idx] & ((uint64_t)UINT32_MAX << (32 * !hi))) |
                 ((uint64_t)cs->info.fau.words[i].constant << (32 * hi));
   }

   /* One pass of 64-bit stores into uncached memory. */
   for (unsigned i = 0; i < cs->fau.total_count; i++)
      out[i] = tmp[i];
   return p.gpu;
}

/* Resource table: driver set, then the sets the shader uses. Internal
 * shaders add their textures after the dynamic buffers (they have none). */
static uint64_t
build_res_table(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
                const struct mali_desc_state *desc, const uint32_t (*textures)[8],
                unsigned texture_count)
{
   const struct mali_shader_desc_info *di = &cs->desc;

   /* Table 0 for compute: the dummy sampler, then the dynamic buffers. */
   uint32_t drv_count = 1 + di->dyn_bufs.count + texture_count;
   struct mali_ptr drv = mali_cmd_alloc(cmd, drv_count * MALI_DESCRIPTOR_SIZE,
                                        MALI_DESCRIPTOR_SIZE);
   if (!drv.cpu)
      return 0;
   mali_pack_dummy_sampler(drv.cpu);
   for (uint32_t i = 0; i < di->dyn_bufs.count; i++) {
      uint32_t h = di->dyn_bufs.map[i];
      uint32_t set = MALI_COPY_DESC_HANDLE_SET(h);
      uint32_t idx = MALI_COPY_DESC_HANDLE_INDEX(h);
      const struct mali_descriptor_set *s = desc ? desc->sets[set] : NULL;
      uint8_t *out = (uint8_t *)drv.cpu + (1 + i) * MALI_DESCRIPTOR_SIZE;
      if (s)
         mali_descriptor_set_pack_dyn_buf(s, idx, desc->dyn_offsets[set][idx], out);
      else
         memset(out, 0, MALI_DESCRIPTOR_SIZE);
   }
   for (unsigned i = 0; i < texture_count; i++)
      memcpy((uint8_t *)drv.cpu + (1 + di->dyn_bufs.count + i) * MALI_DESCRIPTOR_SIZE,
             textures[i], MALI_DESCRIPTOR_SIZE);

   uint32_t last = util_last_bit(di->used_set_mask);
   uint32_t count = ALIGN_POT(1 + last, MALI_RESOURCE_TABLE_SIZE_ALIGNMENT);
   struct mali_ptr t = mali_cmd_alloc(cmd, count * MALI_RESOURCE_SIZE, 64);
   if (!t.cpu)
      return 0;

   pan_cast_and_pack(t.cpu, RESOURCE, cfg) {
      cfg.address = drv.gpu;
      cfg.size = drv_count * MALI_DESCRIPTOR_SIZE;
      cfg.contains_descriptors = true;
   }
   for (uint32_t i = 1; i < count; i++) {
      uint32_t set = i - 1;
      const struct mali_descriptor_set *s =
         (set < MALI_MAX_SETS && (di->used_set_mask & BITFIELD_BIT(set)) && desc) ?
            desc->sets[set] : NULL;
      mali_descriptor_set_pack_resource(s, (uint8_t *)t.cpu + i * MALI_RESOURCE_SIZE);
   }
   return t.gpu | count;
}

uint64_t
mali_cmd_tls_buffer(struct mali_cmd_buffer *cmd, uint32_t tls_size)
{
   struct mali_device *dev = cmd->dev;
   const struct thread_props tp = thread_props(dev);

   /* Per thread: the next power of two of the size, at least 16 bytes; for
    * every thread slot of every core (pan_get_total_stack_size). */
   unsigned per_thread = util_next_power_of_two(ALIGN_POT(tls_size, 16));
   if (per_thread > cmd->tls.size) {
      uint64_t total = (uint64_t)per_thread * tp.max_threads_per_core * tp.core_id_range;
      /* TLS memory: GPU group 9. */
      struct mali_cmd_slab *s = calloc(1, sizeof(*s));
      const struct mali_kbase_alloc_info ai = {
         .size = total,
         .flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_TLS),
         .mem_class = MALI_KBASE_MEM_CLASS_INTERNAL_TLS,
      };
      if (!s || mali_kbase_alloc(dev->kbase, &ai, &s->bo) != MALI_KBASE_SUCCESS) {
         free(s);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return 0;
      }
      list_addtail(&s->link, &cmd->slabs);
      cmd->tls.size = per_thread;
      cmd->tls.gpu = s->bo.gpu_va;
   }
   return cmd->tls.gpu;
}

/* Local Storage descriptor (pan_emit_tls for v11). */
static uint64_t
build_tsd(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
          const struct thread_props *tp, const uint32_t groups[3])
{
   struct mali_device *dev = cmd->dev;
   uint64_t tls_ptr = 0, wls_ptr = 0;
   unsigned wls_instances = 0, wls_size = 0;

   if (cs->info.tls_size) {
      tls_ptr = mali_cmd_tls_buffer(cmd, cs->info.tls_size);
      if (!tls_ptr)
         return 0;
   }

   if (cs->info.wls_size) {
      /* pan_calc_wls_instances / pan_calc_total_wls_size. */
      unsigned threads_per_wg =
         cs->cs.local_size[0] * cs->cs.local_size[1] * cs->cs.local_size[2];
      unsigned wg_per_task =
         DIV_ROUND_UP(tp->max_threads_per_core / tp->max_tasks_per_core, threads_per_wg);
      unsigned max_per_core = util_next_power_of_two(wg_per_task * tp->max_tasks_per_core);
      unsigned dispatch = util_next_power_of_two(groups[0]) *
                          util_next_power_of_two(groups[1]) *
                          util_next_power_of_two(groups[2]);
      wls_instances = MIN2(dispatch, max_per_core);
      wls_size = util_next_power_of_two(MAX2(cs->info.wls_size, 128));
      uint64_t total = (uint64_t)wls_size * wls_instances * tp->core_id_range;

      /* 4 KiB aligned and inside one 4 GiB window (the descriptor's
       * requirement): its own allocation, kept in one 4 GiB page. */
      struct mali_cmd_slab *s = calloc(1, sizeof(*s));
      const struct mali_kbase_alloc_info ai = {
         .size = total,
         .flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_TLS |
                                                KB_MEM_GPU_VA_SAME_4GB_PAGE),
         .mem_class = MALI_KBASE_MEM_CLASS_INTERNAL_TLS,
      };
      if (!s || mali_kbase_alloc(dev->kbase, &ai, &s->bo) != MALI_KBASE_SUCCESS) {
         free(s);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return 0;
      }
      list_addtail(&s->link, &cmd->slabs);
      wls_ptr = s->bo.gpu_va;
   }

   struct mali_ptr p = mali_cmd_alloc(cmd, pan_size(LOCAL_STORAGE), 64);
   if (!p.cpu)
      return 0;
   pan_cast_and_pack(p.cpu, LOCAL_STORAGE, cfg) {
      if (tls_ptr) {
         cfg.tls_size = util_logbase2_ceil(DIV_ROUND_UP(cs->info.tls_size, 16));
         cfg.tls_address_mode = MALI_ADDRESS_MODE_PACKED;
         cfg.tls_base_pointer = tls_ptr >> 8;
      }
      if (wls_ptr) {
         cfg.wls_base_pointer = wls_ptr;
         cfg.wls_instances = wls_instances;
         cfg.wls_size_scale = util_logbase2(wls_size) + 1;
      } else {
         cfg.wls_instances = MALI_LOCAL_STORAGE_NO_WORKGROUP_MEM;
      }
   }
   return p.gpu;
}

/* ---------------------------------------------------------------------- */
/* Encoding                                                                */

/* v11 cs_iter_sb_update with nothing inside: select the next iterator
 * scoreboard entry for the next asynchronous job, make it the indirect
 * wait mask, and keep the current one out of the stream mask. */
static void
next_iter_sb(struct mali_cmd_buffer *cmd, struct cs_builder *b)
{
   const struct mali_csf_device *csf = cmd->dev->csf;
   struct cs_index next_sb = mali_cs_scratch_reg32(b, 0);
   struct cs_index sb_mask = mali_cs_scratch_reg32(b, 1);

   cs_next_sb_entry(b, next_sb, MALI_CS_SCOREBOARD_TYPE_ENDPOINT,
                    MALI_CS_NEXT_SB_ENTRY_FORMAT_INDEX);
   cs_move32_to(b, sb_mask, 0);
   cs_bit_set32(b, sb_mask, sb_mask, next_sb);
   cs_set_state(b, MALI_CS_SET_STATE_TYPE_SB_MASK_WAIT, sb_mask);
   cs_move32_to(b, sb_mask, csf->sb.all_iters_mask);
   cs_bit_clear32(b, sb_mask, sb_mask, next_sb);
   cs_set_state(b, MALI_CS_SET_STATE_TYPE_SB_MASK_STREAM, sb_mask);
}

static void
dispatch(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
         const struct mali_desc_state *desc, const void *push, uint32_t push_size,
         const uint32_t base[3], const uint32_t groups[3], const uint32_t (*textures)[8],
         unsigned texture_count)
{
   if (!cs || !cs->spd[MALI_SPD_MAIN])
      return;
   if (!groups[0] || !groups[1] || !groups[2])
      return;

   const struct thread_props tp = thread_props(cmd->dev);

   uint64_t fau = build_fau(cmd, cs, push, push_size, base, groups);
   uint64_t srt = build_res_table(cmd, cs, desc, textures, texture_count);
   uint64_t tsd = build_tsd(cmd, cs, &tp, groups);
   if (vk_command_buffer_has_error(&cmd->vk) || !srt || !tsd ||
       (cs->fau.total_count && !fau))
      return;

   unsigned axis, inc;
   task_axis_and_increment(&tp, cs, groups, &axis, &inc);

   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_COMPUTE);

   cs_move64_to(b, cs_reg64(b, COMPUTE_SRT), srt);
   cs_move64_to(b, cs_reg64(b, COMPUTE_FAU),
                fau | ((uint64_t)cs->fau.total_count << 56));
   cs_move64_to(b, cs_reg64(b, COMPUTE_SPD), cs->spd[MALI_SPD_MAIN]);
   cs_move64_to(b, cs_reg64(b, COMPUTE_TSD), tsd);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, GLOBAL_ATTRIBUTE_OFFSET), 0);

   struct mali_compute_size_workgroup_packed wg;
   pan_pack(&wg, COMPUTE_SIZE_WORKGROUP, cfg) {
      cfg.workgroup_size_x = cs->cs.local_size[0];
      cfg.workgroup_size_y = cs->cs.local_size[1];
      cfg.workgroup_size_z = cs->cs.local_size[2];
      cfg.allow_merging_workgroups = cs->info.cs.allow_merging_workgroups;
   }
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, WG_SIZE), wg.opaque[0]);

   /* The base workgroup is added by the shader (a system value); the job
    * offset does not apply to the workgroup ID on Mali (panvk). */
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_OFFSET_X), 0);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_OFFSET_Y), 0);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_OFFSET_Z), 0);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_SIZE_X), groups[0]);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_SIZE_Y), groups[1]);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_SIZE_Z), groups[2]);

   /* Timing. */
   uint32_t mh = 0;
   if (unlikely(cmd->measure)) {
      mh = mali_measure_begin(cmd, MALI_MEASURE_DISPATCH, MALI_SUBQUEUE_COMPUTE,
                              cmd->dispatches);
      uint64_t key;
      memcpy(&key, cs->key, sizeof(key));
      mali_measure_info(cmd, mh, groups[0], groups[1], groups[2],
                        cs->cs.local_size[0] * cs->cs.local_size[1] * cs->cs.local_size[2]);
      mali_measure_shader(cmd, mh, key);
   }

   next_iter_sb(cmd, b);
   cs_run_compute(b, inc, axis, cs_shader_res_sel(0, 0, 0, 0));
   cmd->dispatches++;
   if (unlikely(mh))
      mali_measure_end(cmd, mh);
}

void
mali_cmd_dispatch_shader(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
                         const struct mali_desc_state *desc, const void *push,
                         uint32_t push_size, const uint32_t base[3],
                         const uint32_t groups[3])
{
   dispatch(cmd, cs, desc, push, push_size, base, groups, NULL, 0);
}

void
mali_cmd_dispatch_meta(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
                       const void *push, uint32_t push_size, const uint32_t groups[3],
                       const uint32_t (*textures)[8], unsigned texture_count)
{
   const uint32_t base[3] = {0, 0, 0};
   dispatch(cmd, cs, NULL, push, push_size, base, groups, textures, texture_count);
}

VKAPI_ATTR void VKAPI_CALL
mali_CmdDispatchBase(VkCommandBuffer commandBuffer, uint32_t baseGroupX,
                     uint32_t baseGroupY, uint32_t baseGroupZ,
                     uint32_t groupCountX, uint32_t groupCountY,
                     uint32_t groupCountZ)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   const uint32_t base[3] = {baseGroupX, baseGroupY, baseGroupZ};
   const uint32_t groups[3] = {groupCountX, groupCountY, groupCountZ};
   mali_cmd_dispatch_shader(cmd, cmd->compute.shader, &cmd->compute.desc,
                            cmd->push_constants, sizeof(cmd->push_constants),
                            base, groups);
}
