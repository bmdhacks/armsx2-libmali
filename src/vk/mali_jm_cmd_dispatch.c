/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Compute dispatch on the job manager (v9): one 128-byte Compute job per
 * dispatch in the open batch's vtc chain (Mesa gallium pan_jm.c
 * jm_launch_grid; the T820 blob writes the same job). The FAU block,
 * resource table and Local Storage descriptor come from the builders
 * shared with the v11 back half (mali_cmd_state.h); what differs is where
 * they go:
 *
 *  - workgroup size, counts and task axis/increment in the Compute
 *    payload; offsets 0, as on v11 (the base workgroup is a system value);
 *  - one Shader Environment: the resource table, the shader, the TSD, and
 *    the FAU block as a pointer and a separate 8-bit count of 64-bit words
 *    (the builders return v11's packed FAU word, address | count << 56);
 *  - no Barrier bit of its own: dispatches without a barrier between them
 *    may overlap, as Vulkan allows; a recorded barrier puts the bit (and a
 *    Cache Flush job) in front of the next job (mali_jm_cmd_buffer.c).
 *
 * The internal dispatches (cmd_dispatch_shader, cmd_dispatch_meta) are the
 * transfers done by compute (copies, fills, updates) and meet the barriers
 * whose destination is only the transfer stages; vkCmdDispatch does not.
 */

#include "mali_cmd_state.h"

#include "mali_measure.h"

static void
dispatch(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
         const struct mali_desc_state *desc, const void *push, uint32_t push_size,
         const uint32_t base[3], const uint32_t groups[3], const uint32_t (*textures)[8],
         unsigned texture_count, bool xfer)
{
   if (!cs || !cs->spd[MALI_SPD_MAIN])
      return;
   if (!groups[0] || !groups[1] || !groups[2])
      return;

   const struct mali_thread_props tp = mali_thread_props(cmd->dev);

   const uint64_t fau = mali_compute_fau(cmd, cs, push, push_size, base, groups);
   const uint64_t srt = mali_compute_res_table(cmd, cs, desc, textures, texture_count);
   const uint64_t tsd = mali_compute_tsd(cmd, cs, &tp, groups);
   if (vk_command_buffer_has_error(&cmd->vk) || !srt || !tsd ||
       (cs->fau.total_count && !fau))
      return;

   unsigned axis, inc;
   mali_compute_task_axis(&tp, cs, groups, &axis, &inc);

   struct mali_jm_chain *c = xfer ? mali_jm_cmd_vtc_transfer(cmd) : mali_jm_cmd_vtc(cmd);
   if (!c)
      return;

   /* Timing. */
   uint32_t mh = 0;
   if (unlikely(cmd->jm.measure)) {
      mh = mali_jm_measure_begin(cmd, MALI_MEASURE_DISPATCH, c, MALI_JM_SLOT_VTC,
                                 cmd->dispatches);
      uint64_t key;
      memcpy(&key, cs->key, sizeof(key));
      mali_jm_measure_info(cmd, mh, groups[0], groups[1], groups[2],
                          cs->cs.local_size[0] * cs->cs.local_size[1] * cs->cs.local_size[2]);
      mali_jm_measure_shader(cmd, mh, key);
   }

   struct mali_ptr job =
      mali_jm_cmd_add_job(cmd, c, MALI_JOB_TYPE_COMPUTE, pan_size(COMPUTE_JOB), false, 0);
   if (!job.cpu) {
      if (unlikely(mh))
         mali_jm_measure_end(cmd, mh, c);
      return;
   }

   pan_section_pack(job.cpu, COMPUTE_JOB, PAYLOAD, cfg) {
      cfg.workgroup_size_x = cs->cs.local_size[0];
      cfg.workgroup_size_y = cs->cs.local_size[1];
      cfg.workgroup_size_z = cs->cs.local_size[2];
      cfg.allow_merging_workgroups = cs->info.cs.allow_merging_workgroups;
      /* 14 bits; the builder's increment is a few workgroups per task. */
      cfg.task_increment = MIN2(inc, (1u << 14) - 1);
      cfg.task_axis = axis;
      cfg.workgroup_count_x = groups[0];
      cfg.workgroup_count_y = groups[1];
      cfg.workgroup_count_z = groups[2];
      cfg.compute.resources = srt;
      cfg.compute.shader = cs->spd[MALI_SPD_MAIN];
      cfg.compute.thread_storage = tsd;
      cfg.compute.fau = fau & BITFIELD64_MASK(56);
      cfg.compute.fau_count = fau >> 56;
   }
   cmd->dispatches++;
   if (unlikely(mh))
      mali_jm_measure_end(cmd, mh, c);
}

void
MALI_PER_ARCH(cmd_dispatch_shader)(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
                                   const struct mali_desc_state *desc, const void *push,
                                   uint32_t push_size, const uint32_t base[3],
                                   const uint32_t groups[3])
{
   dispatch(cmd, cs, desc, push, push_size, base, groups, NULL, 0, true);
}

void
MALI_PER_ARCH(cmd_dispatch_meta)(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
                                 const void *push, uint32_t push_size, const uint32_t groups[3],
                                 const uint32_t (*textures)[8], unsigned texture_count)
{
   const uint32_t base[3] = {0, 0, 0};
   dispatch(cmd, cs, NULL, push, push_size, base, groups, textures, texture_count, true);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdDispatchBase)(VkCommandBuffer commandBuffer, uint32_t baseGroupX,
                               uint32_t baseGroupY, uint32_t baseGroupZ,
                               uint32_t groupCountX, uint32_t groupCountY,
                               uint32_t groupCountZ)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   const uint32_t base[3] = {baseGroupX, baseGroupY, baseGroupZ};
   const uint32_t groups[3] = {groupCountX, groupCountY, groupCountZ};
   dispatch(cmd, cmd->compute.shader, &cmd->compute.desc, cmd->push_constants,
            sizeof(cmd->push_constants), base, groups, NULL, 0, false);
}
