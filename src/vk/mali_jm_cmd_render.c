/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Render passes on the job manager (v9): the tiler side (heap slot, Tiler
 * Contexts, the pass's Local Storage descriptor), the fragment jobs, and
 * the full-screen jobs (in-pass clears and the in-pass by-region barrier).
 * What does not depend on the frontend (tile-buffer layout, preloads and
 * frame shaders, framebuffer descriptors, vkCmdBeginRendering) is in
 * mali_fb.c, shared with the v11 back half.
 *
 * A pass lives inside one batch (mali_jm.h). Its order of events:
 *
 *  - begin: the open batch is closed first if it is full (its pass count,
 *    vtc job count or heap estimate); then the shared layout. A pass that
 *    preloads reads earlier attachment writes through the texture unit,
 *    which a fragment -> fragment barrier with a cache flush on the
 *    fragment chain covers (the v11 pass does the same with its stream
 *    barrier);
 *  - first tiled job (a draw, a full-screen job): the barriers owed to vtc
 *    work are applied here and not at begin (mali_jm_cmd_vtc), because
 *    the batch may still close at this point (nothing of the pass is in it
 *    yet). Then the batch takes a heap slot if it has none
 *    (mali_jm_heap.c), and the pass gets one 192-byte Tiler Context per
 *    layer, every word written, all pointing at the batch's one heap
 *    descriptor. A pass that never tiles has no contexts (the FBD's tiler
 *    pointer stays 0, as Mesa's v9 gallium driver does for a clear-only
 *    batch) and no heap slot;
 *  - end: the framebuffer descriptors (mali_fb.c, one per layer) and one
 *    128-byte Fragment job per layer in the open batch's current fragment
 *    segment, bounding box in 16-pixel tiles. Pending fragment-side
 *    barriers are applied by mali_jm_cmd_frag (a Barrier bit and Cache
 *    Flush job in the chain). A pass with nothing to store writes
 *    nothing.
 *
 * Differences from the blob's passes and why:
 *
 *  - no heap-setup compute job and no JIT atoms: the CPU owns the heap
 *    memory and writes the descriptor (mali_jm_heap.c);
 *  - one Tiler Context per layer, not two: the blob points full-screen jobs
 *    at a second context per layer (hierarchy mask 0x1000); we point them
 *    at the layer's own context, as Mesa does (to be confirmed on the
 *    device);
 *  - Layer offset 0 in every context. Each layer has its own context and
 *    polygon lists, and the FBD selects internal layer 0. The blob writes
 *    -i into layer i's draw context, so that a primitive's layer index
 *    picks its context. A full-screen job has no shader to give its
 *    primitive a layer; if the tiler takes it as layer 0, an offset of -i
 *    would drop it from every layer but the first. Noysz's arch-9 port
 *    runs one context per layer with offset 0 and passes the layered
 *    rendering tests on a G57. To be confirmed on the device. A draw of a
 *    layered pass goes to layer 0's context only: no v9 shader writes a
 *    layer, so all its primitives are in layer 0 (mali_jm_cmd_draw.c);
 *  - all layers' fragment jobs in one fragment chain (the blob has a chain
 *    per layer): they need no order between them;
 *  - transaction elimination (CRC) is off on v9 until the device shows it
 *    pays: no image has CRC state, so no pass picks a CRC target. Turning
 *    it on means putting the image's seed into the CRC Clear Color here,
 *    on the CPU, from a per-image seed counter bumped at record time.
 */

#include "mali_cmd_state.h"

#include "vk_log.h"

#if PAN_ARCH != 9
#error "mali_jm_cmd_render.c is the v9 (job manager) back half"
#endif

/* What one tiled pass adds to its batch's heap estimate on top of its
 * draws: bin pointers and the first polygon-list blocks. A guess, to be
 * measured on the device; it only matters for batches of hundreds
 * of passes, which the pass limit closes first. */
#define MALI_JM_HEAP_PASS_EST (64 * 1024)

_Static_assert(MALI_LAYERS_PER_TILER_CTX == 1, "one Tiler Context per layer on v9");

/* ---------------------------------------------------------------------- */
/* Begin                                                                   */

void
MALI_PER_ARCH(cmd_render_begin)(struct mali_cmd_buffer *cmd, const struct mali_render_desc *desc)
{
   /* The batch limits (mali_jm.h): a pass's tiling stays in one
    * batch, so a full batch closes before the pass starts. */
   if (cmd->jm.open) {
      const struct mali_jm_batch *b = &cmd->jm.cur;
      if (b->passes >= MALI_JM_BATCH_MAX_PASSES || b->vtc.jobs >= MALI_JM_BATCH_MAX_JOBS ||
          b->est_heap_bytes >= MALI_JM_HEAP_SLOT_SIZE / 2)
         mali_jm_cmd_batch_close(cmd);
   }

   if (MALI_PER_ARCH(fb_begin)(cmd, desc)) {
      /* Preloads go through the texture unit: earlier attachment writes of
       * this batch must be done and its caches invalidated (applied to the
       * pass's first fragment job, mali_jm_cmd_frag). */
      mali_jm_cmd_barrier(cmd,
                          VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                             VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                          VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                          VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
   }
}

/* ---------------------------------------------------------------------- */
/* Tiler side                                                              */

/* The tiler's hierarchy levels: the blob's fixed 0xaa (levels 1, 3, 5, 7)
 * when the tiler can keep four levels active, which covers framebuffers
 * up to 2048 pixels; otherwise Mesa's spread for fewer levels
 * (pan_select_tiler_hierarchy_mask). To be checked against the G57's
 * TILER_FEATURES on the device. */
static uint32_t
hierarchy_mask(struct mali_device *dev)
{
   static const uint32_t fewer[4] = {MALI_TILER_HIERARCHY_MASK, 0x80, 0x82, 0xa2};
   const unsigned levels = mali_device_physical(dev)->props.tiler_max_active_levels;
   return levels < 4 ? fewer[levels] : MALI_TILER_HIERARCHY_MASK;
}

bool
MALI_PER_ARCH(cmd_render_tiler)(struct mali_cmd_buffer *cmd)
{
   struct mali_render_state *r = &cmd->gfx.render;
   assert(r->active);
   if (r->tiler)
      return true;
   if (!mali_fb_alloc_tsd(cmd, r))
      return false;

   /* The barriers owed to vtc work; the batch may close here. */
   if (!mali_jm_cmd_vtc(cmd))
      return false;
   struct mali_jm_batch *b = &cmd->jm.cur;
   if (b->heap_slot < 0 && !mali_jm_heap_take(cmd, b))
      return false;

   const uint32_t layers = MAX2(r->desc.layer_count, 1);
   struct mali_ptr p = mali_cmd_alloc(cmd, (uint64_t)layers * pan_size(TILER_CONTEXT), 64);
   if (!p.cpu)
      return false;

   /* Every word, packed on the CPU and copied (command memory is
    * write-combined). The tiler writes the polygon list and the state
    * words; a command buffer that runs again gets the template back. */
   struct mali_tiler_context_packed tc;
   const uint32_t mask = hierarchy_mask(cmd->dev);
   for (uint32_t i = 0; i < layers; i++) {
      pan_pack(&tc, TILER_CONTEXT, cfg) {
         cfg.polygon_list = 0;
         cfg.hierarchy_mask = mask;
         cfg.sample_pattern = MALI_SAMPLE_PATTERN_SINGLE_SAMPLED;
         cfg.first_provoking_vertex = true;
         cfg.fb_width = r->desc.width;
         cfg.fb_height = r->desc.height;
         cfg.layer_count = 1;
         cfg.layer_offset = 0;
         cfg.heap = b->heap_desc;
      }
      void *dst = (uint8_t *)p.cpu + (uint64_t)i * pan_size(TILER_CONTEXT);
      memcpy(dst, &tc, sizeof(tc));
      mali_jm_cmd_note_reset(cmd, dst, &tc, sizeof(tc));
   }
   r->tiler = p.gpu;
   r->tiler_cpu = p.cpu;
   r->td_count = layers;
   b->est_heap_bytes += MALI_JM_HEAP_PASS_EST;

   cmd->gfx.draw.dirty |= MALI_GFX_DIRTY_PASS;
   return true;
}

/* ---------------------------------------------------------------------- */
/* Full-screen jobs                                                        */

/*
 * Fullscreen jobs (type 12, 128 bytes) running the Draw descriptor dcd over
 * rect in layers [base_layer, base_layer + layer_count), one per layer, on
 * the layer's tiler context. They are tiler-side jobs in the batch's vtc
 * chain (Dependency 2 = the previous tiler-side job). barrier: the header's
 * Barrier bit, which the blob sets on its in-pass clears.
 */
static void
fullscreen_jobs(struct mali_cmd_buffer *cmd, uint64_t dcd, const VkRect2D *rect,
                uint32_t base_layer, uint32_t layer_count, bool barrier)
{
   struct mali_render_state *r = &cmd->gfx.render;
   if (!MALI_PER_ARCH(cmd_render_tiler)(cmd)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   const int minx = MAX2(rect->offset.x, 0), miny = MAX2(rect->offset.y, 0);
   const int maxx = MIN2(rect->offset.x + (int)rect->extent.width, (int)r->desc.width) - 1;
   const int maxy = MIN2(rect->offset.y + (int)rect->extent.height, (int)r->desc.height) - 1;
   if (maxx < minx || maxy < miny)
      return;

   /* Bytes 32-127 of the job: the sections, and zero between them. */
   uint8_t tmpl[128] __attribute__((aligned(16)));
   memset(tmpl, 0, sizeof(tmpl));
   /* As Mesa's v9 full-screen draw (jm_launch_draw_fullscreen): no draw
    * mode, no indices, the genxml defaults otherwise. */
   pan_section_pack(tmpl, FULLSCREEN_JOB, PRIMITIVE, cfg) {
      cfg.scissor_array_enable = false;
   }
   pan_section_pack(tmpl, FULLSCREEN_JOB, DCD, cfg) {
      cfg.address = dcd;
   }
   pan_section_pack(tmpl, FULLSCREEN_JOB, SCISSOR, cfg) {
      cfg.scissor_minimum_x = minx;
      cfg.scissor_minimum_y = miny;
      cfg.scissor_maximum_x = maxx;
      cfg.scissor_maximum_y = maxy;
   }

   struct mali_jm_chain *c = &cmd->jm.cur.vtc;
   const uint32_t end = MIN2(base_layer + layer_count, r->td_count);
   for (uint32_t l = base_layer; l < end; l++) {
      pan_section_pack(tmpl, FULLSCREEN_JOB, TILER, cfg) {
         cfg.address = mali_jm_pass_tiler(r, l);
      }
      struct mali_ptr job = mali_jm_cmd_add_job(cmd, c, MALI_JOB_TYPE_FULLSCREEN,
                                                pan_size(FULLSCREEN_JOB), barrier, 0);
      if (!job.cpu)
         return;
      memcpy((uint8_t *)job.cpu + 32, tmpl + 32, sizeof(tmpl) - 32);
   }
   r->draws++;
}

void
MALI_PER_ARCH(cmd_run_fullscreen)(struct mali_cmd_buffer *cmd, uint64_t dcd, const VkRect2D *rect,
                                  uint32_t base_layer, uint32_t layer_count)
{
   fullscreen_jobs(cmd, dcd, rect, base_layer, layer_count, true);
}

/*
 * A by-region fragment -> fragment barrier inside a pass: later
 * fragments of the pass wait in each tile for earlier ones. One draw per
 * layer whose Draw descriptor sets Primitive Barrier, with no fragment
 * shader, both forward-pixel-kill bits clear and a depth/stencil
 * descriptor that tests ALWAYS and writes nothing (panvk's v11
 * cmd_fb_barrier and the blob both). The blob draws it as a 4-vertex quad
 * in a Malloc Vertex job with an internal vertex shader; a Fullscreen job
 * carrying the same descriptor is Mesa's shape and needs no shader. No
 * Cache Flush job and no header Barrier bit: the ordering is the tiler's
 * and the fragment front end's, inside each tile.
 */
void
MALI_PER_ARCH(cmd_fb_barrier)(struct mali_cmd_buffer *cmd)
{
   struct mali_render_state *r = &cmd->gfx.render;
   struct mali_ptr zsd = mali_cmd_alloc(cmd, pan_size(DEPTH_STENCIL), 32);
   struct mali_ptr dcd = mali_cmd_alloc(cmd, pan_size(DRAW), 64);
   if (!zsd.cpu || !dcd.cpu)
      return;

   pan_cast_and_pack(zsd.cpu, DEPTH_STENCIL, cfg) {
      cfg.stencil_test_enable = false;
      cfg.depth_write_enable = false;
      cfg.depth_function = MALI_FUNC_ALWAYS;
   }
   pan_cast_and_pack(dcd.cpu, DRAW, cfg) {
      cfg.flags_0.allow_forward_pixel_to_kill = false;
      cfg.flags_0.allow_forward_pixel_to_be_killed = false;
      cfg.flags_0.primitive_barrier = true;
      cfg.flags_0.occlusion_query = MALI_OCCLUSION_MODE_DISABLED;
      cfg.vertex_array.packet = true;
      cfg.minimum_z = 0.0f;
      cfg.maximum_z = 1.0f;
      cfg.depth_stencil = zsd.gpu;
   }
   const VkRect2D all = {{0, 0}, {r->desc.width, r->desc.height}};
   fullscreen_jobs(cmd, dcd.gpu, &all, 0, MAX2(r->desc.layer_count, 1), false);
}

/* ---------------------------------------------------------------------- */
/* End                                                                     */

/* One Fragment job per layer: bounding box in 16-pixel tiles (Mesa
 * pan_emit_fragment_job_payload), the layer's FBD pointer, tagged. */
static void
fragment_jobs(struct mali_cmd_buffer *cmd, const struct mali_render_state *r, uint64_t fbds,
              uint32_t fbd_size)
{
   struct mali_jm_chain *c = mali_jm_cmd_frag(cmd, false);
   if (!c)
      return;

   uint8_t tmpl[128] __attribute__((aligned(16)));
   memset(tmpl, 0, sizeof(tmpl));
   const uint32_t layers = MAX2(r->desc.layer_count, 1);
   for (uint32_t l = 0; l < layers; l++) {
      pan_section_pack(tmpl, FRAGMENT_JOB, PAYLOAD, cfg) {
         cfg.bound_min_x = r->minx >> MALI_TILE_SHIFT;
         cfg.bound_min_y = r->miny >> MALI_TILE_SHIFT;
         cfg.bound_max_x = r->maxx >> MALI_TILE_SHIFT;
         cfg.bound_max_y = r->maxy >> MALI_TILE_SHIFT;
         cfg.framebuffer = fbds + (uint64_t)l * fbd_size;
      }
      struct mali_ptr job =
         mali_jm_cmd_add_job(cmd, c, MALI_JOB_TYPE_FRAGMENT, sizeof(tmpl), false, 0);
      if (!job.cpu)
         return;
      memcpy((uint8_t *)job.cpu + 32, tmpl + 32, sizeof(tmpl) - 32);
   }
}

void
MALI_PER_ARCH(cmd_render_end)(struct mali_cmd_buffer *cmd)
{
   struct mali_render_state *r = &cmd->gfx.render;
   if (!r->active)
      return;

   if (MALI_PER_ARCH(fb_pass_has_work)(r) && !vk_command_buffer_has_error(&cmd->vk)) {
      bool ok = mali_fb_alloc_tsd(cmd, r);
      uint32_t fbd_size = 0;
      const uint64_t fbds = ok ? MALI_PER_ARCH(fb_build)(cmd, r, &fbd_size) : 0;
      if (!fbds) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      } else {
         /* No CRC state on v9 (see the file comment). */
         assert(r->crc_rt < 0);
         /* Each layer's descriptors are 64-byte aligned, so the tag of the
          * first pointer holds for all. */
         assert(fbd_size % 64 == 0);
         MALI_PER_ARCH(fb_fill_tsd)(cmd, r);
         fragment_jobs(cmd, r, fbds, fbd_size);
         cmd->jm.cur.passes++;
      }
   }

   r->active = false;
   cmd->gfx.draw.dirty = MALI_GFX_DIRTY_ALL;
}

/* CRCs (transaction elimination) are off on v9 until the device shows
 * they pay; with no CRC state there is nothing to
 * invalidate. */
void
MALI_PER_ARCH(cmd_crc_invalidate)(struct mali_cmd_buffer *cmd, const struct mali_image *image)
{
}
