/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Render passes and draws: what the command buffer keeps about the
 * current render pass and the graphics state, and the internal interfaces
 * between mali_fb.c (pass layout, framebuffer descriptors, frame shaders),
 * mali_cmd_state.[ch] (binds, the per-draw descriptor builders and dirty
 * bits), mali_cmd_render.c (tiler descriptors, fragment jobs),
 * mali_cmd_draw.c (draws, full-screen draws) and mali_cmd_meta_gfx.c
 * (internal fragment shaders for preloads, clears and blits). mali_fb.c,
 * mali_cmd_state.[ch] and mali_cmd_meta_gfx.c are shared with the
 * job-manager back half and built per arch, as are the back halves'
 * functions declared here (mali_cmd_render.c and mali_cmd_draw.c on v11,
 * the mali_jm_* files on v9): every function below has a mali_v9_ and a
 * mali_v11_ variant (MALI_PER_ARCH).
 */

#ifndef MALI_CMD_GFX_H
#define MALI_CMD_GFX_H

#include <stdbool.h>
#include <stdint.h>

#include "util/format/u_formats.h"
#include "vulkan/vulkan_core.h"

#include "kbase/kbase.h"
#include "util/list.h"

#include "mali_arch.h"
#include "mali_descriptor_set_layout.h"
#include "mali_pipeline.h"

struct mali_cmd_buffer;
struct mali_descriptor_set;
struct mali_device;
struct mali_graphics_pipeline;
struct mali_image;
struct mali_shader;

/* ---------------------------------------------------------------------- */
/* Command-buffer memory (both back halves)                                */

/* Command-buffer memory comes in 64 KiB slabs recycled through the device;
 * anything larger gets its own allocation, freed at reset. */
#define MALI_CMD_SLAB_SIZE (64 * 1024)

struct mali_cmd_slab {
   struct list_head link;
   struct mali_kbase_bo bo;
   bool recycle;             /* a standard slab (else a dedicated BO) */
};

struct mali_ptr {
   void *cpu;
   uint64_t gpu;
};

/* The slow path of mali_cmd_alloc (the back half's command buffer
 * header): a new slab, or a dedicated allocation for a large block. */
MALI_PER_ARCH_DECL(struct mali_ptr, cmd_alloc_slow,
                   (struct mali_cmd_buffer *cmd, uint64_t size, uint64_t align));

/* Descriptor sets bound at one bind point, with their dynamic offsets. */
struct mali_desc_state {
   struct mali_descriptor_set *sets[MALI_MAX_SETS];
   uint32_t dyn_offsets[MALI_MAX_SETS][MALI_MAX_DYNAMIC_BUFFERS];
};

/* Layers one Tiler Context covers: 8 on v11, 1 on v9 (Mesa v9, panvk JM
 * and the T820 blob use one per layer). Per-arch code only. */
#if defined(PAN_ARCH) && PAN_ARCH < 10
#define MALI_LAYERS_PER_TILER_CTX 1
#else
#define MALI_LAYERS_PER_TILER_CTX 8
#endif
#define MALI_MAX_RENDER_LAYERS 256
#define MALI_MAX_VBS 16

/* The fixed hierarchy mask the blob runs every ARMSX2-sized pass with
 * (levels 1, 3, 5, 7). */
#define MALI_TILER_HIERARCHY_MASK 0xaa

enum mali_att_load {
   MALI_ATT_LOAD_DONT_CARE = 0,
   MALI_ATT_LOAD_CLEAR,
   MALI_ATT_LOAD_LOAD,
};

/*
 * One attachment plane as the framebuffer descriptors see it: a plane of
 * an image, one mip level, a range of layers starting at `layer`.
 */
struct mali_fb_target {
   const struct mali_image *image;  /* NULL: no attachment */
   unsigned plane;
   unsigned level;
   unsigned layer;
   enum pipe_format format;         /* colour: view format; Z/S: the plane's */
   enum mali_att_load load;
   bool store;
   /* The frame shader writes this target: a load, or a clear that a
    * partly covered tile cannot do through the tile buffer. */
   bool preload;
   /* Every tile of the bounding box is written back (a blit's target). */
   bool always_write;
};

struct mali_meta_fs_key;
struct mali_meta_fs_push;

/* An internal pass whose colour frame shader is given (blits): it runs on
 * every tile and writes render target 0. */
struct mali_render_blit {
   const struct mali_meta_fs_key *key;
   const struct mali_meta_fs_push *push;
   const uint32_t (*textures)[8];
   unsigned texture_count;
   const uint32_t *sampler;
};

/* What a render pass instance is made of: vkCmdBeginRendering's info, or an
 * internal pass (image clears, blits). */
struct mali_render_desc {
   uint32_t width, height;          /* framebuffer size */
   VkRect2D area;                   /* render area */
   uint32_t layer_count;
   uint32_t rt_count;               /* colour attachments (0..8) */
   struct mali_fb_target rt[8];
   VkClearColorValue clear_color[8];
   struct mali_fb_target z, s;
   float clear_depth;
   uint8_t clear_stencil;
   const struct mali_render_blit *blit;
};

struct mali_render_state {
   bool active;
   /* A draw of the pass copied uniform-buffer words into a FAU block with
    * prebuilt stores the stream builder does not track (emit_ubo_push);
    * flush_tiling waits for them. */
   bool ubo_stores;
   struct mali_render_desc desc;

   /* Tile-buffer layout (pan_select_fb_tile_size). */
   uint32_t rt_count;               /* >= 1: the hardware wants one */
   uint32_t tile_size;              /* pixels per tile */
   uint32_t cbuf_alloc;             /* colour bytes per tile, 1 KiB aligned */
   uint32_t rt_offset[8];           /* tile-buffer offset of each target */

   /* Bounding box of the fragment job, inclusive, and whether the render
    * area leaves tiles partly covered. */
   uint32_t minx, miny, maxx, maxy;
   bool partial_tiles;

   /* Transaction elimination: the colour target whose CRCs the pass
    * keeps (-1: none) and the low word of its CRC Clear Color without the
    * seed (mali_fb_build). */
   int crc_rt;
   uint32_t crc_clear_lo;

   /* Set by the first draw or full-screen draw of the pass. */
   uint64_t tiler;                  /* Tiler Contexts, 0 if nothing tiled */
   void *tiler_cpu;
   uint32_t td_count;
   uint64_t tsd;                    /* the pass's Local Storage descriptor */
   void *tsd_cpu;
   uint32_t tls_size;               /* largest per-thread TLS of the pass */
   uint32_t draws;                  /* RUN_IDVS + RUN_FULLSCREEN in the pass */
   /* Timing regions of the pass, 0 when not timed. */
   uint32_t measure_vt, measure_frag;
};

/* Graphics state registers the next draw has to (re)write. */
enum mali_gfx_dirty {
   MALI_GFX_DIRTY_PIPELINE = 1u << 0,   /* SPDs, tiler flags, DCD words, sizes */
   MALI_GFX_DIRTY_VS_SRT   = 1u << 1,
   MALI_GFX_DIRTY_FS_SRT   = 1u << 2,
   MALI_GFX_DIRTY_VS_FAU   = 1u << 3,
   MALI_GFX_DIRTY_FS_FAU   = 1u << 4,
   MALI_GFX_DIRTY_BLEND    = 1u << 5,
   MALI_GFX_DIRTY_ZSD      = 1u << 6,
   MALI_GFX_DIRTY_VIEWPORT = 1u << 7,   /* scissor box, depth clamps */
   MALI_GFX_DIRTY_INDEX    = 1u << 8,
   MALI_GFX_DIRTY_PASS     = 1u << 9,   /* TILER_CTX, TSD */
   MALI_GFX_DIRTY_ALL      = (1u << 10) - 1,
};

struct mali_gfx_draw_state {
   struct mali_graphics_pipeline *pipeline;
   uint32_t dirty;

   struct {
      uint64_t addr;
      uint64_t size;
   } vb[MALI_MAX_VBS];

   struct {
      uint64_t addr;
      uint32_t size;
      uint32_t index_size;          /* bytes, 0 when none bound */
   } ib;

   /* The last values the FAU blocks were built with. */
   int32_t first_vertex;
   uint32_t base_instance;
   uint64_t blend_descs[8];         /* fs.blend_descs, from the last blend build */
   uint32_t blend_shader_mask;      /* targets blended by a shader, last blend build */

   /* Vertex input the draws read: the bound pipeline's static state, or
    * the command buffer's dynamic state where the pipeline leaves it
    * dynamic (mali_cmd_bind_graphics). */
   const struct vk_vertex_input_state *vi;
   const uint16_t *vi_strides;

   /* The mali_gfx_bind table keys of the tables in SRT_0 and SRT_2. */
   uint32_t vs_srt_key, fs_srt_key;
   /* Descriptor sets bound since each stage's table was built. */
   uint32_t vs_sets_dirty, fs_sets_dirty;

   /* What the IDVS staging registers 0..63 hold, per 32-bit register,
    * where regs_valid says it is known: a draw writes only the registers
    * whose value changes. Only mali_cmd_draw.c writes these registers;
    * anything else that does clears regs_valid. */
   uint64_t regs_valid;
   uint32_t regs[64];

   /* The pipeline's words repacked with this command buffer's dynamic
    * state, for pipelines whose baked words depend on it. */
   struct mali_gfx_baked dyn_baked;
};

/* ---------------------------------------------------------------------- */
/* Compute dispatch (CSF: mali_cmd_dispatch.c; JM: mali_jm_cmd_dispatch.c) */

/*
 * One compute dispatch: FAU from push constants and the compute system
 * values, the resource table from the bound sets (desc may be NULL for
 * shaders without descriptors), TLS/WLS; RUN_COMPUTE on the compute
 * subqueue (CSF) or a Compute job in the open batch's vtc chain (JM).
 */
MALI_PER_ARCH_DECL(void, cmd_dispatch_shader,
                   (struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
                    const struct mali_desc_state *desc, const void *push, uint32_t push_size,
                    const uint32_t base[3], const uint32_t groups[3]));

/* A dispatch of an internal shader whose resource table 0 holds a sampler
 * at index 0 and `textures` (packed Texture descriptors) from index 1. */
MALI_PER_ARCH_DECL(void, cmd_dispatch_meta,
                   (struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
                    const void *push, uint32_t push_size, const uint32_t groups[3],
                    const uint32_t (*textures)[8], unsigned texture_count));

/* The internal copy/fill shaders, blend shaders and internal fragment
 * shaders of a device, freed at device destruction (mali_cmd_copy.c). */
MALI_PER_ARCH_DECL(void, meta_finish, (struct mali_device *dev));

/* ---------------------------------------------------------------------- */
/* mali_cmd_render.c (the back half's render pass; JM: mali_jm_cmd_render.c) */

/* Start a render pass instance. rt/z/s images must be bound. */
MALI_PER_ARCH_DECL(void, cmd_render_begin,
                   (struct mali_cmd_buffer *cmd, const struct mali_render_desc *desc));
/* End it: tiling finished, fragment jobs, heap release. */
MALI_PER_ARCH_DECL(void, cmd_render_end, (struct mali_cmd_buffer *cmd));

/* Before the first tiled job of the pass: tiler contexts, the pass TSD,
 * HEAP_OPERATION{Vertex/Tiler Started}. False on allocation failure (the
 * command buffer has the error). */
MALI_PER_ARCH_DECL(bool, cmd_render_tiler, (struct mali_cmd_buffer *cmd));

/* ---------------------------------------------------------------------- */
/* mali_fb.c: the frontend-neutral part of a render pass                   */

/*
 * The start of a pass: resets gfx.render to desc, picks the tile size and
 * tile-buffer layout, the bounding box, and which targets the frame
 * shaders preload (loads, and border loads of partly covered tiles); marks
 * every draw state group dirty. True when some target is preloaded: the
 * back half then makes earlier attachment writes visible to the texture
 * unit.
 */
MALI_PER_ARCH_DECL(bool, fb_begin,
                   (struct mali_cmd_buffer *cmd, const struct mali_render_desc *desc));

/* Whether the pass needs fragment work: it tiled something, or a stored
 * target is cleared or always written. */
MALI_PER_ARCH_DECL(bool, fb_pass_has_work, (const struct mali_render_state *r));

/*
 * The pass's framebuffer descriptors, one per layer, each followed by its
 * ZS/CRC extension (when there is depth, stencil or a CRC target) and its
 * render targets; the frame shaders' Draw descriptors. Chooses the CRC
 * target (r->crc_rt, r->crc_clear_lo: the CRC Clear Color's low word
 * without the seed, which the back half fills in). The tiler pointer of
 * layer l is r->tiler + (l / MALI_LAYERS_PER_TILER_CTX) Tiler Contexts.
 * Returns the first descriptor's address tagged for the fragment job's
 * FBD pointer and *size_out the size of one layer's descriptors; 0 on
 * failure.
 */
MALI_PER_ARCH_DECL(uint64_t, fb_build,
                   (struct mali_cmd_buffer *cmd, struct mali_render_state *r,
                    uint32_t *size_out));

/* Fills the pass's Local Storage descriptor (allocated by
 * mali_fb_alloc_tsd, mali_cmd_state.h): TLS for the largest per-thread
 * need of the pass's shaders, no workgroup memory. */
MALI_PER_ARCH_DECL(void, fb_fill_tsd,
                   (struct mali_cmd_buffer *cmd, struct mali_render_state *r));

/* ---------------------------------------------------------------------- */
/* mali_cmd_state.c                                                        */

/* The per-thread TLS buffer of the command buffer, grown to at least
 * tls_size bytes per thread. 0 on failure. */
MALI_PER_ARCH_DECL(uint64_t, cmd_tls_buffer, (struct mali_cmd_buffer *cmd, uint32_t tls_size));

/*
 * Transaction elimination (CRC).
 *
 * mali_cmd_crc_invalidate: the image's level 0 was or will be written
 * other than by a render pass that keeps its CRCs (a copy, a barrier from
 * an undefined layout, a pass where it is not the CRC target). Bumps the
 * CRC seed in the image's header from the fragment stream, which is where
 * every pass reads it, so the bump lands between the passes before and
 * after it in recording order. No-op for images without CRC.
 */
MALI_PER_ARCH_DECL(void, cmd_crc_invalidate,
                   (struct mali_cmd_buffer *cmd, const struct mali_image *image));

/* Knobs for the device CRC test; the driver runs with the defaults
 * (mali_image.c, compiled once: one set for both arches). */
struct mali_crc_options {
   bool disable;               /* no CRC at all */
   bool skip_copy_invalidate;  /* tests only: copies leave the CRC valid */
   bool always_preload;        /* colour preload runs on every tile */
   bool empty_tile_read;
   bool empty_tile_write;
};
/* Hidden like everything in the .so, said here so the per-pass code that
 * reads it from another file addresses it directly, not through the GOT. */
extern struct mali_crc_options mali_crc_options __attribute__((visibility("hidden")));

/* A render-target Blend descriptor's words for a colour target (the
 * internal part the fragment shader's BLEND instruction reads;
 * mali_fb.c). */
MALI_PER_ARCH_DECL(void, pack_opaque_blend,
                   (enum pipe_format format, unsigned rt, uint32_t out[4]));

/* vkCmdBindPipeline for a graphics pipeline. */
MALI_PER_ARCH_DECL(void, cmd_bind_graphics,
                   (struct mali_cmd_buffer *cmd, struct mali_graphics_pipeline *p));

/* The FAU block of a graphics-stage shader: its used sysval words, then
 * its used push-constant words, then promoted constants. `sysvals` is a
 * struct mali_graphics_sysvals. Returns the GPU address with the word
 * count in bits 56+, or 0 on failure / no FAU. */
MALI_PER_ARCH_DECL(uint64_t, cmd_gfx_fau,
                   (struct mali_cmd_buffer *cmd, const struct mali_shader *s,
                    const void *sysvals, const void *push, uint32_t push_size));

/* ---------------------------------------------------------------------- */
/* mali_cmd_draw.c                                                         */

/* Set RUN_IDVS bit 55 (default true; tests only change it). */
extern bool mali_idvs_bit55;

/* A primitive barrier inside the render pass (by-region dependencies). */
MALI_PER_ARCH_DECL(void, cmd_fb_barrier, (struct mali_cmd_buffer *cmd));

/*
 * A full-screen fragment draw on the vertex/tiler subqueue: RUN_FULLSCREEN
 * of the Draw descriptor `dcd` over `rect` of layers [base_layer,
 * base_layer + layer_count). Clobbers the scissor, tiler flags and tiler
 * context registers; the next draw rewrites them.
 */
MALI_PER_ARCH_DECL(void, cmd_run_fullscreen,
                   (struct mali_cmd_buffer *cmd, uint64_t dcd, const VkRect2D *rect,
                    uint32_t base_layer, uint32_t layer_count));

/* ---------------------------------------------------------------------- */
/* mali_cmd_meta_gfx.c                                                     */

/*
 * Internal fragment shaders. A key says what each output does; the shader
 * reads its sources from resource table 0 (index 0: a sampler, 1 + i:
 * texture i in output order) and its constants from push constants
 * (struct mali_meta_fs_push).
 */
enum mali_meta_fs_op {
   MALI_META_FS_NONE = 0,
   MALI_META_FS_LOAD,          /* texel fetch at the fragment's position */
   MALI_META_FS_CLEAR,         /* the push-constant clear value */
   MALI_META_FS_CLEAR_IN_AREA, /* clear inside the push-constant rect, load outside */
   MALI_META_FS_BLIT,          /* inside the area: sample texture 0 at the
                                * transformed position; outside: fetch
                                * texture 1 (the target itself) */
   MALI_META_FS_COPY,          /* raw copy into an AFBC target read as a
                                * uint format of its block size: inside the
                                * area, the source block at the fragment's
                                * position + copy_delta (texture 0 when
                                * copy_src_tex, else a raw surface in memory);
                                * outside, texture copy_src_tex ? 1 : 0 (the
                                * target itself). Render target 0 only. */
};

enum mali_meta_fs_type {
   MALI_META_FS_FLOAT = 0,
   MALI_META_FS_UINT,
   MALI_META_FS_SINT,
};

struct mali_meta_fs_key {
   uint8_t rt_op[8];           /* enum mali_meta_fs_op */
   uint8_t rt_type[8];         /* enum mali_meta_fs_type */
   uint8_t z_op, s_op;
   uint8_t layered;            /* sources are arrays indexed by the layer */
   uint8_t copy_src_tex;       /* COPY: the source is texture 0 */
   uint8_t copy_elem_log2;     /* COPY: log2 of the block size in bytes */
   uint8_t pad[3];
};

struct mali_meta_fs_push {
   /* Render area for CLEAR_IN_AREA, inclusive: min x, min y, max x, max y. */
   int32_t area[4];
   /* Blit: source = (x, y) * scale + offset, in normalized coordinates;
    * z = source layer. */
   float blit_scale[2];
   float blit_offset[2];
   float blit_layer;
   float clear_depth;
   uint32_t clear_stencil;
   uint32_t pad;
   uint32_t clear_color[8][4];
   /* COPY: source block = fragment position + delta (in blocks); a raw
    * source surface is addressed as the raw compute copy does
    * (mali_meta_copy_push), its block (0, 0) of layer 0 at copy_src. */
   int32_t copy_delta[2];
   uint32_t copy_row_stride;
   uint32_t copy_tile_shift;
   uint64_t copy_src;
   uint64_t copy_layer_stride;
};

/* The compiled shader for a key, cached for the device's lifetime. NULL on
 * failure (the command buffer gets VK_ERROR_OUT_OF_DEVICE_MEMORY). */
MALI_PER_ARCH_DECL(const struct mali_shader *, meta_fs_get,
                   (struct mali_cmd_buffer *cmd, const struct mali_meta_fs_key *key));
MALI_PER_ARCH_DECL(void, meta_gfx_finish, (struct mali_device *dev));

/*
 * Raw block copies between image planes and buffers (mali_cmd_image.c),
 * one invocation per texel block of 2^elem_log2 bytes. A surface is
 * addressed as linear rows or as u-interleaved tiles of 2^tile_shift
 * blocks square (pan_tiling.c's layout; tile_shift 0 is linear).
 */
struct mali_meta_copy_push {
   uint64_t src, dst;
   uint64_t src_layer_stride, dst_layer_stride;
   uint32_t src_row_stride, dst_row_stride;
   int32_t src_x, src_y, dst_x, dst_y;
   uint32_t width, height;           /* blocks */
   uint32_t src_tile_shift, dst_tile_shift;
};

#define MALI_META_COPY_WG 8

MALI_PER_ARCH_DECL(const struct mali_shader *, meta_copy_get,
                   (struct mali_cmd_buffer *cmd, unsigned elem_log2));

/* The same copy with the source read through texture 0 (a 2D array of the
 * copied layers, in the uint format of the block size) at (src_x + x,
 * src_y + y, z): for AFBC sources, which have no raw addressing. */
MALI_PER_ARCH_DECL(const struct mali_shader *, meta_copy_tex_get,
                   (struct mali_cmd_buffer *cmd, unsigned elem_log2));

/*
 * Pack into `out` (a Draw descriptor, 64-byte aligned) the frame-shader or
 * full-screen draw that runs `fs` with `key`'s outputs over the current
 * render pass. `textures` are packed Texture descriptors in output order
 * (one per output that reads), `sampler` a packed Sampler (NULL: nearest).
 * False on failure.
 */
MALI_PER_ARCH_DECL(bool, meta_fs_dcd,
                   (struct mali_cmd_buffer *cmd, const struct mali_shader *fs,
                    const struct mali_meta_fs_key *key, const struct mali_meta_fs_push *push,
                    const uint32_t (*textures)[8], unsigned texture_count,
                    const uint32_t *sampler, bool frame_shader, void *out));

/* A 2D (array) Texture descriptor for one level and a layer range of an
 * image plane, read as `format` (mali_image_view.c). The plane
 * descriptors go into command-buffer memory. False on failure. */
MALI_PER_ARCH_DECL(bool, cmd_pack_plane_texture,
                   (struct mali_cmd_buffer *cmd, const struct mali_image *image,
                    unsigned plane, enum pipe_format format, unsigned level,
                    unsigned first_layer, unsigned layer_count, uint32_t out[8]));

#endif
