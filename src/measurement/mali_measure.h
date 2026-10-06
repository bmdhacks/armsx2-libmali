/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Measurement:
 *
 *  - timing: GPU timestamps around every render pass (vertex/tiler
 *    and fragment side separately), every compute dispatch and, when asked
 *    for, every draw. The STORE_STATE{Timestamp} instructions are written
 *    when the vkCmd* is recorded, into slots in command memory; the
 *    results are read once the submit has completed, without waiting for
 *    it, and written as CSV rows.
 *  - capture: at vkQueueSubmit, the words written into the kernel rings,
 *    the recorded streams they CALL and the command memory those streams
 *    point at, one file per submit, for tools/mali_cs_decode.py.
 *
 * One configuration string, from the environment variable LIBMALI_MEASURE
 * or, on Android, the property debug.libmali.measure:
 *
 *   timing | draws | csf | both, then any of dir=PATH chunk=N ring=N
 *   start=N count=N csfmax=N trigger=PATH, comma-separated
 *
 * With neither set, dev->measure and every cmd->measure are NULL and the
 * driver pays one pointer test per hook.
 */

#ifndef MALI_MEASURE_H
#define MALI_MEASURE_H

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "util/list.h"
#include "util/u_dynarray.h"

/* The command-stream back half's region API (mali_measure_begin) takes its
 * subqueue enum. The v9 per-arch files include this header only for the
 * shader dump directory (mali_measure_shader_dir). */
#if !defined(PAN_ARCH) || PAN_ARCH >= 10
#include "mali_cs.h"
#endif

struct mali_cmd_buffer;
struct mali_device;
struct mali_queue;
struct vk_queue_submit;

/* ---------------------------------------------------------------------- */
/* Configuration (config.c)                                                */

struct mali_measure_config {
   bool timing;            /* passes and dispatches */
   bool draws;             /* also every draw (implies timing) */
   bool csf;               /* command-stream capture */
   bool shaders;           /* every compiled shader, to <dir>/shaders */
   char dir[512];          /* output directory, created if missing */
   uint32_t chunk;         /* submits per timing file */
   uint32_t ring;          /* timing files and captures kept; 0 = all */
   uint64_t start;         /* first submit written */
   uint64_t count;         /* submits written from start; 0 = no limit */
   uint32_t csf_max;       /* captures written at most; 0 = no limit */
   char trigger[512];      /* if set: write only while this file exists */
};

/* Parse a configuration string. Unknown words and bad numbers are an
 * error (*err points at the word). An empty string or "off" leaves every
 * mode off. */
bool mali_measure_config_parse(const char *str, struct mali_measure_config *cfg,
                               const char **err);

/* The configuration of this process: LIBMALI_MEASURE, else (Android)
 * debug.libmali.measure. False when nothing is enabled. */
bool mali_measure_config_get(struct mali_measure_config *cfg);

/* The directory compiled shaders are written to ("shaders" mode:
 * <dir>/shaders, created on first use), or NULL when the mode is off.
 * Read once per process. */
const char *mali_measure_shader_dir(void);

/* ---------------------------------------------------------------------- */
/* Per device                                                              */

enum mali_measure_kind {
   MALI_MEASURE_PASS_VT,     /* a render pass, vertex/tiler subqueue */
   MALI_MEASURE_PASS_FRAG,   /* a render pass, fragment subqueue */
   MALI_MEASURE_DISPATCH,    /* RUN_COMPUTE, internal copies included */
   MALI_MEASURE_DRAW,        /* RUN_IDVS */
   MALI_MEASURE_KIND_COUNT,
};

struct mali_measure_file {
   char name[40];
   bool capture;            /* else a timing file */
   bool deleted;            /* rotated out (ring) */
   uint64_t first_submit, last_submit;
   uint64_t rows;           /* timing rows, or bytes of a capture */
};

struct mali_measure {
   struct mali_device *dev;
   struct mali_measure_config cfg;
   char cfg_str[256];

   /* Guards everything below: submits, command-buffer resets on other
    * threads, and device loss all write here. */
   pthread_mutex_t lock;

   /* Command buffers submitted whose timestamps have not been read yet. */
   struct list_head pending;

   /* GPU timestamp rate and its tie to CLOCK_MONOTONIC. */
   uint64_t hz;
   uint64_t cntfrq;          /* CNTFRQ_EL0, 0 where unreadable */
   uint64_t measured_hz;     /* from two kbase samples at start */
   const char *hz_source;
   uint64_t cal_ts, cal_mono;           /* at device creation */
   uint64_t end_ts, end_mono;           /* at device destruction */
   bool cal_ok, end_ok;

   /* Output. */
   FILE *timing;             /* the open timing file, or NULL */
   uint32_t timing_index;    /* number of the next timing file */
   uint64_t timing_first;    /* first submit that belongs in the open file */
   struct util_dynarray files;   /* struct mali_measure_file, in order */
   uint32_t live_timing, live_captures;   /* files not rotated out */
   uint32_t captures;        /* captures written */
   uint64_t submits;         /* last submit number seen */
   uint64_t rows, rows_missing, rows_reordered, dropped;
   struct util_dynarray events;  /* char *, e.g. device loss */
   bool write_failed;
};

/* From vkCreateDevice, after the kbase context exists and before the
 * queues. Leaves dev->measure NULL when measurement is off or its output
 * directory cannot be made (logged). */
void mali_measure_init(struct mali_device *dev);

/* From vkDestroyDevice, before the queues go: reads what has completed,
 * closes the files and writes the manifest. */
void mali_measure_finish(struct mali_device *dev);

/* An event for the manifest (device loss). */
void mali_measure_event(struct mali_device *dev, const char *fmt, ...)
   __attribute__((format(printf, 2, 3)));

/* ---------------------------------------------------------------------- */
/* Per command buffer (timing.c)                                           */

struct mali_measure_region {
   uint8_t kind;             /* enum mali_measure_kind */
   uint8_t sq;               /* enum mali_subqueue */
   uint32_t index;           /* pass, dispatch or draw number in the command buffer */
   uint32_t info[4];         /* pass: w, h, render targets, draws;
                                dispatch: groups x, y, z, local size x*y*z;
                                draw: count, instances, indexed, pass */
   uint64_t shader;          /* dispatch, draw: first 8 bytes of the (fragment) shader key */
   uint32_t extra;           /* draw: render targets blended by a blend shader */
   uint64_t *slot;           /* CPU view: begin, end */
   uint64_t slot_va;
};

struct mali_measure_cmd {
   struct mali_measure *m;
   bool draws;
   /* In m->pending, under m->lock, from submit until read. */
   struct list_head link;
   bool pending;
   bool active;              /* its submit is inside the output window */
   uint64_t submit;
   uint32_t submit_pos;      /* index in the submit's command buffers */
   uint32_t sq_mask;
   struct util_dynarray regions;   /* struct mali_measure_region */
   uint64_t *slots;          /* free slots in the current block */
   uint64_t slots_va;
   uint32_t slots_left;
};

void mali_measure_cmd_create(struct mali_cmd_buffer *cmd);
/* Reset and destroy read the last submit's results first: Vulkan
 * guarantees it has completed. */
void mali_measure_cmd_reset(struct mali_cmd_buffer *cmd);
void mali_measure_cmd_destroy(struct mali_cmd_buffer *cmd);

/*
 * A timed region on subqueue sq: begin writes a timestamp once the work
 * already issued on that subqueue's iterators has finished (so regions on
 * one subqueue do not overlap), end once the work issued since has
 * finished. Neither stalls the stream. Returns a handle (0 = none: the
 * slot memory could not be allocated).
 */
#if PAN_ARCH >= 10
uint32_t mali_measure_begin(struct mali_cmd_buffer *cmd, enum mali_measure_kind kind,
                            enum mali_subqueue sq, uint32_t index);
#endif
void mali_measure_end(struct mali_cmd_buffer *cmd, uint32_t handle);
void mali_measure_info(struct mali_cmd_buffer *cmd, uint32_t handle, uint32_t a,
                       uint32_t b, uint32_t c, uint32_t d);
void mali_measure_shader(struct mali_cmd_buffer *cmd, uint32_t handle, uint64_t key);
void mali_measure_extra(struct mali_cmd_buffer *cmd, uint32_t handle, uint32_t extra);

/* ---------------------------------------------------------------------- */
/* Submit (timing.c, recorder.c)                                           */

/*
 * From mali_queue_submit once the submit number is known and before its
 * words go into the rings: reads earlier submits that have completed and
 * queues this submit's command buffers for reading. Takes the
 * frontend-neutral mali_queue (not the CSF-typed mali_csf_queue) so this
 * declaration serves any frontend's submit path alike; today's body
 * (timing.c) is CSF-only and reaches the CSF queue through queue->csf.
 */
void mali_measure_submit(struct mali_device *dev, struct mali_queue *queue,
                         struct vk_queue_submit *submit, uint64_t seqno);

struct mali_measure_capture;

/* A capture of this submit, or NULL (off, outside the window, or the
 * limit reached). */
struct mali_measure_capture *
mali_measure_capture_begin(struct mali_device *dev, struct mali_queue *queue,
                           struct vk_queue_submit *submit, uint64_t seqno);
/* Words the submit writes into subqueue sq's ring. */
void mali_measure_capture_ring(struct mali_measure_capture *cap, unsigned sq,
                               const uint64_t *words, uint32_t count);
/* Write the capture file (before the rings are published, so the memory
 * is as recorded) and free cap. */
void mali_measure_capture_end(struct mali_device *dev, struct mali_measure_capture *cap);

/* ---------------------------------------------------------------------- */
/* Output (config.c), with m->lock held                                    */

/* The submit window (start, count, trigger file) for submit seqno. */
bool mali_measure_window(struct mali_measure *m, uint64_t seqno);
/* A timing row for a region of submit seqno. */
void mali_measure_row(struct mali_measure *m, const struct mali_measure_cmd *mc,
                      const struct mali_measure_region *r);
/* Close the timing file when seqno starts a new chunk. */
void mali_measure_rotate(struct mali_measure *m, uint64_t seqno);
/* Account for a capture file just written (ring rotation included). */
void mali_measure_add_capture(struct mali_measure *m, const char *name, uint64_t seqno,
                              uint64_t bytes);
void mali_measure_write_manifest(struct mali_measure *m);
/* Path of name inside the output directory. */
void mali_measure_path(const struct mali_measure *m, const char *name, char *out,
                       size_t size);

/* Read the completed pending command buffers (all of them with force). */
void mali_measure_collect(struct mali_measure *m, bool force);

#endif
