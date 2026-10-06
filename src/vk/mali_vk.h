/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The Vulkan front half: instance, physical device, device and queue, built
 * on Mesa's Vulkan runtime.
 */

#ifndef MALI_VK_H
#define MALI_VK_H

#include <pthread.h>
#include <stdint.h>

#include "util/list.h"
#include "util/simple_mtx.h"

#include "vk_device.h"
#include "vk_instance.h"
#include "vk_log.h"
#include "vk_physical_device.h"
#include "vk_queue.h"

#include "kbase/kbase.h"

#include "mali_bo_pool.h"

#include "mali_entrypoints.h"

/* Highest number of physical devices (kbase nodes /dev/mali0..N-1) we look
 * at; the blob creates at most four. */
#define MALI_MAX_PHYSICAL_DEVICES 4

/* The Vulkan version we implement: ARMSX2 needs 1.1 and uses nothing
 * newer. */
#define MALI_API_VERSION VK_MAKE_API_VERSION(0, 1, 1, VK_HEADER_VERSION)

/* One queue family, one queue. */
#define MALI_QUEUE_COUNT 1

struct mali_instance {
   struct vk_instance vk;
   /* How the kbase layer reaches the kernel; the OS backend except in
    * tests (mali_instance_set_kbase_backend). */
   const struct mali_kbase_backend *kbase_backend;
};

VK_DEFINE_HANDLE_CASTS(mali_instance, vk.base, VkInstance, VK_OBJECT_TYPE_INSTANCE)

/*
 * Tests only: kbase backend for instances created after this call (NULL
 * restores the OS backend). Not exported from the driver .so.
 */
void mali_instance_set_kbase_backend(const struct mali_kbase_backend *backend);

struct mali_physical_device {
   struct vk_physical_device vk;

   char path[32];                      /* "/dev/mali0" */
   struct mali_kbase_gpu_props props;  /* decoded GET_GPUPROPS */
   uint32_t cs_work_registers;

   /* props.arch_major, cached here so code outside the kbase layer picks
    * the per-arch device entry points (mali_CreateDevice) and internals
    * (mali_arch_dispatch) without reaching back into props. */
   uint32_t arch;

   VkPhysicalDeviceMemoryProperties memory;
   uint64_t timestamp_hz;              /* 0: no timestamps */
};

VK_DEFINE_HANDLE_CASTS(mali_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)

/* vk_instance::physical_devices.enumerate / .destroy */
VkResult mali_physical_devices_enumerate(struct vk_instance *instance);
void mali_physical_device_destroy(struct vk_physical_device *pdev);

/* The deviceName for a GPU, or false if it is not a GPU this driver was
 * built for. */
bool mali_device_name(const struct mali_kbase_gpu_props *p, char *buf, size_t size);

/* Log sink for the kbase layer: kbase messages go to the Vulkan log. */
void mali_kbase_log_to_vk(void *user, const char *msg);

struct mali_csf_queue;
struct mali_csf_device;
struct mali_measure;
struct mali_shader;

/* Internal compute/fragment shaders for copies, fills and blits
 * (mali_cmd_copy.c, mali_cmd_meta_gfx.c). Frontend-neutral: every GPU
 * architecture needs the same internal shaders, compiled on first use and
 * cached for the device's lifetime. */
enum mali_meta_shader {
   MALI_META_COPY_16,     /* 16 bytes per invocation */
   MALI_META_COPY_4,
   MALI_META_COPY_1,
   MALI_META_FILL_16,
   MALI_META_FILL_4,
   MALI_META_COUNT,
};

struct mali_queue {
   struct vk_queue vk;
   /* Queue group, rings, submit state (mali_queue.h). */
   struct mali_csf_queue *csf;
   /* Same object as csf above, untyped: lets frontend-neutral code (none
    * yet) hold a queue pointer without the CSF type. Set alongside csf;
    * the CSF (and, later, JM) queue-init path owns both. */
   void *fe;
};

VK_DEFINE_HANDLE_CASTS(mali_queue, vk.base, VkQueue, VK_OBJECT_TYPE_QUEUE)

struct mali_device {
   struct vk_device vk;

   /* This device's kbase context (a VkDevice owns one, as in the blob). */
   struct mali_kbase *kbase;

   struct mali_queue queues[MALI_QUEUE_COUNT];
   uint32_t queue_count;

   /* Shader memory: code in the executable zone, SPDs and other pipeline
    * descriptors. */
   struct mali_bo_pool exec_pool;
   struct mali_bo_pool desc_pool;

   /*
    * Device state every GPU architecture needs, regardless of frontend
    * (CSF or JM): command-buffer memory, internal shaders, the sync
    * waiter lock and device-loss state, and submit statistics. This used
    * to live in struct mali_csf_device; it moved here because none of it
    * is CSF-specific (the JM backend will need the same bookkeeping).
    * Initialized by mali_device.c before the frontend's own device-init
    * runs, so it exists under either frontend.
    */

   /* Command-buffer memory: free 64 KiB slabs, shared by every command
    * pool of the device. */
   simple_mtx_t slab_lock;
   struct list_head free_slabs;
   unsigned free_slab_count;

   /* Internal shaders, compiled on first use. */
   simple_mtx_t meta_lock;
   struct mali_shader *meta[MALI_META_COUNT];
   /* Internal fragment shaders (mali_cmd_meta_gfx.c), by key. */
   void *meta_gfx;
   /* Blend shaders (mali_blend.c), by key. Under meta_lock. */
   void *blend_shaders;

   /* Guards every mali_sync's state and the device's loss state; waiters
    * sleep on cond, which submits, host signals and kernel (or, on JM,
    * on-demand) events broadcast. */
   pthread_mutex_t lock;
   pthread_cond_t cond;
   bool lost;
   uint32_t lost_reported;        /* vk_device_set_lost called (atomic) */

   struct {
      uint64_t kernel_events;       /* EVENT notifications read */
      uint64_t group_errors;
      uint64_t waits;               /* host waits that had to sleep */
      uint64_t wakeups;             /* condition wake-ups while waiting */
      uint64_t submits;
   } stats;

   /* Command-stream state specific to the device's frontend: struct
    * mali_csf_device (CSF, v11) today; struct mali_jm_device (JM, v9)
    * once that frontend exists. Scoreboard facts, the event thread, the
    * queue group and the kcpu queue stay inside it (mali_queue.h). */
   struct mali_csf_device *csf;
   /* Same object as csf above, untyped: lets frontend-neutral code hold a
    * device pointer without the CSF type. Set alongside csf. */
   void *fe;

   /* Timing and command-stream capture (src/measurement/); NULL unless
    * LIBMALI_MEASURE or debug.libmali.measure turns them on. */
   struct mali_measure *measure;

   /* Interned byte strings (mali_device_intern_key): small ids that
    * pipelines compare at bind instead of the data behind them. */
   struct {
      simple_mtx_t lock;
      struct hash_table *table;
      uint32_t count;
   } keys;

   /* Shader compiles and pipeline cache hits since device creation. */
   struct {
      uint32_t shaders_compiled;
      uint32_t shader_cache_hits;
   } pipeline_stats;
};

VK_DEFINE_HANDLE_CASTS(mali_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

static inline struct mali_physical_device *
mali_device_physical(struct mali_device *dev)
{
   return container_of(dev->vk.physical, struct mali_physical_device, vk);
}

#endif
