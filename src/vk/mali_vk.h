/*
 * Copyright 2026 The libmali authors
 * SPDX-License-Identifier: MIT
 */

/*
 * The Vulkan front half: instance, physical device, device and queue, built
 * on Mesa's Vulkan runtime.
 */

#ifndef MALI_VK_H
#define MALI_VK_H

#include <stdint.h>

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

struct mali_queue {
   struct vk_queue vk;
   /* Queue group, rings, submit state (mali_queue.h). */
   struct mali_csf_queue *csf;
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

   /* Command-stream state shared by the queue and the command buffers
    * (mali_queue.h). */
   struct mali_csf_device *csf;

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
