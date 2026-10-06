/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * VkDevice and VkQueue. A device owns one kbase context (as the blob's
 * device context does). The queue group, its command-stream queues and
 * submission are mali_queue.c's.
 */

#include "mali_vk.h"

#include <time.h>

#include "vk_alloc.h"
#include "vk_util.h"

#include "mali_cmd_buffer.h"
#include "mali_queue.h"
#include "mali_measure.h"
#include "mali_waist.h"
#include "mali_compiler.h"
#include "vk_pipeline_cache.h"

/*
 * Device state every frontend needs alike (design doc §2.3): the
 * command-buffer slab cache, internal shader caches, and the sync-object
 * lock, condition and device-loss state. Set up before the frontend's own
 * device-init (mali_csf_device_init today; a JM equivalent later) so that
 * state exists under either frontend, and torn down after it, since
 * nothing here depends on dev->csf.
 */
static void
device_fe_state_init(struct mali_device *dev)
{
   simple_mtx_init(&dev->slab_lock, mtx_plain);
   list_inithead(&dev->free_slabs);
   simple_mtx_init(&dev->meta_lock, mtx_plain);

   pthread_condattr_t ca;
   pthread_condattr_init(&ca);
   pthread_condattr_setclock(&ca, CLOCK_MONOTONIC);
   pthread_cond_init(&dev->cond, &ca);
   pthread_condattr_destroy(&ca);
   pthread_mutex_init(&dev->lock, NULL);
}

static void
device_fe_state_finish(struct mali_device *dev)
{
   mali_meta_finish(dev);
   mali_cmd_slabs_finish(dev);

   pthread_cond_destroy(&dev->cond);
   pthread_mutex_destroy(&dev->lock);
   simple_mtx_destroy(&dev->meta_lock);
   simple_mtx_destroy(&dev->slab_lock);
}

static VkResult
queue_init(struct mali_device *dev, struct mali_queue *queue,
           const VkDeviceQueueCreateInfo *info, uint32_t index)
{
   VkResult result = vk_queue_init(&queue->vk, &dev->vk, info, index);
   if (result != VK_SUCCESS)
      return result;
   /* The queue group, its rings and the tiler heap (mali_queue.c). */
   result = mali_csf_queue_init(dev, queue);
   if (result != VK_SUCCESS) {
      vk_queue_finish(&queue->vk);
      return result;
   }
   queue->vk.driver_submit = mali_queue_submit;
   return VK_SUCCESS;
}

static void
queue_finish(struct mali_device *dev, struct mali_queue *queue)
{
   mali_csf_queue_finish(dev, queue);
   vk_queue_finish(&queue->vk);
}

static void
device_destroy(struct mali_device *dev, const VkAllocationCallbacks *alloc)
{
   /* Reads the last results while the queue's done slots still exist. */
   mali_measure_finish(dev);
   for (uint32_t i = 0; i < dev->queue_count; i++)
      queue_finish(dev, &dev->queues[i]);
   /* Command-stream state: event thread (CSF-specific; mali_csf_device
    * itself). */
   mali_csf_device_finish(dev);
   /* Frontend-neutral device state: command memory, internal shaders
    * (their code lives in the pools below), the sync lock/condition. */
   device_fe_state_finish(dev);
   /* The cache holds shaders, which live in the pools. */
   if (dev->vk.mem_cache)
      vk_pipeline_cache_destroy(dev->vk.mem_cache, NULL);
   mali_device_keys_finish(dev);
   if (dev->kbase) {
      mali_bo_pool_finish(&dev->desc_pool);
      mali_bo_pool_finish(&dev->exec_pool);
   }
   mali_kbase_destroy(dev->kbase);
   vk_device_finish(&dev->vk);
   vk_free2(&dev->vk.physical->instance->alloc, alloc, dev);
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_CreateDevice(VkPhysicalDevice physicalDevice,
                  const VkDeviceCreateInfo *pCreateInfo,
                  const VkAllocationCallbacks *pAllocator,
                  VkDevice *pDevice)
{
   VK_FROM_HANDLE(mali_physical_device, pdev, physicalDevice);
   struct mali_instance *instance =
      container_of(pdev->vk.instance, struct mali_instance, vk);
   VkResult result;

   struct mali_device *dev =
      vk_zalloc2(&instance->vk.alloc, pAllocator, sizeof(*dev), 8,
                 VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!dev)
      return vk_error(pdev, VK_ERROR_OUT_OF_HOST_MEMORY);

   /*
    * Four layers, first match wins: the device's own arch (mali_v9_* or
    * mali_v11_*, MALI_PER_ARCH in the per-arch and command-stream files),
    * then our arch-independent entry points, then Mesa's common entry
    * points (added by vk_device_init), then the waist stubs. Only the
    * first layer depends on pdev->arch; everything below it is the same
    * table regardless of which GPU this is.
    */
   struct vk_device_dispatch_table dispatch;
   vk_device_dispatch_table_from_entrypoints(
      &dispatch, pdev->arch == 9 ? &mali_v9_device_entrypoints : &mali_v11_device_entrypoints,
      true);
   vk_device_dispatch_table_from_entrypoints(&dispatch, &mali_device_entrypoints, false);

   /* Checks the extension names and the requested features against what
    * the physical device reports (VK_ERROR_EXTENSION_NOT_PRESENT /
    * VK_ERROR_FEATURE_NOT_PRESENT). */
   result = vk_device_init(&dev->vk, &pdev->vk, &dispatch, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free2(&instance->vk.alloc, pAllocator, dev);
      return result;
   }
   vk_device_dispatch_table_from_entrypoints(&dev->vk.dispatch_table,
                                             &mali_waist_device_entrypoints, false);

   /* The queue requests are checked before the kernel is touched. One
    * family, MALI_QUEUE_COUNT queues, no protected queues. */
   for (uint32_t i = 0; i < pCreateInfo->queueCreateInfoCount; i++) {
      const VkDeviceQueueCreateInfo *q = &pCreateInfo->pQueueCreateInfos[i];
      if (q->queueFamilyIndex != 0 ||
          q->queueCount > MALI_QUEUE_COUNT - dev->queue_count ||
          (q->flags & VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT)) {
         result = vk_errorf(pdev, VK_ERROR_INITIALIZATION_FAILED,
                            "unsupported queue request: family %u, %u queues, flags 0x%x",
                            q->queueFamilyIndex, q->queueCount, q->flags);
         vk_device_finish(&dev->vk);
         vk_free2(&instance->vk.alloc, pAllocator, dev);
         return result;
      }
      dev->queue_count += q->queueCount;
   }
   dev->queue_count = 0;

   /*
    * The device's kbase context. The blob creates it with its memory
    * pools, shader program pool and blend-shader cache; those come with
    * our own memory and pipeline handling. No window system is involved.
    */
   const struct mali_kbase_create_info kinfo = {
      .device_path = pdev->path,
      .backend = instance->kbase_backend,
      .log = mali_kbase_log_to_vk,
   };
   enum mali_kbase_result kr = mali_kbase_create(&kinfo, &dev->kbase);
   if (kr != MALI_KBASE_SUCCESS) {
      result = vk_errorf(pdev,
                         kr == MALI_KBASE_ERROR_OUT_OF_HOST_MEMORY ?
                            VK_ERROR_OUT_OF_HOST_MEMORY :
                            VK_ERROR_INITIALIZATION_FAILED,
                         "cannot open %s: %s", pdev->path, mali_kbase_result_str(kr));
      vk_device_finish(&dev->vk);
      vk_free2(&instance->vk.alloc, pAllocator, dev);
      return result;
   }

   /* Shaders: kraid must be the compiler, code goes to the executable
    * zone, and pipelines created without a VkPipelineCache still share
    * identical shaders through the device's own in-memory cache. */
   mali_bo_pool_init_exec(&dev->exec_pool, dev->kbase);
   mali_bo_pool_init_desc(&dev->desc_pool, dev->kbase);
   mali_device_keys_init(dev);
   /* Frontend-neutral device state (design doc §2.3): every path below
    * this point that can fail calls device_destroy, which tears this down
    * again, so it must exist before the first of them. */
   device_fe_state_init(dev);
   if (mali_compiler_init(pdev->props.gpu_id) != MALI_COMPILE_OK) {
      result = vk_errorf(pdev, VK_ERROR_INITIALIZATION_FAILED,
                         "the shader compiler is not configured for kraid");
      device_destroy(dev, pAllocator);
      return result;
   }
   /* Command buffers and the queue's shared state. */
   dev->vk.command_buffer_ops = &mali_cmd_buffer_ops;
   result = mali_csf_device_init(dev);
   if (result != VK_SUCCESS) {
      device_destroy(dev, pAllocator);
      return result;
   }
   /* Timing and capture, when LIBMALI_MEASURE asks for them. */
   mali_measure_init(dev);

   const struct vk_pipeline_cache_create_info mem_cache_info = {
      .weak_ref = true,
      .skip_disk_cache = true,
   };
   dev->vk.mem_cache = vk_pipeline_cache_create(&dev->vk, &mem_cache_info, NULL);
   if (!dev->vk.mem_cache) {
      device_destroy(dev, pAllocator);
      return vk_error(pdev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   for (uint32_t i = 0; i < pCreateInfo->queueCreateInfoCount; i++) {
      const VkDeviceQueueCreateInfo *q = &pCreateInfo->pQueueCreateInfos[i];
      for (uint32_t j = 0; j < q->queueCount; j++) {
         result = queue_init(dev, &dev->queues[dev->queue_count], q, j);
         if (result != VK_SUCCESS) {
            device_destroy(dev, pAllocator);
            return result;
         }
         dev->queue_count++;
      }
   }

   *pDevice = mali_device_to_handle(dev);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
mali_DestroyDevice(VkDevice _device, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(mali_device, dev, _device);

   if (!dev)
      return;
   device_destroy(dev, pAllocator);
}
