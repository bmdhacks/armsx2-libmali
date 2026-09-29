/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The loader-facing symbols. vk_icdGetInstanceProcAddr is ours; Mesa's
 * runtime defines vk_icdNegotiateLoaderICDInterfaceVersion (loader
 * interface 7) and vk_icdGetPhysicalDeviceProcAddr (vk_instance.c). All
 * three are exported through mali_icd.sym; nothing else is.
 */

#include "mali_vk.h"

#define MALI_EXPORT __attribute__((visibility("default")))

MALI_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance, const char *pName);

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance, const char *pName)
{
   return mali_GetInstanceProcAddr(instance, pName);
}
