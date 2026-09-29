# SPDX-License-Identifier: MIT
# Derived from Mesa src/vulkan/meson.build: the shared Vulkan runtime.
# Only util/ and runtime/ are built. The only window-system platform is
# Android's: an Android build defines VK_USE_PLATFORM_ANDROID_KHR
# (../../meson.build); vk_android.c is not built, the driver has its own.

_vv = mesa_src / 'vulkan'
vk_api_xml = files(_vv / 'registry' / 'vk.xml')

# Mesa builds without beta extensions by default.
with_vulkan_beta = false

idep_vulkan_wsi_defines = declare_dependency(compile_args : [])
vulkan_wsi_deps = [idep_vulkan_wsi_defines]

subdir('util')
subdir('runtime')
