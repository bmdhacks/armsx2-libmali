# SPDX-License-Identifier: MIT
# Derived from Mesa src/vulkan/util/meson.build.

_vvu = mesa_src / 'vulkan' / 'util'

# The generators import vk_extensions.py / vk_entrypoints.py from their own
# directory; list them so edits there rerun the generators.
vk_extensions_depend_files = []
vk_entrypoints_depend_files = [files(_vvu / 'vk_extensions.py')]
vk_extensions_gen_depend_files = [files(_vvu / 'vk_extensions.py')]
vk_dispatch_table_gen_depend_files = files(_vvu / 'vk_entrypoints.py', _vvu / 'vk_extensions.py')
vk_dispatch_trampolines_gen_depend_files = vk_dispatch_table_gen_depend_files
vk_entrypoints_gen_depend_files = vk_dispatch_table_gen_depend_files
vk_cmd_queue_gen_depend_files = vk_dispatch_table_gen_depend_files
vk_physical_device_features_gen_depend_files = files(_vvu / 'vk_extensions.py')
vk_physical_device_properties_gen_depend_files = files(_vvu / 'vk_extensions.py')
vk_physical_device_spirv_caps_gen_depend_files = files(
  _vvu / 'vk_extensions.py', _vvu / 'vk_physical_device_features_gen.py')
vk_synchronization_helpers_gen_depend_files = files(_vvu / 'vk_extensions.py')
vk_enum_to_str_depend_files = files(_vvu / 'vk_extensions.py')
vk_struct_type_cast_depend_files = files(_vvu / 'vk_extensions.py')

vk_entrypoints_gen = files(_vvu / 'vk_entrypoints_gen.py')
vk_extensions_gen = files(_vvu / 'vk_extensions_gen.py')
vk_cmd_queue_gen = files(_vvu / 'vk_cmd_queue_gen.py')
vk_dispatch_trampolines_gen = files(_vvu / 'vk_dispatch_trampolines_gen.py')
vk_physical_device_features_gen = files(_vvu / 'vk_physical_device_features_gen.py')
vk_physical_device_properties_gen = files(_vvu / 'vk_physical_device_properties_gen.py')
vk_physical_device_spirv_caps_gen = files(_vvu / 'vk_physical_device_spirv_caps_gen.py')
vk_synchronization_helpers_gen = files(_vvu / 'vk_synchronization_helpers_gen.py')

inc_vulkan_util = include_directories('.', '../../../../mesa/src/vulkan/util')

files_vulkan_util = files(
  _vvu / 'vk_alloc.c',
  _vvu / 'vk_format.c',
  _vvu / 'vk_util.c',
)

vk_dispatch_table = custom_target(
  'vk_dispatch_table',
  input : [files(_vvu / 'vk_dispatch_table_gen.py'), vk_api_xml],
  output : ['vk_dispatch_table.c', 'vk_dispatch_table.h'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@',
    '--out-c', '@OUTPUT0@', '--out-h', '@OUTPUT1@',
    '--beta', with_vulkan_beta.to_string()
  ],
  depend_files : vk_dispatch_table_gen_depend_files,
)

vk_enum_to_str = custom_target(
  'vk_enum_to_str',
  input : [files(_vvu / 'gen_enum_to_str.py'), vk_api_xml],
  output : ['vk_enum_to_str.c', 'vk_enum_to_str.h', 'vk_enum_defines.h'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@',
    '--out-c', '@OUTPUT0@',
    '--out-h', '@OUTPUT1@',
    '--out-d', '@OUTPUT2@',
    '--beta', with_vulkan_beta.to_string()
  ],
  depend_files : vk_enum_to_str_depend_files,
)

vk_struct_type_cast = custom_target(
  'vk_struct_type_cast',
  input : [files(_vvu / 'vk_struct_type_cast_gen.py'), vk_api_xml],
  output : ['vk_struct_type_cast.h'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@',
    '--out', '@OUTPUT@',
    '--beta', with_vulkan_beta.to_string()
  ],
  depend_files : vk_struct_type_cast_depend_files,
)

vk_extensions = custom_target(
  'vk_extensions',
  input : [vk_extensions_gen, vk_api_xml],
  output : ['vk_extensions.c', 'vk_extensions.h'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@',
    '--out-c', '@OUTPUT0@', '--out-h', '@OUTPUT1@'
  ],
  depend_files : vk_extensions_gen_depend_files,
)

_libvulkan_util = static_library(
  'vulkan_util',
  [files_vulkan_util, vk_dispatch_table, vk_enum_to_str,
   vk_struct_type_cast, vk_extensions],
  include_directories : [inc_include, inc_src, inc_vulkan_util],
  dependencies : [vulkan_wsi_deps, idep_mesautil, idep_nir_headers],
  c_args : mesa_c_args,
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

idep_vulkan_util_headers = declare_dependency(
  sources : [vk_dispatch_table[1], vk_enum_to_str[1], vk_enum_to_str[2],
             vk_struct_type_cast[0], vk_extensions[1]],
  include_directories : inc_vulkan_util,
)

idep_vulkan_util = declare_dependency(
  link_with : _libvulkan_util,
  dependencies : [idep_vulkan_util_headers, idep_mesautil],
)
