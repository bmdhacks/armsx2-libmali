# SPDX-License-Identifier: MIT
# Derived from Mesa src/compiler/meson.build.

_vc = mesa_src / 'compiler'
inc_compiler = include_directories('.', '../../../mesa/src/compiler')

builtin_types_h = custom_target(
  'builtin_types.h',
  input : files(_vc / 'builtin_types_h.py'),
  output : 'builtin_types.h',
  command : [prog_python, '@INPUT@', '@OUTPUT@'],
  depend_files : files(_vc / 'builtin_types.py'),
  depends : u_format_gen_h,
)

builtin_types_c = custom_target(
  'builtin_types.c',
  input : files(_vc / 'builtin_types_c.py'),
  output : 'builtin_types.c',
  command : [prog_python, '@INPUT@', '@OUTPUT@'],
  depend_files : files(_vc / 'builtin_types.py'),
)

# Mesa generates this into src/compiler/ (not glsl/).
ir_expression_operation_h = custom_target(
  'ir_expression_operation.h',
  input : files(_vc / 'glsl' / 'ir_expression_operation.py'),
  output : 'ir_expression_operation.h',
  command : [prog_python, '@INPUT@', 'enum'],
  capture : true,
)

_libcompiler = static_library(
  'compiler',
  [files(_vc / 'glsl_types.c', _vc / 'shader_enums.c'),
   ir_expression_operation_h, builtin_types_h, builtin_types_c],
  include_directories : [inc_compiler, inc_include, inc_src],
  dependencies : idep_mesautil,
  c_args : mesa_c_args,
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

idep_compiler = declare_dependency(
  sources : [ir_expression_operation_h, builtin_types_h],
  include_directories : inc_compiler,
  link_with : _libcompiler,
)

subdir('nir')
subdir('spirv')
subdir('rust')
