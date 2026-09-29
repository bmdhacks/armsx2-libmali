# SPDX-License-Identifier: MIT
# Derived from Mesa src/panfrost/compiler/midgard/meson.build. Built only
# because pan_compiler.c references it (Midgard is arch <= 5).

_vm = mesa_src / 'panfrost' / 'compiler' / 'midgard'

_mg_names = [
  'midgard_compile.c', 'mir.c', 'midgard_address.c', 'midgard_print.c',
  'midgard_schedule.c', 'midgard_derivatives.c', 'midgard_emit.c',
  'midgard_helper_invocations.c', 'midgard_ra.c', 'midgard_ra_pipeline.c',
  'midgard_liveness.c', 'midgard_ops.c', 'mir_promote_uniforms.c',
  'mir_squeeze.c', 'midgard_nir_lower_image_bitsize.c',
  'midgard_nir_lower_writeout.c', 'midgard_nir_type_csel.c',
  'midgard_opt_copy_prop.c', 'midgard_opt_dce.c', 'midgard_opt_perspective.c',
  'midgard_opt_prop.c', 'midgard_errata_lod.c', 'nir_fuse_io_16.c',
]
_mg_files = []
foreach n : _mg_names
  _mg_files += files(_vm / n)
endforeach

midgard_nir_algebraic_c = custom_target(
  'midgard_nir_algebraic.c',
  input : files(_vm / 'midgard_nir_algebraic.py'),
  output : 'midgard_nir_algebraic.c',
  command : [prog_python, '@INPUT@', '-p', dir_compiler_nir],
  capture : true,
  depend_files : nir_algebraic_depends,
)

_inc_midgard = include_directories('.', '../../../../../mesa/src/panfrost/compiler/midgard')

_libpanfrost_midgard_disasm = static_library(
  'panfrost_midgard_disasm',
  files(_vm / 'disassemble.c', _vm / 'midgard_ops.c', _vm / 'midgard_print_constant.c'),
  include_directories : [inc_include, inc_src, _inc_midgard],
  c_args : mesa_c_args,
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

libpanfrost_midgard = static_library(
  'panfrost_midgard',
  [_mg_files, midgard_nir_algebraic_c],
  include_directories : [inc_include, inc_src, _inc_midgard, inc_panfrost],
  dependencies : [idep_nir, idep_mesautil],
  link_with : _libpanfrost_midgard_disasm,
  c_args : mesa_c_args,
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)
