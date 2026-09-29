# SPDX-License-Identifier: MIT
# Derived from Mesa src/panfrost/compiler/bifrost/valhall/meson.build.

_vv = mesa_src / 'panfrost' / 'compiler' / 'bifrost' / 'valhall'

valhall_c = custom_target(
  'valhall_c',
  input : files(_vv / 'valhall.c.py', _vv / 'ISA.xml'),
  output : 'valhall.c',
  command : [prog_python, '@INPUT0@', '--xml', '@INPUT1@'],
  capture : true,
  depend_files : files(_vv / 'valhall.py'),
)

valhall_enums_h = custom_target(
  'valhall_enums.h',
  input : files(_vv / 'valhall_enums.h.py', _vv / 'ISA.xml'),
  output : 'valhall_enums.h',
  command : [prog_python, '@INPUT0@', '--xml', '@INPUT1@'],
  capture : true,
  depend_files : files(_vv / 'valhall.py'),
)

idep_valhall_enums_h = declare_dependency(
  sources : [valhall_enums_h],
  include_directories : include_directories('.', '../../../../../../mesa/src/panfrost/compiler/bifrost/valhall'),
)

valhall_disasm_c = custom_target(
  'valhall_disasm_c',
  input : files(_vv / 'disasm.py', _vv / 'ISA.xml'),
  output : 'valhall_disasm.c',
  command : [prog_python, '@INPUT0@', '--xml', '@INPUT1@'],
  capture : true,
  depend_files : files(_vv / 'valhall.py'),
)

libpanfrost_valhall_disasm = static_library(
  'panfrost_valhall_disasm',
  valhall_disasm_c,
  include_directories : [inc_include, inc_src,
                         include_directories('../../../../../../mesa/src/panfrost/compiler/bifrost/valhall')],
  c_args : mesa_c_args,
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)
