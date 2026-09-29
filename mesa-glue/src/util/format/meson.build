# SPDX-License-Identifier: MIT
# Derived from Mesa src/util/format/meson.build.

_vf = mesa_src / 'util' / 'format'
files_mesa_format = []
foreach f : ['u_format.c', 'u_format_bptc.c', 'u_format_etc.c', 'u_format_fxt1.c',
             'u_format_latc.c', 'u_format_other.c', 'u_format_rgtc.c',
             'u_format_s3tc.c', 'u_format_tests.c', 'u_format_unpack_neon.c',
             'u_format_yuv.c', 'u_format_zs.c']
  files_mesa_format += files(_vf / f)
endforeach

_fmt_in = files(_vf / 'u_format_table.py', _vf / 'u_format.yaml')
_fmt_deps = files(_vf / 'u_format_pack.py', _vf / 'u_format_parse.py')

u_format_gen_h = custom_target(
  'u_format_gen.h',
  input : _fmt_in,
  output : 'u_format_gen.h',
  command : [prog_python, '@INPUT@', '--enums'],
  depend_files : _fmt_deps,
  capture : true,
)

u_format_pack_h = custom_target(
  'u_format_pack.h',
  input : _fmt_in,
  output : 'u_format_pack.h',
  command : [prog_python, '@INPUT@', '--header'],
  depend_files : _fmt_deps,
  capture : true,
)

u_format_table_c = custom_target(
  'u_format_table.c',
  input : _fmt_in,
  output : 'u_format_table.c',
  command : [prog_python, '@INPUT@'],
  depend_files : _fmt_deps,
  capture : true,
)

idep_mesautilformat = declare_dependency(sources : u_format_gen_h)

files_mesa_format += [u_format_gen_h, u_format_pack_h, u_format_table_c]
