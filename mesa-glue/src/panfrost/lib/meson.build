# SPDX-License-Identifier: MIT
# Derived from Mesa src/panfrost/lib/meson.build. Only the pixel-format
# tables (pan_format.c) are built, for our architecture: the driver's format
# queries use them.

libpanfrost_pixel_format = static_library(
  'pan-format-v@0@'.format(pan_arch),
  [files(mesa_src / 'panfrost' / 'lib' / 'pan_format.c'), pan_packers],
  include_directories : [inc_src, inc_include, inc_panfrost],
  c_args : [mesa_c_args, '-DPAN_ARCH=@0@'.format(pan_arch)],
  gnu_symbol_visibility : 'hidden',
  dependencies : [idep_pan_packers, idep_mesautil],
  build_by_default : false,
)

idep_pan_format = declare_dependency(
  link_with : libpanfrost_pixel_format,
  include_directories : [inc_panfrost],
  dependencies : [idep_pan_packers],
)
