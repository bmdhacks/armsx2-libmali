# SPDX-License-Identifier: MIT
# Derived from Mesa src/panfrost/lib/meson.build. The pixel-format tables
# (pan_format.c) are built once per entry of pan_archs; idep_pan_format
# below still names the pan_arch one, unchanged from a single-arch build.

libpanfrost_pixel_format_by_arch = {}
idep_pan_format_by_arch = {}
foreach arch : pan_archs
  _key = '@0@'.format(arch)
  libpanfrost_pixel_format_by_arch += {_key : static_library(
    'pan-format-v@0@'.format(arch),
    [files(mesa_src / 'panfrost' / 'lib' / 'pan_format.c'), pan_packers_by_arch[_key]],
    include_directories : [inc_src, inc_include, inc_panfrost],
    c_args : [mesa_c_args, '-DPAN_ARCH=@0@'.format(arch)],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_pan_packers_by_arch[_key], idep_mesautil],
    build_by_default : false,
  )}
  idep_pan_format_by_arch += {_key : declare_dependency(
    link_with : libpanfrost_pixel_format_by_arch[_key],
    include_directories : [inc_panfrost],
    dependencies : [idep_pan_packers_by_arch[_key]],
  )}
endforeach

# The format library for pan_arch, the arch the driver's vk/ sources build
# for today: unchanged name and content from before pan_archs existed.
libpanfrost_pixel_format = libpanfrost_pixel_format_by_arch['@0@'.format(pan_arch)]
idep_pan_format = idep_pan_format_by_arch['@0@'.format(pan_arch)]
