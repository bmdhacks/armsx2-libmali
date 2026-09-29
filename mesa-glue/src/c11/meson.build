# SPDX-License-Identifier: MIT
# Derived from Mesa src/c11/impl/meson.build.

_vc11 = mesa_src / 'c11' / 'impl'
_c11_files = files(_vc11 / 'time.c')
if not with_c11_threads
  _c11_files += files(_vc11 / 'threads_posix.c')
endif

_libmesa_util_c11 = static_library(
  'mesa_util_c11',
  _c11_files,
  include_directories : [inc_include, inc_src],
  c_args : mesa_c_args,
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

idep_mesautilc11 = declare_dependency(
  link_with : _libmesa_util_c11,
  include_directories : inc_src,
  dependencies : [dep_thread, dep_m],
)
