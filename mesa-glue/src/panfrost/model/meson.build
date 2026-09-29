# SPDX-License-Identifier: MIT
# Derived from Mesa src/panfrost/model/meson.build.

_libpanfrost_model = static_library(
  'panfrost_model_lib',
  files(mesa_src / 'panfrost' / 'model' / 'pan_model.c'),
  include_directories : [inc_include, inc_src],
  c_args : mesa_c_args,
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

libpanfrost_model_dep = declare_dependency(
  link_with : _libpanfrost_model,
  include_directories : [inc_include, inc_src, inc_panfrost],
)
