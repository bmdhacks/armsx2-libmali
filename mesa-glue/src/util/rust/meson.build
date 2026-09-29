# SPDX-License-Identifier: MIT
# Derived from Mesa src/util/rust/meson.build.

_libmesa_util_rs = static_library(
  'mesa_util',
  files(mesa_src / 'util' / 'rust' / 'lib.rs'),
  gnu_symbol_visibility : 'hidden',
  rust_abi : 'rust',
  rust_args : [rust_global_args, rust_2024_lint_args, '-Anon_snake_case'],
)

idep_mesa_util_rs = declare_dependency(link_with : _libmesa_util_rs)
