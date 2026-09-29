# SPDX-License-Identifier: MIT
# Derived from Mesa src/panfrost/meson.build.

inc_panfrost = include_directories(
  '.', 'compiler', 'model',
  '../../../mesa/src/panfrost',
  '../../../mesa/src/panfrost/compiler',
  '../../../mesa/src/panfrost/lib',
  '../../../mesa/src/panfrost/model',
)

panfrost_rust_args = [
  rust_global_args,
  rust_2024_lint_args,
  '-Anon_snake_case',
]

subdir('model')
subdir('genxml')
subdir('lib')
subdir('compiler')
