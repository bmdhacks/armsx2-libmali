/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Mesa's decode_common.c dispatches every pandecode entry point on the GPU's
 * architecture, so it references the decoders of every arch. The test-only
 * pandecode build compiles the v9 decoders only; these definitions stand in
 * for the others and abort if anything reaches them.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct pandecode_context;

static void
not_built(const char *what)
{
   fprintf(stderr, "pandecode: %s is not built (v9-only test build)\n", what);
   abort();
}

#define JC_STUB(v)                                                             \
   void pandecode_jc_v##v(struct pandecode_context *ctx, uint64_t jc,          \
                          uint64_t gpu_id);                                    \
   void pandecode_jc_v##v(struct pandecode_context *ctx, uint64_t jc,          \
                          uint64_t gpu_id)                                     \
   {                                                                           \
      not_built("pandecode_jc_v" #v);                                          \
   }                                                                           \
   void pandecode_abort_on_fault_v##v(struct pandecode_context *ctx,           \
                                      uint64_t jc);                            \
   void pandecode_abort_on_fault_v##v(struct pandecode_context *ctx,           \
                                      uint64_t jc)                             \
   {                                                                           \
      not_built("pandecode_abort_on_fault_v" #v);                              \
   }

#define CS_STUB(v)                                                             \
   void pandecode_interpret_cs_v##v(struct pandecode_context *ctx,             \
                                    uint64_t queue, uint32_t size,             \
                                    uint64_t gpu_id, uint32_t *regs);          \
   void pandecode_interpret_cs_v##v(struct pandecode_context *ctx,             \
                                    uint64_t queue, uint32_t size,             \
                                    uint64_t gpu_id, uint32_t *regs)           \
   {                                                                           \
      not_built("pandecode_interpret_cs_v" #v);                                \
   }                                                                           \
   void pandecode_cs_binary_v##v(struct pandecode_context *ctx, uint64_t bin,  \
                                 uint32_t bin_size);                           \
   void pandecode_cs_binary_v##v(struct pandecode_context *ctx, uint64_t bin,  \
                                 uint32_t bin_size)                            \
   {                                                                           \
      not_built("pandecode_cs_binary_v" #v);                                   \
   }                                                                           \
   void pandecode_cs_trace_v##v(struct pandecode_context *ctx, uint64_t trace, \
                                uint32_t trace_size, uint64_t gpu_id);         \
   void pandecode_cs_trace_v##v(struct pandecode_context *ctx, uint64_t trace, \
                                uint32_t trace_size, uint64_t gpu_id)          \
   {                                                                           \
      not_built("pandecode_cs_trace_v" #v);                                    \
   }

JC_STUB(4)
JC_STUB(5)
JC_STUB(6)
JC_STUB(7)
CS_STUB(10)
CS_STUB(11)
CS_STUB(12)
CS_STUB(13)
CS_STUB(14)
