/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * GET_GPUPROPS decoding. The kernel returns a byte stream of records, each
 * a little-endian u32 token (key << 2 | size code) followed by a value of
 * 1, 2, 4 or 8 bytes. We store each key we know into its field of
 * struct mali_kbase_gpu_props and skip the rest; a record that runs past
 * the end of the stream is an error.
 */

#include <string.h>

#include "kbase_priv.h"

struct prop_field {
   uint16_t offset;
   uint8_t width; /* 0: key not kept */
};

#define F(member)                                                          \
   {                                                                       \
      offsetof(struct mali_kbase_gpu_props, member),                       \
         sizeof(((struct mali_kbase_gpu_props *)0)->member)                \
   }
#define G(n) [KB_GPUPROP_COHERENCY_GROUP_0 + (n)] = F(coherent_group_core_mask[n])
/* JS_FEATURES_0..2 (keys 35..37), job manager only: see js_features in
 * kbase.h. */
#define JSF(n) [KB_GPUPROP_RAW_JS_FEATURES_0 + (n)] = F(js_features[n])

static const struct prop_field prop_fields[KB_GPUPROP_KEY_COUNT] = {
   [KB_GPUPROP_PRODUCT_ID] = F(product_id),
   [KB_GPUPROP_VERSION_STATUS] = F(version_status),
   [KB_GPUPROP_MINOR_REVISION] = F(minor_revision),
   [KB_GPUPROP_MAJOR_REVISION] = F(major_revision),
   [KB_GPUPROP_GPU_FREQ_KHZ_MAX] = F(gpu_freq_khz_max),
   [KB_GPUPROP_LOG2_PROGRAM_COUNTER_SIZE] = F(log2_program_counter_size),
   [KB_GPUPROP_TEXTURE_FEATURES_0] = F(texture_features[0]),
   [KB_GPUPROP_TEXTURE_FEATURES_1] = F(texture_features[1]),
   [KB_GPUPROP_TEXTURE_FEATURES_2] = F(texture_features[2]),
   [KB_GPUPROP_TEXTURE_FEATURES_3] = F(texture_features[3]),
   [KB_GPUPROP_GPU_AVAILABLE_MEMORY_SIZE] = F(gpu_available_memory_size),
   [KB_GPUPROP_NUM_EXEC_ENGINES] = F(num_exec_engines),
   [KB_GPUPROP_L2_LOG2_LINE_SIZE] = F(l2_log2_line_size),
   [KB_GPUPROP_L2_LOG2_CACHE_SIZE] = F(l2_log2_cache_size),
   [KB_GPUPROP_L2_NUM_L2_SLICES] = F(l2_num_l2_slices),
   [KB_GPUPROP_TILER_BIN_SIZE_BYTES] = F(tiler_bin_size_bytes),
   [KB_GPUPROP_TILER_MAX_ACTIVE_LEVELS] = F(tiler_max_active_levels),
   [KB_GPUPROP_MAX_THREADS] = F(max_threads),
   [KB_GPUPROP_MAX_WORKGROUP_SIZE] = F(max_workgroup_size),
   [KB_GPUPROP_MAX_BARRIER_SIZE] = F(max_barrier_size),
   [KB_GPUPROP_MAX_REGISTERS] = F(max_registers),
   [KB_GPUPROP_MAX_TASK_QUEUE] = F(max_task_queue),
   [KB_GPUPROP_MAX_THREAD_GROUP_SPLIT] = F(max_thread_group_split),
   [KB_GPUPROP_IMPL_TECH] = F(impl_tech),
   [KB_GPUPROP_TLS_ALLOC] = F(tls_alloc),
   [KB_GPUPROP_RAW_SHADER_PRESENT] = F(shader_present),
   [KB_GPUPROP_RAW_TILER_PRESENT] = F(tiler_present),
   [KB_GPUPROP_RAW_L2_PRESENT] = F(l2_present),
   [KB_GPUPROP_RAW_STACK_PRESENT] = F(stack_present),
   [KB_GPUPROP_RAW_L2_FEATURES] = F(l2_features),
   [KB_GPUPROP_RAW_CORE_FEATURES] = F(core_features),
   [KB_GPUPROP_RAW_MEM_FEATURES] = F(mem_features),
   [KB_GPUPROP_RAW_MMU_FEATURES] = F(mmu_features),
   [KB_GPUPROP_RAW_AS_PRESENT] = F(as_present),
   [KB_GPUPROP_RAW_JS_PRESENT] = F(js_present),
   JSF(0), JSF(1), JSF(2),
   [KB_GPUPROP_RAW_TILER_FEATURES] = F(tiler_features),
   [KB_GPUPROP_RAW_TEXTURE_FEATURES_0] = F(raw_texture_features[0]),
   [KB_GPUPROP_RAW_TEXTURE_FEATURES_1] = F(raw_texture_features[1]),
   [KB_GPUPROP_RAW_TEXTURE_FEATURES_2] = F(raw_texture_features[2]),
   [KB_GPUPROP_RAW_TEXTURE_FEATURES_3] = F(raw_texture_features[3]),
   /* u64 on CSF, u32 on a job-manager kernel; store() below zero-extends
    * either size into the 64-bit field. */
   [KB_GPUPROP_RAW_GPU_ID] = F(gpu_id),
   [KB_GPUPROP_RAW_THREAD_MAX_THREADS] = F(thread_max_threads),
   [KB_GPUPROP_RAW_THREAD_MAX_WORKGROUP_SIZE] = F(thread_max_workgroup_size),
   [KB_GPUPROP_RAW_THREAD_MAX_BARRIER_SIZE] = F(thread_max_barrier_size),
   [KB_GPUPROP_RAW_THREAD_FEATURES] = F(thread_features),
   [KB_GPUPROP_RAW_COHERENCY_MODE] = F(coherency_mode),
   [KB_GPUPROP_RAW_THREAD_TLS_ALLOC] = F(thread_tls_alloc),
   [KB_GPUPROP_RAW_GPU_FEATURES] = F(gpu_features),
   [KB_GPUPROP_COHERENCY_NUM_GROUPS] = F(num_coherent_groups),
   [KB_GPUPROP_COHERENCY_NUM_CORE_GROUPS] = F(num_core_groups),
   [KB_GPUPROP_COHERENCY_COHERENCY] = F(coherency),
   G(0), G(1), G(2), G(3), G(4), G(5), G(6), G(7),
   G(8), G(9), G(10), G(11), G(12), G(13), G(14), G(15),
};

#undef F
#undef G
#undef JSF

static uint64_t
read_le(const uint8_t *p, unsigned bytes)
{
   uint64_t v = 0;
   for (unsigned i = 0; i < bytes; i++)
      v |= (uint64_t)p[i] << (8 * i);
   return v;
}

static void
store(struct mali_kbase_gpu_props *props, const struct prop_field *f, uint64_t v)
{
   uint8_t *dst = (uint8_t *)props + f->offset;
   if (f->width == 8) {
      memcpy(dst, &v, 8);
   } else {
      uint32_t v32 = (uint32_t)v;
      memcpy(dst, &v32, 4);
   }
}

static unsigned
popcount64(uint64_t v)
{
   unsigned n = 0;
   for (; v; v &= v - 1)
      n++;
   return n;
}

enum mali_kbase_result
mali_kbase_gpu_props_decode(const void *stream, size_t size,
                            struct mali_kbase_gpu_props *props)
{
   const uint8_t *p = stream;
   size_t pos = 0;
   enum mali_kbase_result result = MALI_KBASE_SUCCESS;

   memset(props, 0, sizeof(*props));

   while (pos < size) {
      if (size - pos < 4) {
         result = MALI_KBASE_ERROR_MALFORMED_PROPERTIES;
         break;
      }
      uint32_t token = (uint32_t)read_le(p + pos, 4);
      unsigned key = token >> 2;
      unsigned bytes = 1u << (token & 3);
      pos += 4;
      if (size - pos < bytes) {
         result = MALI_KBASE_ERROR_MALFORMED_PROPERTIES;
         break;
      }
      uint64_t value = read_le(p + pos, bytes);
      pos += bytes;

      if (key < KB_GPUPROP_KEY_COUNT && prop_fields[key].width) {
         store(props, &prop_fields[key], value);
         props->keys_seen[key / 64] |= 1ull << (key % 64);
      }
   }

   /* Derived values. PRODUCT_ID is GPU_ID bits 31:16: arch major, arch
    * minor, arch rev, product major, 4 bits each. */
   props->arch_major = (props->product_id >> 12) & 0xf;
   props->arch_minor = (props->product_id >> 8) & 0xf;
   props->arch_rev = (props->product_id >> 4) & 0xf;
   props->product_major = props->product_id & 0xf;
   props->core_count = popcount64(props->shader_present);
   props->va_bits = props->mmu_features & 0xff;
   /* As the blob does after decoding: each coherent group's core count is
    * the popcount of its mask. */
   for (unsigned i = 0; i < MALI_KBASE_MAX_COHERENT_GROUPS; i++)
      props->coherent_group_num_cores[i] = popcount64(props->coherent_group_core_mask[i]);

   return result;
}
