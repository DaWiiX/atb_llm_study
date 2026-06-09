#pragma once
#include "io/weight_loader.h"
#include "core/tensor_allocator.h"
#include "atb/types.h"
#include <cstdint>
#include <string>
#include <vector>

namespace atb_llm {
namespace io {

/// Copy a safetensors weight to NPU as float16.
/// Handles bf16->fp16, fp32->fp16, fp16 direct, and int64 copy.
/// @param loader   Weight loader (from safetensors file)
/// @param key      Weight key name (e.g., "model.language_model.layers.0.self_attn.q_proj.weight")
/// @param alloc    Tensor allocator for NPU memory
/// @param[out] dst Output ATB tensor (allocated and filled)
/// @return STATUS_OK on success
Status CopyWeightToFp16NPU(WeightLoader& loader,
                           const std::string& key,
                           TensorAllocator& alloc,
                           atb::Tensor& dst);

/// Copy a safetensors weight to host memory, converting bf16/fp32 -> fp16.
/// @param loader   Weight loader
/// @param key      Weight key name
/// @param host_dst Destination host buffer
/// @param dst_capacity_bytes  Size of host buffer in bytes
/// @return STATUS_OK on success
Status CopyWeightToFp16Host(WeightLoader& loader,
                            const std::string& key,
                            void* host_dst,
                            size_t dst_capacity_bytes);

/// Entry describing a single weight to load via LoadLinearWeights.
struct WeightLoadEntry {
    const char* suffix;       // e.g., "self_attn.q_proj.weight"
    atb::Tensor* dst;         // destination tensor
};

/// Load a set of linear layer weights to NPU as fp16.
/// Given a prefix and a list of (suffix, dst*) pairs, load each weight.
/// @param loader   Weight loader
/// @param alloc    Tensor allocator
/// @param prefix   Key prefix (e.g., "model.language_model.layers.0.")
/// @param entries  List of (suffix, destination tensor) pairs
/// @return STATUS_OK on success, or first error encountered
Status LoadLinearWeights(WeightLoader& loader,
                          TensorAllocator& alloc,
                          const std::string& prefix,
                          const WeightLoadEntry* entries,
                          size_t num_entries);

}  // namespace io
}  // namespace atb_llm
