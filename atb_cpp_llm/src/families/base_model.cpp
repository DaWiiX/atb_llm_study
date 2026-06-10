#include "families/base_model.h"
#include "atb_llm/build_result.h"
#include "utils/float_utils.h"
#include "core/tensor_allocator.h"
#include "log/logger.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdlib>

namespace atb_llm {
namespace families {

namespace {
// P4 experiment: when ATB_DISABLE_PER_OP_SYNC=1, skip the per-op Synchronize
// in ExecuteGraph.  On a single stream, FIFO ordering guarantees that op N+1
// does not launch until op N finishes, so the per-op sync is redundant for
// correctness.  The host still needs to sync before reading device→host copies.
//
// ASCEND_LAUNCH_BLOCKING=1 (CANN env) makes every kernel launch synchronous,
// providing an even stronger guarantee — useful for debugging / validation.
bool PerOpSyncDisabled() {
    const char* env = getenv("ATB_DISABLE_PER_OP_SYNC");
    return env != nullptr;
}
}  // namespace

// ═════════════════════════════════════════════════════════════════════
// ExecuteGraph
// ═════════════════════════════════════════════════════════════════════

Status BaseModel::ExecuteGraph(OperationHandle& graph,
                               atb::VariantPack& vp) {
    if (!graph) return ERROR_GRAPH_BUILD;

    auto* ctx = runtime_->GetContext();

    uint64_t ws_size = 0;
    atb::Status atb_s = graph.get()->Setup(vp, ws_size, ctx);
    if (atb_s != atb::NO_ERROR) {
        LOG_ERROR("Graph Setup failed: %d", static_cast<int>(atb_s));
        return ERROR_GRAPH_BUILD;
    }

    uint8_t* ws_ptr = nullptr;
    if (ws_size > 0) {
        auto __atb_pair_ws = runtime_->GetWorkspace(ws_size); auto& ws = __atb_pair_ws.first; auto& ws_s = __atb_pair_ws.second;
        ws_ptr = ws;
        if (ws_s != STATUS_OK) {
            LOG_ERROR("Failed to get workspace: %zu bytes", static_cast<size_t>(ws_size));
            return ws_s;
        }
        if (ws_ptr == nullptr) {
            LOG_ERROR("Workspace pointer is null despite size=%zu", static_cast<size_t>(ws_size));
            return ERROR_NPU_MEMORY;
        }
    } else {
        // GRAPH_LAUNCH_MODE requires non-null workspace device pointer
        auto __atb_pair_ws = runtime_->GetWorkspace(1); auto& ws = __atb_pair_ws.first; auto& ws_s = __atb_pair_ws.second;
        if (ws_s == STATUS_OK && ws != nullptr) {
            ws_ptr = ws;
            ws_size = 1;
        }
    }

    atb_s = graph.get()->Execute(vp, ws_ptr, ws_size, ctx);
    if (atb_s != atb::NO_ERROR) {
        LOG_ERROR("Graph Execute failed: %d", static_cast<int>(atb_s));
        return ERROR_INFERENCE;
    }

    // P4: Per-op sync is skipped when ATB_DISABLE_PER_OP_SYNC=1.
    // Stream FIFO ordering guarantees ops are serialised on the same stream;
    // the host only *must* sync before reading device→host copies.
    if (!PerOpSyncDisabled()) {
        Status sync_s = runtime_->Synchronize();
        if (sync_s != STATUS_OK) {
            LOG_ERROR("Stream sync failed after Execute: %d", static_cast<int>(sync_s));
            return sync_s;
        }
    }

    return STATUS_OK;
}

// ═════════════════════════════════════════════════════════════════════
// ValidateVariantPack (free function, declared in build_result.h)
// ═════════════════════════════════════════════════════════════════════

Status ValidateVariantPack(const atb::VariantPack& vp,
                           size_t expected_in_count,
                           size_t expected_out_count,
                           const char* context) {
#ifdef DEBUG
    if (vp.inTensors.size() != expected_in_count) {
        LOG_ERROR("VariantPack inTensors size mismatch in %s: got %zu, expected %zu",
                  context ? context : "(unknown)",
                  vp.inTensors.size(), expected_in_count);
        return ERROR_INVALID_PARAM;
    }
    if (vp.outTensors.size() != expected_out_count) {
        LOG_ERROR("VariantPack outTensors size mismatch in %s: got %zu, expected %zu",
                  context ? context : "(unknown)",
                  vp.outTensors.size(), expected_out_count);
        return ERROR_INVALID_PARAM;
    }
#else
    (void)vp;
    (void)expected_in_count;
    (void)expected_out_count;
    (void)context;
#endif
    return STATUS_OK;
}

// ═════════════════════════════════════════════════════════════════════
// ExecuteGraphChecked (Debug only)
// ═════════════════════════════════════════════════════════════════════

#ifdef DEBUG
Status BaseModel::ExecuteGraphChecked(OperationHandle& graph, atb::VariantPack& vp,
                                       size_t expected_in_count, size_t expected_out_count,
                                       const char* context) {
    Status v = ValidateVariantPack(vp, expected_in_count, expected_out_count, context);
    if (v != STATUS_OK) return v;
    return ExecuteGraph(graph, vp);
}
#endif

// ═════════════════════════════════════════════════════════════════════
// EmbeddingLookup
// ═════════════════════════════════════════════════════════════════════

void BaseModel::EmbeddingLookup(const int64_t* input_ids, int64_t seq_len,
                                const uint16_t* embed_table,
                                int64_t hidden_size, int64_t vocab_size,
                                uint16_t* output) {
    for (int64_t s = 0; s < seq_len; s++) {
        int64_t token_id = input_ids[s];
        if (token_id < 0 || token_id >= vocab_size) {
            LOG_ERROR("EmbeddingLookup: token_id %ld out of range [0, %ld)",
                      static_cast<long>(token_id), static_cast<long>(vocab_size));
            // Fill with zeros for invalid token
            std::memset(output + s * hidden_size, 0, hidden_size * sizeof(uint16_t));
            continue;
        }
        std::memcpy(output + s * hidden_size,
                    embed_table + token_id * hidden_size,
                    hidden_size * sizeof(uint16_t));
    }
}

// ═════════════════════════════════════════════════════════════════════
// FindImageTokenPositions
// ═════════════════════════════════════════════════════════════════════

std::vector<int64_t> BaseModel::FindImageTokenPositions(
        const int64_t* input_ids, int64_t seq_len,
        int64_t image_token_id) {
    std::vector<int64_t> positions;
    for (int64_t s = 0; s < seq_len; s++) {
        if (input_ids[s] == image_token_id) {
            positions.push_back(s);
        }
    }
    return positions;
}

// ═════════════════════════════════════════════════════════════════════
// RunPooling
// ═════════════════════════════════════════════════════════════════════

Status BaseModel::RunPooling(const uint16_t* hidden_states, int64_t seq_len,
                             int64_t hidden_size, bool normalize,
                             PoolingStrategy strategy,
                             InferResult& result,
                             const int64_t* attention_mask) {
    const uint16_t* pool_token = nullptr;

    switch (strategy) {
    case PoolingStrategy::LAST_TOKEN: {
        int64_t last_pos = seq_len - 1;
        pool_token = hidden_states + last_pos * hidden_size;
        break;
    }
    case PoolingStrategy::LAST_TOKEN_BY_MASK: {
        // Find the last non-padded token from the attention_mask.
        // Matches Qwen3VLEmbedder._pooling_last: flip mask, find first
        // non-zero, convert to seq position.
        if (!attention_mask) {
            LOG_ERROR("RunPooling: LAST_TOKEN_BY_MASK requires attention_mask");
            return ERROR_INVALID_PARAM;
        }
        int64_t pool_pos = seq_len - 1;  // default: last token
        for (int64_t i = seq_len - 1; i >= 0; i--) {
            if (attention_mask[i] != 0) {
                pool_pos = i;
                break;
            }
        }
        pool_token = hidden_states + pool_pos * hidden_size;
        break;
    }
    case PoolingStrategy::MEAN:
        LOG_ERROR("RunPooling: MEAN strategy not yet implemented");
        return ERROR_UNSUPPORTED;
    case PoolingStrategy::CLS:
        LOG_ERROR("RunPooling: CLS strategy not yet implemented");
        return ERROR_UNSUPPORTED;
    default:
        LOG_ERROR("RunPooling: unknown strategy %d", static_cast<int>(strategy));
        return ERROR_UNSUPPORTED;
    }

    result.shape = {hidden_size};
    result.dtype = ACL_FLOAT16;
    result.data.resize(hidden_size * sizeof(uint16_t));

    if (normalize) {
        float norm = 0.0f;
        for (int64_t i = 0; i < hidden_size; i++) {
            float v = atb_llm::Fp16ToF32(pool_token[i]);
            norm += v * v;
        }
        norm = std::sqrt(norm + 1e-12f);
        uint16_t* out = reinterpret_cast<uint16_t*>(result.data.data());
        for (int64_t i = 0; i < hidden_size; i++)
            out[i] = atb_llm::Fp32ToFp16(atb_llm::Fp16ToF32(pool_token[i]) / norm);
    } else {
        std::memcpy(result.data.data(), pool_token, hidden_size * sizeof(uint16_t));
    }

    return STATUS_OK;
}

}  // namespace families
}  // namespace atb_llm
