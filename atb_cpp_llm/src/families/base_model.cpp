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
// Per-op sync is OFF by default. On the single ATB stream, FIFO ordering
// serialises graph launches, so a Synchronize after every ExecuteGraph is
// redundant for correctness and breaks the graph compiler's async pipelining
// (A/B validated a ~12-13% e2e win on 910B: text 65.5→57.8ms, io/mm 116→100ms,
// stddev 0.71→0.11). The correctness syncs the host actually needs — before
// D2H reads (e.g. text FinalNorm / vision merger in qwen3vl_model.cpp) — stay
// explicit and unconditional; they are not affected by this flag.
//
// Set ATB_ENABLE_PER_OP_SYNC=1 to opt back into per-op Synchronize for
// debugging stream-ordering issues. (Replaces the old opt-out
// ATB_DISABLE_PER_OP_SYNC, which is no longer read — disabling is now the
// default, so the opt-out had no remaining use.)
bool PerOpSyncEnabled() {
    const char* env = getenv("ATB_ENABLE_PER_OP_SYNC");
    return env != nullptr;
}
}  // namespace

// ═════════════════════════════════════════════════════════════════════
// ExecuteGraph
// ═════════════════════════════════════════════════════════════════════

Status BaseModel::ExecuteGraph(OperationHandle& graph,
                               atb::VariantPack& vp) {
    if (!graph) return ERROR_GRAPH_BUILD;

    uint64_t ws_size = 0;

    // Default: async (no per-op Synchronize). Stream FIFO ordering serialises
    // ops on the single ATB stream; ATB_ENABLE_PER_OP_SYNC=1 opts back in for
    // debugging. See PerOpSyncEnabled() for the safety rationale.
    bool sync = PerOpSyncEnabled();

    return ExecuteOperation(graph.get(), vp, runtime_, ws_size, sync);
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
