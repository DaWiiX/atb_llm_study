#include "components/common/self_attention_graph.h"
#include "components/common/attention_builder.h"
#include "log/logger.h"

namespace atb_llm {
namespace components {

// ── AttnConfig-based Build: delegate to builder factory ──────
Status SelfAttentionGraph::Build(const std::string& name,
                                  const AttnConfig& config,
                                  OperationHandle& out) {
    auto builder = CreateAttentionBuilder(config.type);
    if (!builder) {
        LOG_ERROR("SelfAttentionGraph: unknown AttnType %d",
                  static_cast<int>(config.type));
        return ERROR_INVALID_PARAM;
    }
    LOG_INFO("Building attention with %s builder", builder->Name());
    return builder->Build(name, config, out);
}

// ── Legacy Build: GQA implementation ─────────────────────────

Status SelfAttentionGraph::Build(const std::string& name,
                                  int32_t num_heads,
                                  int32_t num_kv_heads,
                                  int32_t head_dim,
                                  int32_t seq_len,
                                  float epsilon,
                                  bool use_mask,
                                  OperationHandle& out,
                                  bool use_qk_norm,
                                  int32_t rotary_dim) {
    (void)seq_len;  // seq_len is unused, preserve this to avoid compiler warning
    AttnConfig config;
    config.type = AttnType::GQA;
    config.num_heads = num_heads;
    config.num_kv_heads = num_kv_heads;
    config.head_dim = head_dim;
    config.epsilon = epsilon;
    config.use_mask = use_mask;
    config.use_qk_norm = use_qk_norm;
    config.rotary_dim = rotary_dim;
    return Build(name, config, out);
}

} // namespace components
} // namespace atb_llm
