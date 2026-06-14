#include "components/common/swiglu_mlp_graph.h"
#include "components/common/mlp_builder.h"
#include "log/logger.h"

namespace atb_llm {
namespace components {

// ── MlpConfig-based Build: delegate to builder factory ───────
Status SwiGluMlpGraph::Build(const std::string& name,
                              const MlpConfig& config,
                              OperationHandle& out) {
    auto builder = CreateMlpBuilder(config.type);
    if (!builder) {
        LOG_ERROR("SwiGluMlpGraph: unknown MlpType %d",
                  static_cast<int>(config.type));
        return ERROR_INVALID_PARAM;
    }
    LOG_INFO("Building MLP with %s builder", builder->Name());
    return builder->Build(name, config, out);
}

// ── Legacy Build: SwiGLU implementation ───────────────────────
Status SwiGluMlpGraph::Build(const std::string& name, OperationHandle& out) {
    MlpConfig config;
    config.type = MlpType::SwiGLU;
    return Build(name, config, out);
}

} // namespace components
} // namespace atb_llm
