#include "components/common/swiglu_mlp_graph.h"
#include "core/graph_builder.h"
#include "ops/linear_op.h"
#include "ops/activation_op.h"
#include "ops/elewise_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace components {

// ── MlpConfig-based Build: dispatch by type ──────────────────
Status SwiGluMlpGraph::Build(const std::string& name,
                              const MlpConfig& config,
                              OperationHandle& out) {
    switch (config.type) {
    case MlpType::SwiGLU:
        return Build(name, out);
    case MlpType::GeGLU:
        LOG_ERROR("SwiGluMlpGraph: GeGLU MLP not yet supported");
        return ERROR_UNSUPPORTED;
    case MlpType::GELU:
        LOG_ERROR("SwiGluMlpGraph: GELU MLP not yet supported");
        return ERROR_UNSUPPORTED;
    case MlpType::MoE:
        LOG_ERROR("SwiGluMlpGraph: MoE MLP not yet supported");
        return ERROR_UNSUPPORTED;
    default:
        LOG_ERROR("SwiGluMlpGraph: unknown MlpType %d",
                  static_cast<int>(config.type));
        return ERROR_INVALID_PARAM;
    }
}

// ── Legacy Build: SwiGLU implementation ───────────────────────
Status SwiGluMlpGraph::Build(const std::string& name, OperationHandle& out) {
    std::unique_ptr<GraphBuilder> builder;
    Status s = GraphBuilder::Create(name, builder);
    if (s != STATUS_OK) return s;

    atb::SVector<std::string> in_names = {
        "hidden_states", "gate_weight", "up_weight", "down_weight"
    };
    atb::SVector<std::string> out_names = {"output"};
    s = builder->Init(name, nullptr, in_names, out_names);
    if (s != STATUS_OK) return s;

    // Helper: create op, add to graph, release ownership
    auto add_op = [&](OperationHandle&& op_h,
                      const atb::SVector<std::string>& ins,
                      const atb::SVector<std::string>& outs) -> Status {
        if (!op_h) return ERROR_GRAPH_BUILD;
        atb::Operation* raw = op_h.release();  // transfer to graph
        Status st = builder->AddOperation(raw, ins, outs);
        return st;
    };

    // gate_proj: hidden_states * gate_weight -> gate_out
    s = add_op(ops::LinearOp::Create(),
               {"hidden_states", "gate_weight"}, {"gate_out"});
    if (s != STATUS_OK) return s;

    // SiLU activation: gate_out -> act_out
    s = add_op(ops::ActivationOp::MakeSiLU(),
               {"gate_out"}, {"act_out"});
    if (s != STATUS_OK) return s;

    // up_proj: hidden_states * up_weight -> up_out
    s = add_op(ops::LinearOp::Create(),
               {"hidden_states", "up_weight"}, {"up_out"});
    if (s != STATUS_OK) return s;

    // Element-wise multiply: act_out * up_out -> mul_out
    s = add_op(ops::ElewiseOp::MakeMul(),
               {"act_out", "up_out"}, {"mul_out"});
    if (s != STATUS_OK) return s;

    // down_proj: mul_out * down_weight -> output
    s = add_op(ops::LinearOp::Create(),
               {"mul_out", "down_weight"}, {"output"});
    if (s != STATUS_OK) return s;

    out = builder->Build();
    if (!out) return ERROR_GRAPH_BUILD;
    return STATUS_OK;
}

} // namespace components
} // namespace atb_llm
