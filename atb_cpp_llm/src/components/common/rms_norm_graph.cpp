#include "components/common/rms_norm_graph.h"
#include "core/graph_builder.h"
#include "ops/rms_norm_op.h"
#include "ops/layer_norm_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace components {

// ── NormConfig-based Build: dispatch by type ─────────────────
Status RmsNormGraph::Build(const std::string& name,
                            const NormConfig& config,
                            OperationHandle& out) {
    switch (config.type) {
    case NormType::RMSNorm:
        return Build(name, config.epsilon, out);
    case NormType::LayerNorm: {
        // Build a LayerNorm graph using the same builder pattern
        std::unique_ptr<GraphBuilder> builder;
        Status s = GraphBuilder::Create(name, builder);
        if (s != STATUS_OK) return s;

        atb::SVector<std::string> in_names;
        if (config.use_bias) {
            in_names = {"input", "weight", "bias"};
        } else {
            in_names = {"input", "weight"};
        }
        atb::SVector<std::string> out_names = {"output"};
        s = builder->Init(name, nullptr, in_names, out_names);
        if (s != STATUS_OK) return s;

        OperationHandle norm_op = ops::LayerNormOp::Create(config.epsilon);
        if (!norm_op) return ERROR_GRAPH_BUILD;

        atb::SVector<std::string> norm_in;
        norm_in.push_back("input");
        norm_in.push_back("weight");
        if (config.use_bias) {
            norm_in.push_back("bias");
        }
        atb::SVector<std::string> norm_out = {"output"};
        s = builder->AddOperation(norm_op.get(), norm_in, norm_out);
        if (s != STATUS_OK) return s;
        norm_op.release();  // graph takes ownership

        out = builder->Build();
        if (!out) return ERROR_GRAPH_BUILD;
        return STATUS_OK;
    }
    default:
        LOG_ERROR("RmsNormGraph: unknown NormType %d",
                  static_cast<int>(config.type));
        return ERROR_INVALID_PARAM;
    }
}

// ── Legacy Build: RMSNorm implementation ──────────────────────
Status RmsNormGraph::Build(const std::string& name,
                           float epsilon,
                           OperationHandle& out) {
    std::unique_ptr<GraphBuilder> builder;
    Status s = GraphBuilder::Create(name, builder);
    if (s != STATUS_OK) return s;

    atb::SVector<std::string> in_names = {"input", "weight"};
    atb::SVector<std::string> out_names = {"output"};
    s = builder->Init(name, nullptr, in_names, out_names);
    if (s != STATUS_OK) return s;

    // Create RMSNorm op, add to graph, release ownership to graph
    OperationHandle norm_op = ops::RmsNormOp::Create(epsilon);
    if (!norm_op) return ERROR_GRAPH_BUILD;

    atb::SVector<std::string> norm_in = {"input", "weight"};
    atb::SVector<std::string> norm_out = {"output"};
    s = builder->AddOperation(norm_op.get(), norm_in, norm_out);
    if (s != STATUS_OK) return s;
    norm_op.release();  // graph takes ownership

    out = builder->Build();
    if (!out) return ERROR_GRAPH_BUILD;
    return STATUS_OK;
}

} // namespace components
} // namespace atb_llm
