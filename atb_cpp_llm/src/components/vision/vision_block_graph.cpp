#include "components/vision/vision_block_graph.h"
#include "components/vision/vision_attention_graph.h"
#include "components/vision/vision_mlp_graph.h"
#include "core/graph_builder.h"
#include "ops/layer_norm_op.h"
#include "ops/elewise_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace components {

Status VisionBlockGraph::Build(const std::string& name,
                                int32_t num_heads,
                                int32_t head_dim,
                                float epsilon,
                                OperationHandle& out) {
    std::unique_ptr<GraphBuilder> builder;
    Status s = GraphBuilder::Create(name, builder);
    if (s != STATUS_OK) return s;

    // Inputs match Python vision_block.py
    atb::SVector<std::string> in_names = {
        "hidden",
        "qkv_w", "qkv_b", "proj_w", "proj_b",
        "fc1_w", "fc1_b", "fc2_w", "fc2_b",
        "n1_w", "n1_b", "n2_w", "n2_b",
        "c", "s", "seq"
    };
    atb::SVector<std::string> out_names = {"output"};
    s = builder->Init(name, nullptr, in_names, out_names);
    if (s != STATUS_OK) return s;

    auto add_op = [&](OperationHandle&& op_h,
                      const atb::SVector<std::string>& ins,
                      const atb::SVector<std::string>& outs) -> Status {
        if (!op_h) return ERROR_GRAPH_BUILD;
        return builder->AddOperation(op_h.release(), ins, outs);
    };

    // ── 1. LayerNorm1: hidden + n1_w + n1_b -> normed1 ──
    s = add_op(ops::LayerNormOp::Create(epsilon),
               {"hidden", "n1_w", "n1_b"}, {"normed1"});
    if (s != STATUS_OK) return s;

    // ── 2. Vision Attention subgraph ──
    // Build the attention as a nested graph
    OperationHandle attn_graph;
    s = VisionAttentionGraph::Build(name + "_Attn", num_heads, head_dim, attn_graph);
    if (s != STATUS_OK) return s;

    // Add attention as a sub-operation
    // Inputs: normed1, qkv_w, qkv_b, proj_w, proj_b, c, s, seq
    s = builder->AddOperation(attn_graph.release(),
        atb::SVector<std::string>{"normed1", "qkv_w", "qkv_b", "proj_w", "proj_b", "c", "s", "seq"},
        atb::SVector<std::string>{"attn_out"});
    if (s != STATUS_OK) return s;

    // ── 3. Residual add: hidden + attn_out -> h1 ──
    s = add_op(ops::ElewiseOp::MakeAdd(),
               {"hidden", "attn_out"}, {"h1"});
    if (s != STATUS_OK) return s;

    // ── 4. LayerNorm2: h1 + n2_w + n2_b -> normed2 ──
    s = add_op(ops::LayerNormOp::Create(epsilon),
               {"h1", "n2_w", "n2_b"}, {"normed2"});
    if (s != STATUS_OK) return s;

    // ── 5. Vision MLP subgraph ──
    OperationHandle mlp_graph;
    s = VisionMlpGraph::Build(name + "_MLP", mlp_graph);
    if (s != STATUS_OK) return s;

    s = builder->AddOperation(mlp_graph.release(),
        atb::SVector<std::string>{"normed2", "fc1_w", "fc1_b", "fc2_w", "fc2_b"},
        atb::SVector<std::string>{"mlp_out"});
    if (s != STATUS_OK) return s;

    // ── 6. Residual add: h1 + mlp_out -> output ──
    s = add_op(ops::ElewiseOp::MakeAdd(),
               {"h1", "mlp_out"}, {"output"});
    if (s != STATUS_OK) return s;

    out = builder->Build();
    if (!out) return ERROR_GRAPH_BUILD;
    return STATUS_OK;
}

} // namespace components
} // namespace atb_llm
