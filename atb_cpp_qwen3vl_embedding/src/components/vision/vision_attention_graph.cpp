#include "components/vision/vision_attention_graph.h"
#include "core/graph_builder.h"
#include "ops/linear_op.h"
#include "ops/split_op.h"
#include "ops/rope_op.h"
#include "ops/self_attention_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace components {

Status VisionAttentionGraph::Build(const std::string& name,
                                    int32_t num_heads,
                                    int32_t head_dim,
                                    OperationHandle& out) {
    std::unique_ptr<GraphBuilder> builder;
    Status s = GraphBuilder::Create(name, builder);
    if (s != STATUS_OK) return s;

    // Inputs match Python vision_attention.py
    atb::SVector<std::string> in_names = {
        "hidden", "qkv_w", "qkv_b", "proj_w", "proj_b", "c", "s", "seq"
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

    int32_t nh = num_heads;
    int32_t hd = head_dim;

    // ── QKV Linear: hidden + qkv_w + qkv_b -> qkv_out (N, 3*nh*hd) ──
    s = add_op(ops::LinearOp::Create(true),
               {"hidden", "qkv_w", "qkv_b"}, {"qkv_out"});
    if (s != STATUS_OK) return s;

    // ── Reshape to 4D: (N, 3*nh*hd) -> (N, 3, nh, hd) ──
    builder->Reshape("qkv_out",
        [nh, hd](const atb::Dims& old_shape, atb::Dims& new_shape) {
            new_shape.dimNum = 4;
            new_shape.dims[0] = old_shape.dims[0];  // N
            new_shape.dims[1] = 3;
            new_shape.dims[2] = nh;
            new_shape.dims[3] = hd;
        }, "qkv4");

    // ── Split into Q, K, V along dim=1: each (N, 1, nh, hd) ──
    OperationHandle split_op = ops::SplitOp::Create(1, 3);
    if (!split_op) return ERROR_GRAPH_BUILD;
    s = builder->AddOperation(split_op.release(),
        atb::SVector<std::string>{"qkv4"},
        atb::SVector<std::string>{"q4d", "k4d", "v4d"});
    if (s != STATUS_OK) return s;

    // ── Q path: Flatten (N, 1, nh, hd) -> (N, nh*hd) ──
    builder->Reshape("q4d",
        [nh, hd](const atb::Dims& old_shape, atb::Dims& new_shape) {
            new_shape.dimNum = 2;
            new_shape.dims[0] = old_shape.dims[0];  // N
            new_shape.dims[1] = nh * hd;
        }, "q_flat");

    // ── K path: Flatten (N, 1, nh, hd) -> (N, nh*hd) ──
    builder->Reshape("k4d",
        [nh, hd](const atb::Dims& old_shape, atb::Dims& new_shape) {
            new_shape.dimNum = 2;
            new_shape.dims[0] = old_shape.dims[0];  // N
            new_shape.dims[1] = nh * hd;
        }, "k_flat");

    // ── RoPE: (q_flat, k_flat, cos, sin, seq) -> (q_rope_flat, k_rope_flat) ──
    s = add_op(ops::RopeOp::Create(),
               {"q_flat", "k_flat", "c", "s", "seq"},
               {"q_rope_flat", "k_rope_flat"});
    if (s != STATUS_OK) return s;

    // Reshape RoPE outputs to 3D for SelfAttention (BSND layout)
    builder->Reshape("q_rope_flat",
        [nh, hd](const atb::Dims& old_shape, atb::Dims& new_shape) {
            new_shape.dimNum = 3;
            new_shape.dims[0] = old_shape.dims[0];
            new_shape.dims[1] = nh;
            new_shape.dims[2] = hd;
        }, "q_rope");

    builder->Reshape("k_rope_flat",
        [nh, hd](const atb::Dims& old_shape, atb::Dims& new_shape) {
            new_shape.dimNum = 3;
            new_shape.dims[0] = old_shape.dims[0];
            new_shape.dims[1] = nh;
            new_shape.dims[2] = hd;
        }, "k_rope");

    // ── V path: Reshape (N, 1, nh, hd) -> (N, nh, hd), no RoPE ──
    builder->Reshape("v4d",
        [nh, hd](const atb::Dims& old_shape, atb::Dims& new_shape) {
            new_shape.dimNum = 3;
            new_shape.dims[0] = old_shape.dims[0];
            new_shape.dims[1] = nh;
            new_shape.dims[2] = hd;
        }, "v3d");

    // ── SelfAttention: no mask for vision ──
    OperationHandle sa = ops::SelfAttentionOp::Create(num_heads, num_heads, head_dim, false);
    if (!sa) return ERROR_GRAPH_BUILD;

    s = builder->AddOperation(sa.release(),
        atb::SVector<std::string>{"q_rope", "k_rope", "v3d", "seq"},
        atb::SVector<std::string>{"sa_out"});
    if (s != STATUS_OK) return s;

    // ── Flatten SA output: (N, nh, hd) -> (N, nh*hd) ──
    builder->Reshape("sa_out",
        [](const atb::Dims& old_shape, atb::Dims& new_shape) {
            new_shape.dimNum = 2;
            new_shape.dims[0] = old_shape.dims[0];
            new_shape.dims[1] = old_shape.dims[1] * old_shape.dims[2];
        }, "sa_flat");

    // ── Output projection: proj Linear with bias ──
    s = add_op(ops::LinearOp::Create(true),
               {"sa_flat", "proj_w", "proj_b"}, {"output"});
    if (s != STATUS_OK) return s;

    out = builder->Build();
    if (!out) return ERROR_GRAPH_BUILD;
    return STATUS_OK;
}

} // namespace components
} // namespace atb_llm
