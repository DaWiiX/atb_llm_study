#include "components/common/self_attention_graph.h"
#include "core/graph_builder.h"
#include "ops/linear_op.h"
#include "ops/rms_norm_op.h"
#include "ops/rope_op.h"
#include "ops/self_attention_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace components {

// ── AttnConfig-based Build: dispatch by type ─────────────────
Status SelfAttentionGraph::Build(const std::string& name,
                                  const AttnConfig& config,
                                  OperationHandle& out) {
    switch (config.type) {
    case AttnType::GQA:
        return Build(name, config.num_heads, config.num_kv_heads,
                     config.head_dim, config.seq_len, config.epsilon,
                     config.use_mask, out, config.use_qk_norm,
                     config.rotary_dim);
    case AttnType::MHA:
        LOG_ERROR("SelfAttentionGraph: MHA attention not yet supported");
        return ERROR_UNSUPPORTED;
    case AttnType::MLA:
        LOG_ERROR("SelfAttentionGraph: MLA attention not yet supported");
        return ERROR_UNSUPPORTED;
    default:
        LOG_ERROR("SelfAttentionGraph: unknown AttnType %d",
                  static_cast<int>(config.type));
        return ERROR_INVALID_PARAM;
    }
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
    (void)seq_len;

    std::unique_ptr<GraphBuilder> builder;
    Status s = GraphBuilder::Create(name, builder);
    if (s != STATUS_OK) return s;

    // Define input/output names matching Python text_attention.py
    atb::SVector<std::string> in_names;
    in_names.push_back("hidden_states");
    in_names.push_back("q_weight");
    in_names.push_back("k_weight");
    in_names.push_back("v_weight");
    in_names.push_back("o_weight");
    if (use_qk_norm) {
        in_names.push_back("q_norm_weight");
        in_names.push_back("k_norm_weight");
    }
    in_names.push_back("cos");
    in_names.push_back("sin");
    if (use_mask) {
        in_names.push_back("mask");
    }
    in_names.push_back("seqlen");

    atb::SVector<std::string> out_names = {"output"};
    s = builder->Init(name, nullptr, in_names, out_names);
    if (s != STATUS_OK) return s;

    // Helper: create op, add to graph, release ownership
    auto add_op = [&](OperationHandle&& op_h,
                      const atb::SVector<std::string>& ins,
                      const atb::SVector<std::string>& outs) -> Status {
        if (!op_h) return ERROR_GRAPH_BUILD;
        atb::Operation* raw = op_h.release();
        return builder->AddOperation(raw, ins, outs);
    };

    int32_t nh = num_heads;
    int32_t kv_nh = num_kv_heads;
    int32_t hd = head_dim;

    // ── Q path: Linear -> [optional: Reshape(3D) -> RMSNorm -> Reshape(2D)] ──
    s = add_op(ops::LinearOp::Create(),
               {"hidden_states", "q_weight"}, {"q_lin_out"});
    if (s != STATUS_OK) return s;

    std::string q_pre_rope = "q_lin_out";
    if (use_qk_norm) {
        builder->Reshape("q_lin_out",
            [nh, hd](const atb::Dims& old_shape, atb::Dims& new_shape) {
                new_shape.dimNum = 3;
                new_shape.dims[0] = old_shape.dims[0];  // B*S
                new_shape.dims[1] = nh;
                new_shape.dims[2] = hd;
            },
            "q_3d");

        s = add_op(ops::RmsNormOp::Create(epsilon),
                   {"q_3d", "q_norm_weight"}, {"q_normed"});
        if (s != STATUS_OK) return s;

        builder->Reshape("q_normed",
            [nh, hd](const atb::Dims& old_shape, atb::Dims& new_shape) {
                new_shape.dimNum = 2;
                new_shape.dims[0] = old_shape.dims[0];  // B*S
                new_shape.dims[1] = nh * hd;
            },
            "q_flat");
        q_pre_rope = "q_flat";
    }

    // ── K path: Linear -> [optional: Reshape(3D) -> RMSNorm -> Reshape(2D)] ──
    s = add_op(ops::LinearOp::Create(),
               {"hidden_states", "k_weight"}, {"k_lin_out"});
    if (s != STATUS_OK) return s;

    std::string k_pre_rope = "k_lin_out";
    if (use_qk_norm) {
        builder->Reshape("k_lin_out",
            [kv_nh, hd](const atb::Dims& old_shape, atb::Dims& new_shape) {
                new_shape.dimNum = 3;
                new_shape.dims[0] = old_shape.dims[0];  // B*S
                new_shape.dims[1] = kv_nh;
                new_shape.dims[2] = hd;
            },
            "k_3d");

        s = add_op(ops::RmsNormOp::Create(epsilon),
                   {"k_3d", "k_norm_weight"}, {"k_normed"});
        if (s != STATUS_OK) return s;

        builder->Reshape("k_normed",
            [kv_nh, hd](const atb::Dims& old_shape, atb::Dims& new_shape) {
                new_shape.dimNum = 2;
                new_shape.dims[0] = old_shape.dims[0];  // B*S
                new_shape.dims[1] = kv_nh * hd;
            },
            "k_flat");
        k_pre_rope = "k_flat";
    }

    // ── V path: Linear -> Reshape(3D) ──
    s = add_op(ops::LinearOp::Create(),
               {"hidden_states", "v_weight"}, {"v_lin_out"});
    if (s != STATUS_OK) return s;

    builder->Reshape("v_lin_out",
        [kv_nh, hd](const atb::Dims& old_shape, atb::Dims& new_shape) {
            new_shape.dimNum = 3;
            new_shape.dims[0] = old_shape.dims[0];  // B*S
            new_shape.dims[1] = kv_nh;
            new_shape.dims[2] = hd;
        },
        "v_3d");

    // ── RoPE: (q_pre_rope, k_pre_rope, cos, sin, seqlen) -> (q_rope, k_rope) ──
    s = add_op(ops::RopeOp::Create(rotary_dim),
               {q_pre_rope, k_pre_rope, "cos", "sin", "seqlen"},
               {"q_rope_flat", "k_rope_flat"});
    if (s != STATUS_OK) return s;

    // Reshape rope outputs to 3D for SelfAttention (BSND layout)
    builder->Reshape("q_rope_flat",
        [nh, hd](const atb::Dims& old_shape, atb::Dims& new_shape) {
            new_shape.dimNum = 3;
            new_shape.dims[0] = old_shape.dims[0];  // B*S
            new_shape.dims[1] = nh;
            new_shape.dims[2] = hd;
        },
        "q_rope");

    builder->Reshape("k_rope_flat",
        [kv_nh, hd](const atb::Dims& old_shape, atb::Dims& new_shape) {
            new_shape.dimNum = 3;
            new_shape.dims[0] = old_shape.dims[0];  // B*S
            new_shape.dims[1] = kv_nh;
            new_shape.dims[2] = hd;
        },
        "k_rope");

    // ── SelfAttention: (q_rope, k_rope, v_3d, [mask,] seqlen) -> attn_out ──
    OperationHandle sa = ops::SelfAttentionOp::Create(num_heads, num_kv_heads, head_dim, use_mask);
    if (!sa) return ERROR_GRAPH_BUILD;

    atb::SVector<std::string> sa_in;
    sa_in.push_back("q_rope");
    sa_in.push_back("k_rope");
    sa_in.push_back("v_3d");
    if (use_mask) {
        sa_in.push_back("mask");
    }
    sa_in.push_back("seqlen");

    s = builder->AddOperation(sa.release(), sa_in,
        atb::SVector<std::string>{"sa_out"});
    if (s != STATUS_OK) return s;

    // Reshape SA output: [B*S, nh, hd] -> [B*S, nh*hd]
    builder->Reshape("sa_out",
        [](const atb::Dims& old_shape, atb::Dims& new_shape) {
            new_shape.dimNum = 2;
            new_shape.dims[0] = old_shape.dims[0];             // B*S
            new_shape.dims[1] = old_shape.dims[1] * old_shape.dims[2];  // nh*hd
        },
        "sa_flat");

    // ── O projection: Linear ──
    s = add_op(ops::LinearOp::Create(),
               {"sa_flat", "o_weight"}, {"output"});
    if (s != STATUS_OK) return s;

    out = builder->Build();
    if (!out) return ERROR_GRAPH_BUILD;
    return STATUS_OK;
}

} // namespace components
} // namespace atb_llm
