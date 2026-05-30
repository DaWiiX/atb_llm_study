#include "layers/text_decoder_layer.h"
#include "core/graph_builder.h"
#include "ops/linear_op.h"
#include "ops/rms_norm_op.h"
#include "ops/rope_op.h"
#include "ops/self_attention_op.h"
#include "ops/activation_op.h"
#include "ops/elewise_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace layers {

Status TextDecoderLayerGraph::Build(const std::string& name,
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

    // Define inputs matching Python text_decoder_layer.py
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
    in_names.push_back("gate_weight");
    in_names.push_back("up_weight");
    in_names.push_back("down_weight");
    in_names.push_back("input_ln_weight");
    in_names.push_back("post_ln_weight");
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

    // ══════════════════════════════════════════════════════════
    // 1. Input LayerNorm: hidden_states + input_ln_weight -> normed
    // ══════════════════════════════════════════════════════════
    s = add_op(ops::RmsNormOp::Create(epsilon),
               {"hidden_states", "input_ln_weight"}, {"normed"});
    if (s != STATUS_OK) return s;

    // ══════════════════════════════════════════════════════════
    // 2. Attention (inline, matching text_attention.py)
    // ══════════════════════════════════════════════════════════

    // Q: Linear -> [optional: Reshape(3D) -> RMSNorm -> Reshape(2D)]
    s = add_op(ops::LinearOp::Create(),
               {"normed", "q_weight"}, {"q_lin_out"});
    if (s != STATUS_OK) return s;

    std::string q_pre_rope = "q_lin_out";
    if (use_qk_norm) {
        builder->Reshape("q_lin_out", [nh, hd](const atb::Dims& o, atb::Dims& n) {
                n.dimNum = 3; n.dims[0] = o.dims[0]; n.dims[1] = nh; n.dims[2] = hd; }, "q_3d");

        s = add_op(ops::RmsNormOp::Create(epsilon),
                   {"q_3d", "q_norm_weight"}, {"q_normed"});
        if (s != STATUS_OK) return s;

        builder->Reshape("q_normed", [nh, hd](const atb::Dims& o, atb::Dims& n) {
                n.dimNum = 2; n.dims[0] = o.dims[0]; n.dims[1] = nh * hd; }, "q_flat");
        q_pre_rope = "q_flat";
    }

    // K: Linear -> [optional: Reshape(3D) -> RMSNorm -> Reshape(2D)]
    s = add_op(ops::LinearOp::Create(),
               {"normed", "k_weight"}, {"k_lin_out"});
    if (s != STATUS_OK) return s;

    std::string k_pre_rope = "k_lin_out";
    if (use_qk_norm) {
        builder->Reshape("k_lin_out", [kv_nh, hd](const atb::Dims& o, atb::Dims& n) {
                n.dimNum = 3; n.dims[0] = o.dims[0]; n.dims[1] = kv_nh; n.dims[2] = hd; }, "k_3d");

        s = add_op(ops::RmsNormOp::Create(epsilon),
                   {"k_3d", "k_norm_weight"}, {"k_normed"});
        if (s != STATUS_OK) return s;

        builder->Reshape("k_normed", [kv_nh, hd](const atb::Dims& o, atb::Dims& n) {
                n.dimNum = 2; n.dims[0] = o.dims[0]; n.dims[1] = kv_nh * hd; }, "k_flat");
        k_pre_rope = "k_flat";
    }

    // V: Linear -> Reshape(3D)
    s = add_op(ops::LinearOp::Create(),
               {"normed", "v_weight"}, {"v_lin_out"});
    if (s != STATUS_OK) return s;

    builder->Reshape("v_lin_out", [kv_nh, hd](const atb::Dims& o, atb::Dims& n) {
            n.dimNum = 3; n.dims[0] = o.dims[0]; n.dims[1] = kv_nh; n.dims[2] = hd; }, "v_3d");

    // RoPE
    s = add_op(ops::RopeOp::Create(rotary_dim),
               {q_pre_rope, k_pre_rope, "cos", "sin", "seqlen"},
               {"q_rope_flat", "k_rope_flat"});
    if (s != STATUS_OK) return s;

    builder->Reshape("q_rope_flat", [nh, hd](const atb::Dims& o, atb::Dims& n) {
            n.dimNum = 3; n.dims[0] = o.dims[0]; n.dims[1] = nh; n.dims[2] = hd; }, "q_rope");
    builder->Reshape("k_rope_flat", [kv_nh, hd](const atb::Dims& o, atb::Dims& n) {
            n.dimNum = 3; n.dims[0] = o.dims[0]; n.dims[1] = kv_nh; n.dims[2] = hd; }, "k_rope");

    // SelfAttention
    OperationHandle sa = ops::SelfAttentionOp::Create(num_heads, num_kv_heads, head_dim, use_mask);
    if (!sa) return ERROR_GRAPH_BUILD;

    atb::SVector<std::string> sa_in;
    sa_in.push_back("q_rope");
    sa_in.push_back("k_rope");
    sa_in.push_back("v_3d");
    if (use_mask) sa_in.push_back("mask");
    sa_in.push_back("seqlen");

    s = builder->AddOperation(sa.release(), sa_in,
                              atb::SVector<std::string>{"sa_out"});
    if (s != STATUS_OK) return s;

    // Reshape SA output: [B*S, nh, hd] -> [B*S, nh*hd]
    builder->Reshape("sa_out", [](const atb::Dims& o, atb::Dims& n) {
            n.dimNum = 2; n.dims[0] = o.dims[0]; n.dims[1] = o.dims[1] * o.dims[2]; }, "sa_flat");

    // O projection
    s = add_op(ops::LinearOp::Create(),
               {"sa_flat", "o_weight"}, {"attn_out"});
    if (s != STATUS_OK) return s;

    // ══════════════════════════════════════════════════════════
    // 3. Residual add: hidden_states + attn_out -> h1
    // ══════════════════════════════════════════════════════════
    s = add_op(ops::ElewiseOp::MakeAdd(),
               {"hidden_states", "attn_out"}, {"h1"});
    if (s != STATUS_OK) return s;

    // ══════════════════════════════════════════════════════════
    // 4. Post-Attention LayerNorm: h1 + post_ln_weight -> normed_h1
    // ══════════════════════════════════════════════════════════
    s = add_op(ops::RmsNormOp::Create(epsilon),
               {"h1", "post_ln_weight"}, {"normed_h1"});
    if (s != STATUS_OK) return s;

    // ══════════════════════════════════════════════════════════
    // 5. SwiGLU MLP (inline, matching text_mlp.py)
    // ══════════════════════════════════════════════════════════

    // gate_proj -> SiLU
    s = add_op(ops::LinearOp::Create(),
               {"normed_h1", "gate_weight"}, {"gate_out"});
    if (s != STATUS_OK) return s;

    s = add_op(ops::ActivationOp::MakeSiLU(),
               {"gate_out"}, {"act_out"});
    if (s != STATUS_OK) return s;

    // up_proj
    s = add_op(ops::LinearOp::Create(),
               {"normed_h1", "up_weight"}, {"up_out"});
    if (s != STATUS_OK) return s;

    // Element-wise multiply
    s = add_op(ops::ElewiseOp::MakeMul(),
               {"act_out", "up_out"}, {"mul_out"});
    if (s != STATUS_OK) return s;

    // down_proj
    s = add_op(ops::LinearOp::Create(),
               {"mul_out", "down_weight"}, {"mlp_out"});
    if (s != STATUS_OK) return s;

    // ══════════════════════════════════════════════════════════
    // 6. Residual add: h1 + mlp_out -> output
    // ══════════════════════════════════════════════════════════
    s = add_op(ops::ElewiseOp::MakeAdd(),
               {"h1", "mlp_out"}, {"output"});
    if (s != STATUS_OK) return s;

    out = builder->Build();
    if (!out) return ERROR_GRAPH_BUILD;
    return STATUS_OK;
}

}  // namespace layers
}  // namespace atb_llm
