#include "layers/text_decoder_layer.h"
#include "core/graph_builder.h"
#include "components/attention/self_attention_graph.h"
#include "components/mlp/swiglu_mlp_graph.h"
#include "components/norm/rms_norm_graph.h"
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

    // ══════════════════════════════════════════════════════════
    // 1. Input LayerNorm: hidden_states + input_ln_weight -> normed
    // ══════════════════════════════════════════════════════════
    {
        OperationHandle norm_graph;
        s = components::RmsNormGraph::Build(name + "_InputNorm", epsilon, norm_graph);
        if (s != STATUS_OK) return s;

        s = builder->AddOperation(norm_graph.release(),
                                  {"hidden_states", "input_ln_weight"}, {"normed"});
        if (s != STATUS_OK) return s;
    }

    // ══════════════════════════════════════════════════════════
    // 2. Self-Attention (composed from SelfAttentionGraph)
    // ══════════════════════════════════════════════════════════
    {
        OperationHandle attn_graph;
        s = components::SelfAttentionGraph::Build(
            name + "_Attn", num_heads, num_kv_heads, head_dim, seq_len,
            epsilon, use_mask, attn_graph, use_qk_norm, rotary_dim);
        if (s != STATUS_OK) return s;

        // Wire attention inputs from decoder layer inputs
        atb::SVector<std::string> attn_in;
        attn_in.push_back("normed");           // hidden_states -> normed
        attn_in.push_back("q_weight");
        attn_in.push_back("k_weight");
        attn_in.push_back("v_weight");
        attn_in.push_back("o_weight");
        if (use_qk_norm) {
            attn_in.push_back("q_norm_weight");
            attn_in.push_back("k_norm_weight");
        }
        attn_in.push_back("cos");
        attn_in.push_back("sin");
        if (use_mask) {
            attn_in.push_back("mask");
        }
        attn_in.push_back("seqlen");

        s = builder->AddOperation(attn_graph.release(), attn_in,
                                  atb::SVector<std::string>{"attn_out"});
        if (s != STATUS_OK) return s;
    }

    // ══════════════════════════════════════════════════════════
    // 3. Residual add: hidden_states + attn_out -> h1
    // ══════════════════════════════════════════════════════════
    {
        OperationHandle add_op = ops::ElewiseOp::MakeAdd();
        if (!add_op) return ERROR_GRAPH_BUILD;
        s = builder->AddOperation(add_op.release(),
                                  {"hidden_states", "attn_out"}, {"h1"});
        if (s != STATUS_OK) return s;
    }

    // ══════════════════════════════════════════════════════════
    // 4. Post-Attention LayerNorm: h1 + post_ln_weight -> normed_h1
    // ══════════════════════════════════════════════════════════
    {
        OperationHandle norm_graph;
        s = components::RmsNormGraph::Build(name + "_PostNorm", epsilon, norm_graph);
        if (s != STATUS_OK) return s;

        s = builder->AddOperation(norm_graph.release(),
                                  {"h1", "post_ln_weight"}, {"normed_h1"});
        if (s != STATUS_OK) return s;
    }

    // ══════════════════════════════════════════════════════════
    // 5. SwiGLU MLP (composed from SwiGluMlpGraph)
    // ══════════════════════════════════════════════════════════
    {
        OperationHandle mlp_graph;
        s = components::SwiGluMlpGraph::Build(name + "_MLP", mlp_graph);
        if (s != STATUS_OK) return s;

        s = builder->AddOperation(mlp_graph.release(),
                                  {"normed_h1", "gate_weight", "up_weight", "down_weight"},
                                  atb::SVector<std::string>{"mlp_out"});
        if (s != STATUS_OK) return s;
    }

    // ══════════════════════════════════════════════════════════
    // 6. Residual add: h1 + mlp_out -> output
    // ══════════════════════════════════════════════════════════
    {
        OperationHandle add_op = ops::ElewiseOp::MakeAdd();
        if (!add_op) return ERROR_GRAPH_BUILD;
        s = builder->AddOperation(add_op.release(),
                                  {"h1", "mlp_out"}, {"output"});
        if (s != STATUS_OK) return s;
    }

    out = builder->Build();
    if (!out) return ERROR_GRAPH_BUILD;
    return STATUS_OK;
}

}  // namespace layers
}  // namespace atb_llm
