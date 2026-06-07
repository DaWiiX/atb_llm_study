#include "components/vision/vision_merger_graph.h"
#include "core/graph_builder.h"
#include "ops/layer_norm_op.h"
#include "ops/linear_op.h"
#include "ops/activation_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace components {

Status VisionMergerGraph::Build(const std::string& name,
                                 int32_t hidden_size,
                                 int32_t merge_size,
                                 bool is_deepstack,
                                 float epsilon,
                                 OperationHandle& out) {
    std::unique_ptr<GraphBuilder> builder;
    Status s = GraphBuilder::Create(name, builder);
    if (s != STATUS_OK) return s;

    atb::SVector<std::string> in_names = {
        "x", "n_w", "n_b", "f1_w", "f1_b", "f2_w", "f2_b"
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

    int32_t merge = merge_size;
    int32_t mer_hs = hidden_size * merge * merge;

    if (is_deepstack) {
        // Deepstack: reshape first, then norm
        // x: (N, hidden_size) -> (N / merge^2, hidden_size * merge^2)
        builder->Reshape("x",
            [mer_hs](const atb::Dims& old_shape, atb::Dims& new_shape) {
                new_shape.dimNum = 2;
                new_shape.dims[0] = old_shape.dims[0] * old_shape.dims[1] / mer_hs;
                new_shape.dims[1] = mer_hs;
            }, "ds_flat");

        s = add_op(ops::LayerNormOp::Create(epsilon),
                   {"ds_flat", "n_w", "n_b"}, {"normed"});
        if (s != STATUS_OK) return s;

        // fc1 -> GELU -> fc2
        s = add_op(ops::LinearOp::Create(true),
                   {"normed", "f1_w", "f1_b"}, {"fc1_out"});
        if (s != STATUS_OK) return s;
    } else {
        // Main merger: norm first, then reshape
        s = add_op(ops::LayerNormOp::Create(epsilon),
                   {"x", "n_w", "n_b"}, {"normed"});
        if (s != STATUS_OK) return s;

        builder->Reshape("normed",
            [mer_hs](const atb::Dims& old_shape, atb::Dims& new_shape) {
                new_shape.dimNum = 2;
                new_shape.dims[0] = old_shape.dims[0] * old_shape.dims[1] / mer_hs;
                new_shape.dims[1] = mer_hs;
            }, "mf");

        // fc1 -> GELU -> fc2
        s = add_op(ops::LinearOp::Create(true),
                   {"mf", "f1_w", "f1_b"}, {"fc1_out"});
        if (s != STATUS_OK) return s;
    }

    // GELU activation
    s = add_op(ops::ActivationOp::MakeGELU(),
               {"fc1_out"}, {"act_out"});
    if (s != STATUS_OK) return s;

    // fc2 Linear
    s = add_op(ops::LinearOp::Create(true),
               {"act_out", "f2_w", "f2_b"}, {"output"});
    if (s != STATUS_OK) return s;

    out = builder->Build();
    if (!out) return ERROR_GRAPH_BUILD;
    return STATUS_OK;
}

} // namespace components
} // namespace atb_llm
