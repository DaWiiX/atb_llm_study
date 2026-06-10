#include "components/vision/patch_embed_graph.h"
#include "core/graph_builder.h"
#include "ops/linear_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace components {

Status PatchEmbedGraph::Build(const std::string& name,
                               int32_t in_channels,
                               int32_t temporal_patch_size,
                               int32_t patch_size,
                               int32_t embed_dim,
                               OperationHandle& out) {
    (void)embed_dim;  // graph derives output dim from the weight tensor shape at Setup time
    std::unique_ptr<GraphBuilder> builder;
    Status s = GraphBuilder::Create(name, builder);
    if (s != STATUS_OK) return s;

    atb::SVector<std::string> in_names = {"pixels", "w", "b"};
    atb::SVector<std::string> out_names = {"output"};
    s = builder->Init(name, nullptr, in_names, out_names);
    if (s != STATUS_OK) return s;

    auto add_op = [&](OperationHandle&& op_h,
                      const atb::SVector<std::string>& ins,
                      const atb::SVector<std::string>& outs) -> Status {
        if (!op_h) return ERROR_GRAPH_BUILD;
        return builder->AddOperation(op_h.release(), ins, outs);
    };

    int32_t kernel_size = in_channels * temporal_patch_size * patch_size * patch_size;

    // ── Reshape: (N*C*tp*p*p,) -> (N, C*tp*p*p) ──
    builder->Reshape("pixels",
        [kernel_size](const atb::Dims& old_shape, atb::Dims& new_shape) {
            new_shape.dimNum = 2;
            new_shape.dims[0] = old_shape.dims[0] / kernel_size;  // N
            new_shape.dims[1] = kernel_size;                       // C*tp*p*p
        }, "flat");

    // ── Linear: (N, kernel_size) -> (N, embed_dim) ──
    s = add_op(ops::LinearOp::Create(true),
               {"flat", "w", "b"}, {"output"});
    if (s != STATUS_OK) return s;

    out = builder->Build();
    if (!out) return ERROR_GRAPH_BUILD;
    return STATUS_OK;
}

} // namespace components
} // namespace atb_llm
