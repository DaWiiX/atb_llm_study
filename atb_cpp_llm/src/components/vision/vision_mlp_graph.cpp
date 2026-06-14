#include "components/vision/vision_mlp_graph.h"
#include "core/graph_builder.h"
#include "ops/linear_op.h"
#include "ops/activation_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace components {

Status VisionMlpGraph::Build(const std::string& name, OperationHandle& out) {
    std::unique_ptr<GraphBuilder> builder;
    Status s = GraphBuilder::Create(name, builder);
    if (s != STATUS_OK) return s;

    atb::SVector<std::string> in_names = {
        "hidden", "fc1_w", "fc1_b", "fc2_w", "fc2_b"
    };
    atb::SVector<std::string> out_names = {"output"};
    s = builder->Init(name, nullptr, in_names, out_names);
    if (s != STATUS_OK) return s;

    // fc1 Linear (with bias)
    s = builder->AddOp(ops::LinearOp::Create(true),
               {"hidden", "fc1_w", "fc1_b"}, {"fc1_out"});
    if (s != STATUS_OK) return s;

    // GELU activation (not SiLU -- Vision uses GELU)
    s = builder->AddOp(ops::ActivationOp::MakeGELU(),
               {"fc1_out"}, {"act_out"});
    if (s != STATUS_OK) return s;

    // fc2 Linear (with bias)
    s = builder->AddOp(ops::LinearOp::Create(true),
               {"act_out", "fc2_w", "fc2_b"}, {"output"});
    if (s != STATUS_OK) return s;

    out = builder->Build();
    if (!out) return ERROR_GRAPH_BUILD;
    return STATUS_OK;
}

} // namespace components
} // namespace atb_llm
