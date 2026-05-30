#include "models/vision_model.h"
#include "components/vision/vision_block_graph.h"
#include "components/vision/patch_embed_graph.h"
#include "components/vision/vision_merger_graph.h"
#include "components/vision/deepstack_graph.h"
#include "core/graph_builder.h"
#include "ops/elewise_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace models {

VisionModel::VisionModel(const Config& cfg) : cfg_(cfg) {}

Status VisionModel::Build() {
    int32_t nh = cfg_.num_heads;
    int32_t hd = cfg_.hidden_size / nh;

    // 1. Build first layer graph (patch_embed + pos_embed + block 0)
    Status s = BuildVisionFirstLayer(cfg_, first_layer_graph_);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to build VisionFirstLayer graph");
        return s;
    }

    // 2. Build shared vision block graph (looped for blocks 1..depth-1)
    s = components::VisionBlockGraph::Build(
        "VisionBlock", nh, hd, cfg_.epsilon, block_graph_);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to build VisionBlock graph");
        return s;
    }

    // 3. Build main merger graph
    s = components::VisionMergerGraph::Build(
        "VisionMerger", cfg_.hidden_size, cfg_.spatial_merge_size,
        /*is_deepstack=*/false, cfg_.epsilon, merger_graph_);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to build VisionMerger graph");
        return s;
    }

    // 4. Build deepstack merger graph
    s = components::DeepstackGraph::Build(
        "DeepstackMerger", cfg_.hidden_size, cfg_.spatial_merge_size,
        cfg_.epsilon, deepstack_graph_);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to build DeepstackMerger graph");
        return s;
    }

    LOG_INFO("VisionModel built: depth=%d, nh=%d, hd=%d, hs=%d",
             cfg_.depth, nh, hd, cfg_.hidden_size);
    return STATUS_OK;
}

Status BuildVisionFirstLayer(const VisionModel::Config& cfg, OperationHandle& out) {
    int32_t nh = cfg.num_heads;
    int32_t hd = cfg.hidden_size / nh;

    std::unique_ptr<GraphBuilder> builder;
    Status s = GraphBuilder::Create("VisFirstLayer", builder);
    if (s != STATUS_OK) return s;

    // Inputs: pixels, pe_w, pe_b, pos, c, s, seq, + 12 block weights
    atb::SVector<std::string> in_names = {
        "pixels", "pe_w", "pe_b", "pos", "c", "s", "seq",
        "qkv_w", "qkv_b", "proj_w", "proj_b",
        "fc1_w", "fc1_b", "fc2_w", "fc2_b",
        "n1_w", "n1_b", "n2_w", "n2_b"
    };
    atb::SVector<std::string> out_names = {"output"};
    s = builder->Init("VisFirstLayer", nullptr, in_names, out_names);
    if (s != STATUS_OK) return s;

    auto add_op = [&](OperationHandle&& op_h,
                      const atb::SVector<std::string>& ins,
                      const atb::SVector<std::string>& outs) -> Status {
        if (!op_h) return ERROR_GRAPH_BUILD;
        return builder->AddOperation(op_h.release(), ins, outs);
    };

    // ── Patch Embed: pixels -> flat -> Linear -> patched ──
    // Build as a sub-graph
    OperationHandle pe_graph;
    s = components::PatchEmbedGraph::Build(
        "PatchEmbed", cfg.in_channels, cfg.temporal_patch_size,
        cfg.patch_size, cfg.hidden_size, pe_graph);
    if (s != STATUS_OK) return s;

    s = builder->AddOperation(pe_graph.release(),
        atb::SVector<std::string>{"pixels", "pe_w", "pe_b"},
        atb::SVector<std::string>{"patched"});
    if (s != STATUS_OK) return s;

    // ── Add position embedding: patched + pos -> h0 ──
    s = add_op(ops::ElewiseOp::MakeAdd(),
               {"patched", "pos"}, {"h0"});
    if (s != STATUS_OK) return s;

    // ── Vision Block 0 ──
    OperationHandle block_graph;
    s = components::VisionBlockGraph::Build(
        "VisFirstLayer_Block0", nh, hd, cfg.epsilon, block_graph);
    if (s != STATUS_OK) return s;

    s = builder->AddOperation(block_graph.release(),
        atb::SVector<std::string>{
            "h0",
            "qkv_w", "qkv_b", "proj_w", "proj_b",
            "fc1_w", "fc1_b", "fc2_w", "fc2_b",
            "n1_w", "n1_b", "n2_w", "n2_b",
            "c", "s", "seq"
        },
        atb::SVector<std::string>{"output"});
    if (s != STATUS_OK) return s;

    out = builder->Build();
    if (!out) return ERROR_GRAPH_BUILD;
    return STATUS_OK;
}

} // namespace models
} // namespace atb_llm
