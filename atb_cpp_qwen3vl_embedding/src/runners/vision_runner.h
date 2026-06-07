#pragma once
#include "atb_llm/types.h"
#include "core/raii.h"
#include <cstdint>
#include <string>
#include <vector>

namespace atb_llm {
namespace runners {

/// Vision Runner — manages ATB graph lifecycle for vision encoder.
///
/// Split-graph strategy:
///   1. first_layer: patch_embed + pos_embed + block 0  (built once)
///   2. per_block:   single VisionBlock graph            (built once, looped)
///   3. merger:      LayerNorm -> reshape -> fc1 -> GELU -> fc2
///   4. deepstack:   reshape -> LayerNorm -> fc1 -> GELU -> fc2
///
/// This runner only manages graph building.
/// Execution orchestration is the responsibility of the adapter/family layer.
class VisionRunner {
public:
    /// Vision runner configuration.
    struct Config {
        int32_t hidden_size = 1280;
        int32_t num_heads = 16;
        int32_t intermediate_size = 3456;
        int32_t depth = 24;
        int32_t in_channels = 3;
        int32_t temporal_patch_size = 2;
        int32_t patch_size = 14;
        int32_t spatial_merge_size = 2;
        int32_t num_position_embeddings = 2304;
        int32_t batch_size = 1;         // batch dimension (future use)
        std::vector<int32_t> deepstack_visual_indexes = {5, 11, 17};
        float epsilon = 1e-6f;
    };

    /// Constructor.
    explicit VisionRunner(const Config& cfg);

    /// Build all required ATB graphs.
    /// @return STATUS_OK on success
    Status Build();

    /// Get the first layer graph (patch_embed + pos_embed + block 0).
    OperationHandle& GetFirstLayerGraph() { return first_layer_graph_; }

    /// Get the shared vision block graph (looped for blocks 1..depth-1).
    OperationHandle& GetBlockGraph() { return block_graph_; }

    /// Get the main merger graph.
    OperationHandle& GetMergerGraph() { return merger_graph_; }

    /// Get the deepstack merger graph.
    OperationHandle& GetDeepstackGraph() { return deepstack_graph_; }

    /// Get the config.
    const Config& GetConfig() const { return cfg_; }

private:
    Config cfg_;
    OperationHandle first_layer_graph_;
    OperationHandle block_graph_;
    OperationHandle merger_graph_;
    OperationHandle deepstack_graph_;
};

/// Build the vision first layer graph: patch_embed -> +pos_embed -> VisionBlock(block 0).
/// @param cfg  Vision config
/// @param out  Output: RAII operation handle
/// @return STATUS_OK on success
Status BuildVisionFirstLayer(const VisionRunner::Config& cfg, OperationHandle& out);

} // namespace runners
} // namespace atb_llm
