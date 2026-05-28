#include "components/vision/deepstack_graph.h"
#include "components/vision/vision_merger_graph.h"

namespace atb_llm {
namespace components {

Status DeepstackGraph::Build(const std::string& name,
                              int32_t hidden_size,
                              int32_t merge_size,
                              float epsilon,
                              OperationHandle& out) {
    // Deepstack is just VisionMergerGraph with is_deepstack=true
    return VisionMergerGraph::Build(name, hidden_size, merge_size,
                                     true, epsilon, out);
}

} // namespace components
} // namespace atb_llm
