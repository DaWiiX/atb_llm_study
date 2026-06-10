#include "atb_llm/model.h"
#include "adapters/qwen3vl_embedding/qwen3vl_model.h"
#include "io/json_config.h"

namespace atb_llm {

static bool IsQwen3VLCompatible(const std::string& model_type, const JsonConfig& cfg) {
    // Compatible with Qwen3-VL series: model_type contains "qwen3" and has vision_config
    if (model_type.find("qwen3") == std::string::npos) return false;
    return cfg.HasKey("vision_config");
}

REGISTER_MODEL_WITH_CHECK(qwen3vl_embedding,
    []() -> std::unique_ptr<IModel> {
        return std::make_unique<adapters::Qwen3VLModel>();
    },
    IsQwen3VLCompatible,
    10  // priority
)

} // namespace atb_llm
