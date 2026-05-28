#include "atb_llm/model.h"
#include "adapters/qwen3vl_embedding/qwen3vl_model.h"

namespace atb_llm {

REGISTER_MODEL(qwen3vl_embedding, []() -> std::unique_ptr<IModel> {
    return std::make_unique<adapters::Qwen3VLModel>();
});

} // namespace atb_llm
