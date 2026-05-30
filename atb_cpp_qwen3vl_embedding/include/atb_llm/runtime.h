#pragma once
#include "atb_llm/types.h"
#include "atb/atb_infer.h"
#include <memory>
#include <string>
#include <utility>

namespace atb_llm {

class TensorAllocator;
class WeightLoader;

/// GraphOpBuilder RAII deleter
struct GraphOpBuilderDeleter {
    void operator()(atb::GraphOpBuilder* p) {
        if (p) atb::DestroyGraphOpBuilder(p);
    }
};
using GraphOpBuilderPtr = std::unique_ptr<atb::GraphOpBuilder, GraphOpBuilderDeleter>;

/// 运行时服务 -- Engine 提供给 Model 的资源访问接口
class IRuntime {
public:
    virtual ~IRuntime() = default;

    // ── ATB 资源 ─────────────────────────────────────────
    virtual atb::Context* GetContext() = 0;
    virtual aclrtStream GetStream() = 0;
    virtual Status Synchronize() = 0;

    // ── 内存管理 ─────────────────────────────────────────
    virtual TensorAllocator* GetAllocator() = 0;
    virtual std::pair<uint8_t*, Status> GetWorkspace(uint64_t required_size) = 0;

    // ── 权重加载 ─────────────────────────────────────────
    virtual WeightLoader* GetWeightLoader() = 0;
};

} // namespace atb_llm
