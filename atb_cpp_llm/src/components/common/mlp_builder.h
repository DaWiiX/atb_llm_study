#pragma once
#include "atb_llm/types.h"
#include "atb_llm/layer_desc.h"
#include "core/raii.h"
#include <memory>
#include <string>

namespace atb_llm {
namespace components {

/// Interface for building MLP sub-graphs.
/// Each MLP type (SwiGLU, GeGLU, GELU, MoE) gets its own builder
/// implementation. New MLP types can be added by creating a new
/// IMlpBuilder subclass and registering it in CreateMlpBuilder() --
/// without modifying existing builders.
class IMlpBuilder {
public:
    virtual ~IMlpBuilder() = default;

    /// Build the MLP graph.
    /// @param name    Graph name
    /// @param config  MLP configuration
    /// @param out     Output: RAII operation handle
    /// @return STATUS_OK on success, ERROR_UNSUPPORTED if this builder
    ///         doesn't support the config
    virtual Status Build(const std::string& name,
                         const MlpConfig& config,
                         OperationHandle& out) = 0;

    /// Return the name of this builder (for logging).
    virtual const char* Name() const = 0;
};

/// SwiGLU MLP builder.
/// Implements the gate_proj -> SiLU, up_proj, elementwise multiply,
/// down_proj pipeline used by Qwen3, LLaMA, etc.
class SwiGluBuilder : public IMlpBuilder {
public:
    Status Build(const std::string& name,
                 const MlpConfig& config,
                 OperationHandle& out) override;
    const char* Name() const override { return "SwiGLU"; }
};

/// GeGLU MLP builder -- stub.
/// Returns ERROR_UNSUPPORTED until implemented.
class GeGluBuilder : public IMlpBuilder {
public:
    Status Build(const std::string& name,
                 const MlpConfig& config,
                 OperationHandle& out) override {
        (void)name; (void)config; (void)out;
        return ERROR_UNSUPPORTED;
    }
    const char* Name() const override { return "GeGLU"; }
};

/// GELU MLP builder -- stub.
/// Returns ERROR_UNSUPPORTED until implemented.
class GeluBuilder : public IMlpBuilder {
public:
    Status Build(const std::string& name,
                 const MlpConfig& config,
                 OperationHandle& out) override {
        (void)name; (void)config; (void)out;
        return ERROR_UNSUPPORTED;
    }
    const char* Name() const override { return "GELU"; }
};

/// MoE (Mixture of Experts) MLP builder -- stub.
/// Returns ERROR_UNSUPPORTED until implemented.
class MoeBuilder : public IMlpBuilder {
public:
    Status Build(const std::string& name,
                 const MlpConfig& config,
                 OperationHandle& out) override {
        (void)name; (void)config; (void)out;
        return ERROR_UNSUPPORTED;
    }
    const char* Name() const override { return "MoE"; }
};

/// Factory: create the appropriate builder for the given MLP type.
/// Returns nullptr for unknown types.
std::unique_ptr<IMlpBuilder> CreateMlpBuilder(MlpType type);

}  // namespace components
}  // namespace atb_llm
