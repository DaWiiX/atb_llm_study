#pragma once
#include "atb_llm/types.h"
#include "atb_llm/layer_desc.h"
#include "core/raii.h"
#include <memory>
#include <string>

namespace atb_llm {
namespace components {

/// Interface for building attention sub-graphs.
/// Each attention type (GQA, MHA, MLA) gets its own builder implementation.
/// New attention types can be added by creating a new IAttentionBuilder
/// subclass and registering it in CreateAttentionBuilder() -- without
/// modifying existing builders.
class IAttentionBuilder {
public:
    virtual ~IAttentionBuilder() = default;

    /// Build the attention graph.
    /// @param name    Graph name
    /// @param config  Attention configuration
    /// @param out     Output: RAII operation handle
    /// @return STATUS_OK on success, ERROR_UNSUPPORTED if this builder
    ///         doesn't support the config
    virtual Status Build(const std::string& name,
                         const AttnConfig& config,
                         OperationHandle& out) = 0;

    /// Return the name of this builder (for logging).
    virtual const char* Name() const = 0;
};

/// GQA (Grouped-Query Attention) builder.
/// Implements the standard Q/K/V projection -> optional QK-norm -> RoPE ->
/// SelfAttention -> O-projection pipeline used by Qwen3, LLaMA, etc.
class GqaAttentionBuilder : public IAttentionBuilder {
public:
    Status Build(const std::string& name,
                 const AttnConfig& config,
                 OperationHandle& out) override;
    const char* Name() const override { return "GQA"; }
};

/// MHA (Multi-Head Attention) builder -- stub.
/// Returns ERROR_UNSUPPORTED until implemented.
class MhaAttentionBuilder : public IAttentionBuilder {
public:
    Status Build(const std::string& name,
                 const AttnConfig& config,
                 OperationHandle& out) override {
        (void)name; (void)config; (void)out;
        return ERROR_UNSUPPORTED;
    }
    const char* Name() const override { return "MHA"; }
};

/// MLA (Multi-head Latent Attention) builder -- stub.
/// Returns ERROR_UNSUPPORTED until implemented.
class MlaAttentionBuilder : public IAttentionBuilder {
public:
    Status Build(const std::string& name,
                 const AttnConfig& config,
                 OperationHandle& out) override {
        (void)name; (void)config; (void)out;
        return ERROR_UNSUPPORTED;
    }
    const char* Name() const override { return "MLA"; }
};

/// Factory: create the appropriate builder for the given attention type.
/// Returns nullptr for unknown types.
std::unique_ptr<IAttentionBuilder> CreateAttentionBuilder(AttnType type);

}  // namespace components
}  // namespace atb_llm
