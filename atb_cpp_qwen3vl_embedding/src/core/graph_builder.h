#pragma once
#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "atb/graph_op_builder.h"
#include "core/raii.h"
#include <string>

namespace atb_llm {

/// RAII wrapper around atb::GraphOpBuilder with convenient builder API.
/// Supports Init -> AddOp/Reshape -> Build workflow.
///
/// Ownership model:
///   - GraphBuilder owns the underlying GraphOpBuilder via GraphOpBuilderPtr.
///   - Construct via static Create() factory, or move in a GraphOpBuilderPtr.
///   - Build() returns an OperationHandle (RAII) -- caller owns the result.
class GraphBuilder {
public:
    /// Create a GraphBuilder by allocating a new GraphOpBuilder.
    /// Returns STATUS_OK on success; out is nullptr on failure.
    static Status Create(const std::string& name, std::unique_ptr<GraphBuilder>& out);

    /// Take ownership of an existing GraphOpBuilder.
    explicit GraphBuilder(const std::string& name, GraphOpBuilderPtr&& builder);

    ~GraphBuilder() = default;

    // Non-copyable, movable
    GraphBuilder(const GraphBuilder&) = delete;
    GraphBuilder& operator=(const GraphBuilder&) = delete;
    GraphBuilder(GraphBuilder&&) = default;
    GraphBuilder& operator=(GraphBuilder&&) = default;

    /// Initialize the graph operator.
    /// Corresponds to atb::GraphOpBuilder::Init.
    Status Init(const std::string& op_name,
                const atb::InferShapeFunc& infer_func,
                const atb::SVector<std::string>& in_names,
                const atb::SVector<std::string>& out_names);

    /// Add an ATB operation by creating it from OpParam and adding to graph.
    /// Corresponds to atb::GraphOpBuilder::AddOperation(OpParam, ...).
    template <typename OpParam>
    Status AddOp(const OpParam& param,
                 const atb::SVector<std::string>& in_names,
                 const atb::SVector<std::string>& out_names);

    /// Add an already-created operation to the graph.
    Status AddOperation(atb::Operation* op,
                        const atb::SVector<std::string>& in_names,
                        const atb::SVector<std::string>& out_names);

    /// Add a Reshape (change intermediate tensor shape).
    Status Reshape(const std::string& src_name,
                   const atb::ReshapeFunc& func,
                   const std::string& view_name);

    /// Build the graph and return an OperationHandle (RAII).
    /// The caller owns the returned handle.
    /// Subsequent calls to Build() on the same GraphBuilder are undefined behavior --
    /// the internal builder is consumed after Build().
    OperationHandle Build();

    /// Get the raw builder (for advanced use). Returns nullptr if builder was consumed.
    atb::GraphOpBuilder* GetBuilder() const { return builder_.get(); }

private:
    GraphOpBuilderPtr builder_;
    std::string name_;

    // Private default constructor for move semantics
    GraphBuilder() = default;
};

// Template implementation
template <typename OpParam>
Status GraphBuilder::AddOp(const OpParam& param,
                           const atb::SVector<std::string>& in_names,
                           const atb::SVector<std::string>& out_names) {
    if (!builder_) return ERROR_GRAPH_BUILD;
    atb::Status s = builder_->AddOperation(param, in_names, out_names);
    return (s == atb::NO_ERROR) ? STATUS_OK : ERROR_GRAPH_BUILD;
}

} // namespace atb_llm
