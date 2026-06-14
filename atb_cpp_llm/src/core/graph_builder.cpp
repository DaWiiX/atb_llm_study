#include "core/graph_builder.h"
#include "log/logger.h"
#include "utils/cpp11_compat.h"

namespace atb_llm {

Status GraphBuilder::Create(const std::string& name, std::unique_ptr<GraphBuilder>& out) {
    atb::GraphOpBuilder* raw = nullptr;
    atb::Status s = atb::CreateGraphOpBuilder(&raw);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("CreateGraphOpBuilder failed for '%s': %d", name.c_str(), static_cast<int>(s));
        out.reset();
        return ERROR_GRAPH_BUILD;
    }
    out = atb_llm::make_unique<GraphBuilder>(name, GraphOpBuilderPtr(raw));
    return STATUS_OK;
}

GraphBuilder::GraphBuilder(const std::string& name, GraphOpBuilderPtr&& builder)
    : builder_(std::move(builder)), name_(name) {}

Status GraphBuilder::Init(const std::string& op_name,
                           const atb::InferShapeFunc& infer_func,
                           const atb::SVector<std::string>& in_names,
                           const atb::SVector<std::string>& out_names) {
    if (!builder_) return ERROR_GRAPH_BUILD;
    atb::Status s = builder_->Init(op_name, infer_func, in_names, out_names);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("GraphOpBuilder::Init failed for '%s': %d", op_name.c_str(), static_cast<int>(s));
        return ERROR_GRAPH_BUILD;
    }
    return STATUS_OK;
}

Status GraphBuilder::AddOperation(atb::Operation* op,
                                    const atb::SVector<std::string>& in_names,
                                    const atb::SVector<std::string>& out_names) {
    if (!builder_ || !op) return ERROR_GRAPH_BUILD;
    atb::Status s = builder_->AddOperation(op, in_names, out_names);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("GraphOpBuilder::AddOperation failed: %d", static_cast<int>(s));
        return ERROR_GRAPH_BUILD;
    }
    return STATUS_OK;
}

Status GraphBuilder::AddOp(OperationHandle&& op_h,
                           const atb::SVector<std::string>& in_names,
                           const atb::SVector<std::string>& out_names) {
    if (!op_h) return ERROR_GRAPH_BUILD;
    return AddOperation(op_h.release(), in_names, out_names);
}

Status GraphBuilder::Reshape(const std::string& src_name,
                              const atb::ReshapeFunc& func,
                              const std::string& view_name) {
    if (!builder_) return ERROR_GRAPH_BUILD;
    atb::Status s = builder_->Reshape(src_name, func, view_name);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("GraphOpBuilder::Reshape failed: %d", static_cast<int>(s));
        return ERROR_GRAPH_BUILD;
    }
    return STATUS_OK;
}

OperationHandle GraphBuilder::Build() {
    if (!builder_) return OperationHandle(nullptr);
    atb::Operation* op = builder_->Build();
    if (!op) {
        LOG_ERROR("GraphOpBuilder::Build returned null for '%s'", name_.c_str());
    }
    return OperationHandle(op);
}

} // namespace atb_llm
