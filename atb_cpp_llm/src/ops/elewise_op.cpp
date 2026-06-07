#include "ops/elewise_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace ops {

namespace {
OperationHandle CreateElewise(atb::infer::ElewiseParam::ElewiseType type) {
    atb::infer::ElewiseParam param;
    param.elewiseType = type;
    atb::Operation* op = nullptr;
    atb::Status s = atb::CreateOperation(param, &op);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("CreateOperation(ElewiseParam) failed: %d", static_cast<int>(s));
        return OperationHandle(nullptr);
    }
    return OperationHandle(op);
}
} // anonymous namespace

OperationHandle ElewiseOp::MakeAdd() {
    return CreateElewise(atb::infer::ElewiseParam::ELEWISE_ADD);
}

OperationHandle ElewiseOp::MakeMul() {
    return CreateElewise(atb::infer::ElewiseParam::ELEWISE_MUL);
}

OperationHandle ElewiseOp::MakeCast(aclDataType out_dtype) {
    atb::infer::ElewiseParam param;
    param.elewiseType = atb::infer::ElewiseParam::ELEWISE_CAST;
    param.outTensorType = out_dtype;
    atb::Operation* op = nullptr;
    atb::Status s = atb::CreateOperation(param, &op);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("CreateOperation(ElewiseCast) failed: %d", static_cast<int>(s));
        return OperationHandle(nullptr);
    }
    return OperationHandle(op);
}

OperationHandle ElewiseOp::MakeMuls(float value) {
    atb::infer::ElewiseParam param;
    param.elewiseType = atb::infer::ElewiseParam::ELEWISE_MULS;
    param.mulsParam.varAttr = value;
    atb::Operation* op = nullptr;
    atb::Status s = atb::CreateOperation(param, &op);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("CreateOperation(ElewiseMuls) failed: %d", static_cast<int>(s));
        return OperationHandle(nullptr);
    }
    return OperationHandle(op);
}

OperationHandle ElewiseOp::MakeSub() {
    return CreateElewise(atb::infer::ElewiseParam::ELEWISE_SUB);
}

OperationHandle ElewiseOp::MakeCos() {
    return CreateElewise(atb::infer::ElewiseParam::ELEWISE_COS);
}

OperationHandle ElewiseOp::MakeSin() {
    return CreateElewise(atb::infer::ElewiseParam::ELEWISE_SIN);
}

} // namespace ops
} // namespace atb_llm
