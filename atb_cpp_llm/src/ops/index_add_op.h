#pragma once
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "core/raii.h"
#include <cstdint>

namespace atb_llm {
namespace ops {

/// ATB IndexAdd operator wrapper.
///
/// Semantics (mirrors PyTorch torch.Tensor.index_add_):
///   output[indices[i], :] = var[indices[i], :] + updates[i, :]   (for i in range(N))
///   output[j, :]         = var[j, :]                              (for j not in indices)
///
/// Inputs:
///   var      : (M, ...) — base tensor (e.g. fp16)
///   indices  : (N,)     — int32 row indices into var
///   updates  : (N, ...) — values to add at the indexed rows
///
/// Output:
///   output   : (M, ...) — same shape/dtype as var
///
/// @param axis       Which dimension of `var` to index along. axis=0 means
///                   row-wise scatter (most common). Negative is allowed.
class IndexAddOp {
public:
    static OperationHandle Create(int64_t axis = 0);
};

} // namespace ops
} // namespace atb_llm
