#include "components/vision/aclnn_bicubic_resize.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_upsample_bicubic_2d.h"
#include "aclnnop/aclnn_upsample_bicubic2d_aa.h"
#include "atb_llm/runtime.h"
#include "log/logger.h"

namespace atb_llm {

// Build a contiguous 4D NCHW aclTensor (ND format + explicit strides, matching
// the official CANN sample pattern). aclnnUpsampleBicubic2d treats ND as NCHW.
static aclTensor* CreateContiguousNchwTensor(const int64_t dims[4],
                                             aclDataType dtype,
                                             void* device_addr) {
    // contiguous strides: [C*H*W, H*W, W, 1]
    int64_t strides[4];
    strides[3] = 1;
    for (int i = 2; i >= 0; --i) {
        strides[i] = dims[i + 1] * strides[i + 1];
    }
    return aclCreateTensor(dims, 4, dtype, strides, 0,
                           ACL_FORMAT_ND, dims, 4, device_addr);
}

Status NpuBicubicResize(IRuntime* runtime,
                        const void* input_npu,
                        int32_t channels, int32_t in_h, int32_t in_w,
                        int32_t out_h, int32_t out_w,
                        void* output_npu) {
    // Platform guard: aclnnUpsampleBicubic2d supports both 910B (Atlas training)
    // and 310P (Atlas inference) per CANN 9.0.0 docs. Keep an explicit env
    // override for safety but do not hard-reject 310P.
    // (No guard — both supported platforms may use this op.)

    if (runtime == nullptr || input_npu == nullptr || output_npu == nullptr) {
        LOG_ERROR("NpuBicubicResize: null parameter (runtime=%p, input=%p, output=%p)",
                  static_cast<const void*>(runtime), input_npu, output_npu);
        return ERROR_INVALID_PARAM;
    }
    if (channels <= 0 || in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0) {
        LOG_ERROR("NpuBicubicResize: invalid dimensions C=%d H_in=%d W_in=%d H_out=%d W_out=%d",
                  static_cast<int>(channels), static_cast<int>(in_h), static_cast<int>(in_w),
                  static_cast<int>(out_h), static_cast<int>(out_w));
        return ERROR_INVALID_PARAM;
    }

    aclTensor* input_tensor = nullptr;
    aclTensor* output_tensor = nullptr;
    aclIntArray* output_size = nullptr;
    aclOpExecutor* executor = nullptr;
    void* workspace = nullptr;
    Status ret = STATUS_OK;

    int64_t in_dims[4]  = {1, static_cast<int64_t>(channels),
                           static_cast<int64_t>(in_h), static_cast<int64_t>(in_w)};
    int64_t out_dims[4] = {1, static_cast<int64_t>(channels),
                           static_cast<int64_t>(out_h), static_cast<int64_t>(out_w)};

    // aclnn reads input only; const_cast is safe.
    input_tensor = CreateContiguousNchwTensor(in_dims, ACL_FLOAT16,
                                              const_cast<void*>(input_npu));
    if (input_tensor == nullptr) {
        LOG_ERROR("NpuBicubicResize: aclCreateTensor input failed");
        ret = ERROR_NPU_MEMORY;
        goto cleanup;
    }

    output_tensor = CreateContiguousNchwTensor(out_dims, ACL_FLOAT16, output_npu);
    if (output_tensor == nullptr) {
        LOG_ERROR("NpuBicubicResize: aclCreateTensor output failed");
        ret = ERROR_NPU_MEMORY;
        goto cleanup;
    }

    {
        int64_t os[2] = {static_cast<int64_t>(out_h), static_cast<int64_t>(out_w)};
        output_size = aclCreateIntArray(os, 2);
        if (output_size == nullptr) {
            LOG_ERROR("NpuBicubicResize: aclCreateIntArray failed");
            ret = ERROR_NPU_MEMORY;
            goto cleanup;
        }
    }

    {
        uint64_t ws_size = 0;
        aclnnStatus acl_s = aclnnUpsampleBicubic2dGetWorkspaceSize(
            input_tensor, output_size, false, 0.0, 0.0,
            output_tensor, &ws_size, &executor);
        if (acl_s != 0) {
            LOG_ERROR("NpuBicubicResize: GetWorkspaceSize failed, aclnnStatus=%d",
                      static_cast<int>(acl_s));
            ret = ERROR_INFERENCE;
            goto cleanup;
        }

        if (ws_size > 0) {
            aclError acl_e = aclrtMalloc(&workspace, ws_size, ACL_MEM_MALLOC_HUGE_FIRST);
            if (acl_e != ACL_SUCCESS) {
                LOG_ERROR("NpuBicubicResize: aclrtMalloc workspace failed, size=%lu, aclError=%d",
                          static_cast<unsigned long>(ws_size), static_cast<int>(acl_e));
                ret = ERROR_NPU_MEMORY;
                goto cleanup;
            }
        }

        acl_s = aclnnUpsampleBicubic2d(workspace, ws_size, executor,
                                        runtime->GetStream());
        if (acl_s != 0) {
            LOG_ERROR("NpuBicubicResize: execute failed, aclnnStatus=%d",
                      static_cast<int>(acl_s));
            ret = ERROR_INFERENCE;
            goto cleanup;
        }

        // Execute is async — sync before releasing any aclnn-owned resource,
        // otherwise the in-flight kernel may touch freed memory.
        aclError acl_e = aclrtSynchronizeStream(runtime->GetStream());
        if (acl_e != ACL_SUCCESS) {
            LOG_ERROR("NpuBicubicResize: aclrtSynchronizeStream failed, aclError=%d",
                      static_cast<int>(acl_e));
            ret = ERROR_INFERENCE;
            goto cleanup;
        }
    }

cleanup:
    // Per CANN sample: destroy tensors + intarray, free workspace. Do NOT call
    // aclDestroyAclOpExecutor — the executor is stream-lifetime managed and
    // manually destroying it after sync triggers a double-free.
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    if (output_size != nullptr) {
        aclDestroyIntArray(output_size);
    }
    if (output_tensor != nullptr) {
        aclDestroyTensor(output_tensor);
    }
    if (input_tensor != nullptr) {
        aclDestroyTensor(input_tensor);
    }

    return ret;
}

Status NpuBicubicResizeAA(IRuntime* runtime,
                          const void* input_npu,
                          int32_t channels, int32_t in_h, int32_t in_w,
                          int32_t out_h, int32_t out_w,
                          void* output_npu) {
    if (runtime == nullptr || input_npu == nullptr || output_npu == nullptr) {
        LOG_ERROR("NpuBicubicResizeAA: null parameter (runtime=%p, input=%p, output=%p)",
                  static_cast<const void*>(runtime), input_npu, output_npu);
        return ERROR_INVALID_PARAM;
    }
    if (channels <= 0 || in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0) {
        LOG_ERROR("NpuBicubicResizeAA: invalid dimensions C=%d H_in=%d W_in=%d H_out=%d W_out=%d",
                  static_cast<int>(channels), static_cast<int>(in_h), static_cast<int>(in_w),
                  static_cast<int>(out_h), static_cast<int>(out_w));
        return ERROR_INVALID_PARAM;
    }

    aclTensor* input_tensor = nullptr;
    aclTensor* output_tensor = nullptr;
    aclIntArray* output_size = nullptr;
    aclOpExecutor* executor = nullptr;
    void* workspace = nullptr;
    Status ret = STATUS_OK;

    int64_t in_dims[4]  = {1, static_cast<int64_t>(channels),
                           static_cast<int64_t>(in_h), static_cast<int64_t>(in_w)};
    int64_t out_dims[4] = {1, static_cast<int64_t>(channels),
                           static_cast<int64_t>(out_h), static_cast<int64_t>(out_w)};

    input_tensor = CreateContiguousNchwTensor(in_dims, ACL_FLOAT16,
                                              const_cast<void*>(input_npu));
    if (input_tensor == nullptr) {
        LOG_ERROR("NpuBicubicResizeAA: aclCreateTensor input failed");
        ret = ERROR_NPU_MEMORY;
        goto cleanup;
    }

    output_tensor = CreateContiguousNchwTensor(out_dims, ACL_FLOAT16, output_npu);
    if (output_tensor == nullptr) {
        LOG_ERROR("NpuBicubicResizeAA: aclCreateTensor output failed");
        ret = ERROR_NPU_MEMORY;
        goto cleanup;
    }

    {
        int64_t os[2] = {static_cast<int64_t>(out_h), static_cast<int64_t>(out_w)};
        output_size = aclCreateIntArray(os, 2);
        if (output_size == nullptr) {
            LOG_ERROR("NpuBicubicResizeAA: aclCreateIntArray failed");
            ret = ERROR_NPU_MEMORY;
            goto cleanup;
        }
    }

    {
        uint64_t ws_size = 0;
        aclnnStatus acl_s = aclnnUpsampleBicubic2dAAGetWorkspaceSize(
            input_tensor, output_size, false, 0.0, 0.0,
            output_tensor, &ws_size, &executor);
        if (acl_s != 0) {
            LOG_ERROR("NpuBicubicResizeAA: GetWorkspaceSize failed, aclnnStatus=%d",
                      static_cast<int>(acl_s));
            ret = ERROR_INFERENCE;
            goto cleanup;
        }

        if (ws_size > 0) {
            aclError acl_e = aclrtMalloc(&workspace, ws_size, ACL_MEM_MALLOC_HUGE_FIRST);
            if (acl_e != ACL_SUCCESS) {
                LOG_ERROR("NpuBicubicResizeAA: aclrtMalloc workspace failed, size=%lu, aclError=%d",
                          static_cast<unsigned long>(ws_size), static_cast<int>(acl_e));
                ret = ERROR_NPU_MEMORY;
                goto cleanup;
            }
        }

        acl_s = aclnnUpsampleBicubic2dAA(workspace, ws_size, executor,
                                          runtime->GetStream());
        if (acl_s != 0) {
            LOG_ERROR("NpuBicubicResizeAA: execute failed, aclnnStatus=%d",
                      static_cast<int>(acl_s));
            ret = ERROR_INFERENCE;
            goto cleanup;
        }

        aclError acl_e = aclrtSynchronizeStream(runtime->GetStream());
        if (acl_e != ACL_SUCCESS) {
            LOG_ERROR("NpuBicubicResizeAA: aclrtSynchronizeStream failed, aclError=%d",
                      static_cast<int>(acl_e));
            ret = ERROR_INFERENCE;
            goto cleanup;
        }
    }

cleanup:
    // Per CANN sample: destroy tensors + intarray, free workspace. Do NOT call
    // aclDestroyAclOpExecutor — the executor is stream-lifetime managed and
    // manually destroying it after sync triggers a double-free.
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    if (output_size != nullptr) {
        aclDestroyIntArray(output_size);
    }
    if (output_tensor != nullptr) {
        aclDestroyTensor(output_tensor);
    }
    if (input_tensor != nullptr) {
        aclDestroyTensor(input_tensor);
    }

    return ret;
}

}  // namespace atb_llm
