#pragma once
#include "atb_llm/types.h"
#include "atb_llm/operation_handle.h"
#include <string>
#include <vector>

namespace atb_llm {

/// Result of a component's Build() method.
///
/// Contains the built ATB graph handle and the expected input/output
/// tensor names. The names can be used for Debug-mode VariantPack
/// validation to catch tensor ordering mismatches early.
///
/// Usage pattern:
///   BuildResult result;
///   result.graph = builder->Build();
///   result.input_names = {"hidden_states", "weight"};
///   result.output_names = {"output"};
///
/// Future work: component Build() methods will return BuildResult
/// instead of taking OperationHandle& out parameters.
struct BuildResult {
    OperationHandle graph;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

/// Validate that a VariantPack matches expected input/output tensor counts.
///
/// In DEBUG builds, checks that vp.inTensors.size() == expected_in_count
/// and vp.outTensors.size() == expected_out_count. On mismatch, logs an
/// error with the provided context string and returns ERROR_INVALID_PARAM.
///
/// In Release builds, this is a no-op that returns STATUS_OK with zero
/// overhead (all parameters are suppressed via (void) casts).
///
/// @param vp               The VariantPack to validate
/// @param expected_in_count  Expected number of input tensors
/// @param expected_out_count Expected number of output tensors
/// @param context          Human-readable context for error messages (e.g. "DecoderLayer")
/// @return STATUS_OK if validation passes or in Release mode,
///         ERROR_INVALID_PARAM on size mismatch in Debug mode
Status ValidateVariantPack(const atb::VariantPack& vp,
                           size_t expected_in_count,
                           size_t expected_out_count,
                           const char* context);

}  // namespace atb_llm
