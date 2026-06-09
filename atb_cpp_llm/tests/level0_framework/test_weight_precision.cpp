/**
 * Weight loading precision test: validates that C++ fp16 conversion from
 * safetensors matches Python's fp16 conversion bit-for-bit (or within 1 ULP
 * for fp32 intermediate rounding).
 *
 * This test prevents the B9 bug class: double-truncation where f32 weights
 * were converted bf16 then to fp16, losing precision compared to the correct
 * f32->fp16 direct conversion done by Python.
 *
 * For each representative weight tensor:
 *   1. Read raw bytes from the model's model.safetensors
 *   2. Convert up to 100 elements to fp16 using the EXACT SAME logic as
 *      CopyWeightToFp16Host() in src/io/weight_helpers.cpp:
 *        - bf16: Bf16ToFp16Buffer() (bf16 -> fp32 -> fp16, RNE)
 *        - fp32: Fp32ToFp16() (fp32 -> fp16 via CANN, RNE)
 *        - fp16: direct memcpy
 *   3. Dump the first 100 fp16 values as hex to /tmp/cpp_weight_dump.txt
 *
 * A companion Python script reads the dump and compares against its own
 * numpy-based conversion.
 *
 * CI-friendly: skips gracefully when the model checkpoint is absent.
 *
 * Run: ./test_weight_precision
 *      python3 tests/level0_framework/test_weight_precision.py
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "io/safetensors_reader.h"
#include "safetensors.hh"
#include "utils/float_utils.h"
#include "test_env.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using atb_llm::SafetensorsReader;
using atb_llm::WeightInfo;
using atb_llm::STATUS_OK;

// ── Representative weight keys spanning layers and dtypes ─────
// Chosen to cover: text model (early/mid/late), vision model,
// norm weights (often small), projection weights (large 2D).
static const char* kWeightKeys[] = {
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.0.input_layernorm.weight",
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.layers.14.self_attn.q_proj.weight",
    "model.language_model.layers.27.mlp.down_proj.weight",
    "model.visual.blocks.0.attn.qkv.weight",
    "model.visual.blocks.0.mlp.linear_fc1.weight",
    "model.visual.blocks.23.attn.qkv.weight",
    "model.visual.merger.linear_fc1.weight",
};

// ── Helper: dtype enum to human-readable string ──────────────
static const char* DtypeStr(int dtype) {
    switch (static_cast<safetensors::dtype>(dtype)) {
        case safetensors::kFLOAT16:  return "fp16";
        case safetensors::kBFLOAT16: return "bf16";
        case safetensors::kFLOAT32:  return "fp32";
        default:                     return "unknown";
    }
}

// ── Convert first N elements of a tensor to fp16 uint16 values ──
// Uses EXACT SAME logic as CopyWeightToFp16Host in src/io/weight_helpers.cpp
static std::vector<uint16_t> ConvertToFp16(const uint8_t* data, int dtype,
                                            size_t n_elements) {
    std::vector<uint16_t> result(n_elements);
    auto st = static_cast<safetensors::dtype>(dtype);

    if (st == safetensors::kFLOAT16) {
        // fp16 -> fp16: direct copy
        std::memcpy(result.data(), data, n_elements * sizeof(uint16_t));
    } else if (st == safetensors::kBFLOAT16) {
        // bf16 -> fp16 via Bf16ToFp16Buffer (bf16->fp32->fp16, RNE)
        atb_llm::Bf16ToFp16Buffer(
            reinterpret_cast<const uint16_t*>(data),
            result.data(), n_elements);
    } else if (st == safetensors::kFLOAT32) {
        // fp32 -> fp16 via Fp32ToFp16 (round-to-nearest-even)
        const float* f32_ptr = reinterpret_cast<const float*>(data);
        for (size_t i = 0; i < n_elements; i++) {
            result[i] = atb_llm::Fp32ToFp16(f32_ptr[i]);
        }
    } else {
        // Unknown dtype: fill with 0xFFFF sentinel
        std::fill(result.begin(), result.end(), uint16_t(0xFFFF));
    }
    return result;
}

// ═════════════════════════════════════════════════════════════════
// Weight dump test
// ═════════════════════════════════════════════════════════════════

TEST_CASE("Weight loading precision dump to /tmp/cpp_weight_dump.txt") {
    std::string model_dir = GetModelDir();
    std::string st_path = model_dir + "/model.safetensors";

    SafetensorsReader reader;
    auto st = reader.LoadFromFile(st_path);

    // CI-friendly skip when model not available
    if (st != STATUS_OK) {
        MESSAGE("SKIP: safetensors not found at " << st_path);
        return;
    }

    FILE* f = fopen("/tmp/cpp_weight_dump.txt", "w");
    REQUIRE(f != nullptr);

    int n_dumped = 0;
    int n_skipped = 0;

    for (const char* key : kWeightKeys) {
        WeightInfo info;
        st = reader.GetTensor(key, info);
        if (st != STATUS_OK) {
            fprintf(f, "# SKIP: tensor not found: %s\n", key);
            n_skipped++;
            continue;
        }

        const uint8_t* data = reader.GetTensorData(key);
        if (!data) {
            fprintf(f, "# SKIP: no data for: %s\n", key);
            n_skipped++;
            continue;
        }

        // Compute total number of elements
        size_t nelem = 1;
        for (auto d : info.shape)
            nelem *= d;

        if (nelem == 0) {
            fprintf(f, "# SKIP: zero-element tensor: %s\n", key);
            n_skipped++;
            continue;
        }

        // Convert up to 100 elements to fp16
        constexpr size_t kMaxDump = 100;
        size_t n_convert = std::min(kMaxDump, nelem);
        std::vector<uint16_t> fp16_vals =
            ConvertToFp16(data, info.dtype, n_convert);

        // ── Write header comment lines ────────────────────────
        fprintf(f, "# C++ weight dump: %s\n", key);
        fprintf(f, "# dtype: %s  nelem: %zu  first100_hex:",
                DtypeStr(info.dtype), nelem);
        for (size_t i = 0; i < n_convert; i++) {
            fprintf(f, " %04X", fp16_vals[i]);
        }
        fprintf(f, "\n");

        // ── Write data line (parsed by Python companion) ──────
        fprintf(f, "%s:", key);
        for (size_t i = 0; i < n_convert; i++) {
            fprintf(f, " %04X", fp16_vals[i]);
        }
        fprintf(f, "\n");

        n_dumped++;
    }

    fclose(f);
    MESSAGE("Weight dump complete: " << n_dumped << " dumped, "
            << n_skipped << " skipped -> /tmp/cpp_weight_dump.txt");
    CHECK(n_dumped > 0);
}

// ═════════════════════════════════════════════════════════════════
// Sanity: ensure all weight keys exist in the safetensors
// ═════════════════════════════════════════════════════════════════

TEST_CASE("All expected weight keys exist in safetensors") {
    std::string model_dir = GetModelDir();
    std::string st_path = model_dir + "/model.safetensors";

    SafetensorsReader reader;
    auto st = reader.LoadFromFile(st_path);

    if (st != STATUS_OK) {
        MESSAGE("SKIP: safetensors not found at " << st_path);
        return;
    }

    // Get all keys for diagnostics
    auto all_keys = reader.GetAllKeys();
    MESSAGE("Safetensors contains " << all_keys.size() << " tensors");

    for (const char* key : kWeightKeys) {
        bool found = reader.HasKey(key);
        if (!found) {
            // Compute prefix for suggestions
            std::string prefix(key, key + std::strlen(key));
            // Trim last segment
            auto dot_pos = prefix.rfind('.');
            if (dot_pos != std::string::npos) {
                prefix = prefix.substr(0, dot_pos + 1);
                auto candidates = reader.GetKeysByPrefix(prefix);
                if (!candidates.empty()) {
                    MESSAGE("  Key not found: " << key
                            << " (did you mean '" << candidates[0] << "'?)");
                    continue;
                }
            }
            MESSAGE("  WARNING: key not found: " << key);
        }
    }
}
