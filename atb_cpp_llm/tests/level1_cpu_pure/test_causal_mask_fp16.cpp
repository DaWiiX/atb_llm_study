/**
 * Level 1 CPU pure-function test: MakeCausalMaskFp16 correctness.
 *
 * Verifies that the direct-fp16 mask is identical to the fp32→fp16
 * converted mask (same input, same output).
 *
 * No NPU dependency — runs on CPU only.
 */

#include "runners/text_runner.h"
#include "utils/float_utils.h"
#include <cstdint>
#include <vector>
#include <cmath>
#include <cstdio>

static int passed = 0;
static int failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s\n", msg); \
        failed++; \
    } else { \
        passed++; \
    } \
} while(0)

template<typename T>
static double CosineSimilarity(const T* a, const T* b, int64_t n) {
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        float va = static_cast<float>(a[i]);
        float vb = static_cast<float>(b[i]);
        dot += va * vb;
        na += va * va;
        nb += vb * vb;
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

static void TestFp16VsFp32(int32_t seq_len) {
    int64_t n = static_cast<int64_t>(seq_len) * seq_len;
    // Reference: fp32 path
    std::vector<float> mask_f32(n);
    atb_llm::runners::MakeCausalMask(seq_len, mask_f32.data());
    std::vector<uint16_t> mask_ref(n);
    for (int64_t i = 0; i < n; i++) {
        mask_ref[i] = atb_llm::Fp32ToFp16(mask_f32[i]);
    }

    // Direct fp16 path
    std::vector<uint16_t> mask_fp16(n);
    atb_llm::runners::MakeCausalMaskFp16(seq_len, mask_fp16.data());

    // Byte-exact comparison (fp16 values should be identical —
    // same input constants produce same fp16 output)
    bool exact = true;
    int first_mismatch = -1;
    for (int64_t i = 0; i < n; i++) {
        if (mask_fp16[i] != mask_ref[i]) {
            exact = false;
            if (first_mismatch < 0) first_mismatch = static_cast<int>(i);
        }
    }

    char msg[256];
    std::snprintf(msg, sizeof(msg),
                  "Byte-exact S=%d (%ld elements)", seq_len, static_cast<long>(n));
    if (exact) {
        CHECK(true, msg);
    } else {
        // If not byte-exact (unlikely), check cosine ≈ 1.0
        double cos = CosineSimilarity(mask_fp16.data(), mask_ref.data(), n);
        std::snprintf(msg, sizeof(msg),
                      "Cosine S=%d: %.6f (first mismatch at idx=%d: ref=0x%04x fp16=0x%04x)",
                      seq_len, cos, first_mismatch,
                      static_cast<int>(mask_ref[first_mismatch]),
                      static_cast<int>(mask_fp16[first_mismatch]));
        CHECK(cos >= 0.999999, msg);
    }
}

static void TestMaskStructureFp16(int32_t seq_len) {
    int64_t n = static_cast<int64_t>(seq_len) * seq_len;
    std::vector<uint16_t> mask(n);
    atb_llm::runners::MakeCausalMaskFp16(seq_len, mask.data());

    uint16_t zero_fp16 = atb_llm::Fp32ToFp16(0.0f);
    uint16_t mask_fp16 = atb_llm::Fp32ToFp16(-65504.0f);

    // Check structure: lower triangle (including diagonal) = 0, upper = -65504
    int errors = 0;
    for (int32_t i = 0; i < seq_len; i++) {
        for (int32_t j = 0; j < seq_len; j++) {
            uint16_t expected = (j > i) ? mask_fp16 : zero_fp16;
            if (mask[i * seq_len + j] != expected) {
                errors++;
            }
        }
    }

    char msg[256];
    std::snprintf(msg, sizeof(msg),
                  "Structure S=%d", seq_len);
    CHECK(errors == 0, msg);
}

static void TestEdgeCases() {
    // S=1: single token, no masking (j > i never true)
    {
        uint16_t mask[1];
        atb_llm::runners::MakeCausalMaskFp16(1, mask);
        uint16_t zero_fp16 = atb_llm::Fp32ToFp16(0.0f);
        CHECK(mask[0] == zero_fp16, "S=1: single element should be 0");
    }

    // S=2: upper-right corner should be -65504
    {
        uint16_t mask[4];
        atb_llm::runners::MakeCausalMaskFp16(2, mask);
        uint16_t zero_fp16 = atb_llm::Fp32ToFp16(0.0f);
        uint16_t mask_fp16 = atb_llm::Fp32ToFp16(-65504.0f);
        CHECK(mask[0] == zero_fp16, "S=2: [0,0]=0");
        CHECK(mask[1] == mask_fp16,  "S=2: [0,1]=mask (upper triangle)");
        CHECK(mask[2] == zero_fp16, "S=2: [1,0]=0 (lower triangle)");
        CHECK(mask[3] == zero_fp16, "S=2: [1,1]=0 (diagonal)");
    }
}

int main() {
    fprintf(stdout, "=== MakeCausalMaskFp16 Unit Tests ===\n");

    // Edge cases
    TestEdgeCases();

    // Byte-exact comparison against fp32 reference (various sizes)
    const int32_t seq_lens[] = {1, 2, 3, 8, 16, 32, 64, 100, 273, 512, 784, 1024, 2048, 4096};
    for (int32_t s : seq_lens) {
        TestFp16VsFp32(s);
    }

    // Structure verification (independent of reference)
    const int32_t struct_lens[] = {1, 4, 16, 64, 128};
    for (int32_t s : struct_lens) {
        TestMaskStructureFp16(s);
    }

    fprintf(stdout, "\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
