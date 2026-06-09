/**
 * Level 1 CPU pure-function tests: .bin format round-trip.
 *
 * Validates the three binary interchange formats used for C++ <-> Python
 * data exchange:
 *
 *   Format A (tokens, int64 elements):
 *     [int32 count][int64 element_0]...[int64 element_{count-1}]
 *
 *   Format B (pixel_values / fp16, uint16 elements):
 *     [int32 count][uint16 element_0]...[uint16 element_{count-1}]
 *     The uint16 values contain IEEE fp16 BIT PATTERNS (not integer
 *     values).  The reader MUST reinterpret the bits as float16, NOT
 *     cast.  This is the source of a historical bug where Python used
 *     .astype(np.float16) instead of .view(np.float16).
 *
 *   Format C (embedding output, fp32):
 *     [int64 dim][float32 element_0]...[float32 element_{dim-1}]
 *
 * Each TEST_CASE:
 *   1. Creates known reference data in memory
 *   2. Writes it to /tmp/test_bin_{format}.bin
 *   3. Reads it back from the file
 *   4. Asserts bit-exact equality with the original
 *
 * The companion Python script (test_bin_format.py) then loads these
 * files, cross-verifies the values, and performs its own round-trips.
 *
 * NO NPU required — pure-CPU file I/O and value comparison.
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <limits>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

// ═══════════════════════════════════════════════════════════════════
// Helper: write / read Format B (fp16)
//   [int32 count][uint16 * count]
// ═══════════════════════════════════════════════════════════════════
static void SaveBinFp16(const char* path, const uint16_t* data, int32_t count) {
    FILE* f = std::fopen(path, "wb");
    REQUIRE(f != nullptr);
    std::fwrite(&count, sizeof(int32_t), 1, f);
    std::fwrite(data, sizeof(uint16_t), static_cast<size_t>(count), f);
    std::fclose(f);
}

static std::vector<uint16_t> LoadBinFp16(const char* path) {
    FILE* f = std::fopen(path, "rb");
    REQUIRE(f != nullptr);
    int32_t count = 0;
    size_t nread = std::fread(&count, sizeof(int32_t), 1, f);
    REQUIRE(nread == 1);
    REQUIRE(count > 0);
    std::vector<uint16_t> data(static_cast<size_t>(count));
    nread = std::fread(data.data(), sizeof(uint16_t),
                       static_cast<size_t>(count), f);
    REQUIRE(nread == static_cast<size_t>(count));
    std::fclose(f);
    return data;
}

// ═══════════════════════════════════════════════════════════════════
// Helper: write / read Format A (int64)
//   [int32 count][int64 * count]
// ═══════════════════════════════════════════════════════════════════
static void SaveBinInt64(const char* path, const int64_t* data, int32_t count) {
    FILE* f = std::fopen(path, "wb");
    REQUIRE(f != nullptr);
    std::fwrite(&count, sizeof(int32_t), 1, f);
    std::fwrite(data, sizeof(int64_t), static_cast<size_t>(count), f);
    std::fclose(f);
}

static std::vector<int64_t> LoadBinInt64(const char* path) {
    FILE* f = std::fopen(path, "rb");
    REQUIRE(f != nullptr);
    int32_t count = 0;
    size_t nread = std::fread(&count, sizeof(int32_t), 1, f);
    REQUIRE(nread == 1);
    REQUIRE(count > 0);
    std::vector<int64_t> data(static_cast<size_t>(count));
    nread = std::fread(data.data(), sizeof(int64_t),
                       static_cast<size_t>(count), f);
    REQUIRE(nread == static_cast<size_t>(count));
    std::fclose(f);
    return data;
}

// ═══════════════════════════════════════════════════════════════════
// Helper: write / read Format C (fp32)
//   [int64 dim][float32 * dim]
// ═══════════════════════════════════════════════════════════════════
static void SaveBinFp32(const char* path, const float* data, int64_t dim) {
    FILE* f = std::fopen(path, "wb");
    REQUIRE(f != nullptr);
    std::fwrite(&dim, sizeof(int64_t), 1, f);
    std::fwrite(data, sizeof(float), static_cast<size_t>(dim), f);
    std::fclose(f);
}

static std::vector<float> LoadBinFp32(const char* path) {
    FILE* f = std::fopen(path, "rb");
    REQUIRE(f != nullptr);
    int64_t dim = 0;
    size_t nread = std::fread(&dim, sizeof(int64_t), 1, f);
    REQUIRE(nread == 1);
    REQUIRE(dim > 0);
    std::vector<float> data(static_cast<size_t>(dim));
    nread = std::fread(data.data(), sizeof(float),
                       static_cast<size_t>(dim), f);
    REQUIRE(nread == static_cast<size_t>(dim));
    std::fclose(f);
    return data;
}

// ═══════════════════════════════════════════════════════════════════
// Part 1: Format B — fp16 write-then-read self-consistency
// ═══════════════════════════════════════════════════════════════════
TEST_CASE("Format B: fp16 round-trip (uint16 bit patterns)") {
    // Known fp16 bit patterns (IEEE 754-2008 binary16).
    // These are BIT PATTERNS, not integer values — the reader must
    // reinterpret as float16, NOT cast.
    //
    //  0x0000 = +0.0       0x8000 = -0.0
    //  0x3C00 =  1.0       0xBC00 = -1.0
    //  0x7C00 = +inf       0xFC00 = -inf
    //  0x7E00 =  NaN (quiet, canonical)
    //  0x0001 = smallest positive subnormal (≈ 5.96e-8)
    //  0x0400 = smallest positive normal   (≈ 6.10e-5)
    //  0x7BFF = largest finite           (65504.0)
    //  0x4200 =  3.0       0x4400 =  4.0
    const uint16_t original[] = {
        0x0000,  // +0.0
        0x3C00,  //  1.0
        0xBC00,  // -1.0
        0x7C00,  // +inf
        0xFC00,  // -inf
        0x7E00,  //  NaN
        0x0001,  // smallest positive subnormal
        0x0400,  // smallest positive normal
        0x7BFF,  // largest finite
        0x4200,  //  3.0
        0x4400,  //  4.0
    };
    const int32_t N = static_cast<int32_t>(sizeof(original) / sizeof(original[0]));

    // Write
    SaveBinFp16("/tmp/test_bin_fp16.bin", original, N);

    // Read back
    std::vector<uint16_t> loaded = LoadBinFp16("/tmp/test_bin_fp16.bin");

    // Verify
    REQUIRE(loaded.size() == static_cast<size_t>(N));

    const char* names[] = {
        "+0.0", "1.0", "-1.0", "+inf", "-inf",
        "NaN", "denorm-min", "norm-min", "norm-max",
        "3.0", "4.0",
    };
    for (int32_t i = 0; i < N; i++) {
        INFO("idx=", i, " (", names[i], "): expected 0x",
             std::hex, original[i], " got 0x", loaded[i]);
        CHECK(loaded[i] == original[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Part 2: Format A — int64 write-then-read self-consistency
// ═══════════════════════════════════════════════════════════════════
TEST_CASE("Format A: int64 round-trip") {
    // Token IDs and boundary values.
    // 151643, 151652, 151655 are real Qwen2 tokenizer special tokens.
    const int64_t original[] = {
        151643,
        151652,
        151655,
        0,
        -1,
        std::numeric_limits<int64_t>::max(),
        std::numeric_limits<int64_t>::min(),
    };
    const int32_t N = static_cast<int32_t>(sizeof(original) / sizeof(original[0]));

    // Write
    SaveBinInt64("/tmp/test_bin_int64.bin", original, N);

    // Read back
    std::vector<int64_t> loaded = LoadBinInt64("/tmp/test_bin_int64.bin");

    // Verify
    REQUIRE(loaded.size() == static_cast<size_t>(N));
    for (int32_t i = 0; i < N; i++) {
        INFO("idx=", i, ": expected ", original[i], " got ", loaded[i]);
        CHECK(loaded[i] == original[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Part 3: Format C — fp32 write-then-read self-consistency
// ═══════════════════════════════════════════════════════════════════
TEST_CASE("Format C: fp32 round-trip") {
    const float original[] = {
         0.0f,
         1.0f,
        -1.0f,
         3.14159f,
         1.0e-7f,
         1.0e7f,
    };
    const int64_t D = static_cast<int64_t>(sizeof(original) / sizeof(original[0]));

    // Write
    SaveBinFp32("/tmp/test_bin_fp32.bin", original, D);

    // Read back
    std::vector<float> loaded = LoadBinFp32("/tmp/test_bin_fp32.bin");

    // Verify
    REQUIRE(loaded.size() == static_cast<size_t>(D));

    // Bit-exact comparison via memcmp — fp32 I/O should be lossless
    for (int64_t i = 0; i < D; i++) {
        bool bit_exact = (std::memcmp(&original[i], &loaded[i], sizeof(float)) == 0);
        INFO("idx=", i, ": expected ", original[i], " got ", loaded[i]);
        CHECK(bit_exact);
    }
}
