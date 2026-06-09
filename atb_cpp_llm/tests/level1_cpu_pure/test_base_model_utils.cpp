/**
 * Level 1 CPU unit tests for BaseModel utility methods.
 *
 * Covers (CPU-only, no NPU required):
 *   - EmbeddingLookup: fp16 embedding-table lookup
 *   - FindImageTokenPositions: locating image_token_id positions in input_ids
 *
 * RunPooling is NOT covered here: although LAST_TOKEN strategy is computable
 * on CPU, the InferResult buffer / dtype semantics are validated by E2E tests
 * (test_engine.py, test_e2e_full_pipeline.py) and benchmark accuracy checks.
 * MEAN and CLS strategies return ERROR_UNSUPPORTED.
 *
 * Run: ./test_base_model_utils
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "families/base_model.h"
#include "atb_llm/types.h"
#include "utils/float_utils.h"

#include <cstdint>
#include <cstring>
#include <vector>

// ── Test-only subclass exposing protected helpers ───────────────────
//
// BaseModel is abstract and the helpers under test are `protected`.
// TestableModel implements the pure-virtual IModel interface with no-op
// stubs and re-exports the helpers via `using` so they become public.
class TestableModel : public atb_llm::families::BaseModel {
public:
    using BaseModel::EmbeddingLookup;
    using BaseModel::FindImageTokenPositions;

    atb_llm::Status Load(const std::string&, atb_llm::IRuntime*) override {
        return atb_llm::STATUS_OK;
    }
    atb_llm::Status Forward(const atb_llm::InferRequest&,
                            atb_llm::InferResult&) override {
        return atb_llm::STATUS_OK;
    }
    const char* GetName() const override { return "test"; }
    bool HasVision() const override { return false; }
};

// ═══════════════════════════════════════════════════════════════════
// EmbeddingLookup Tests
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("EmbeddingLookup - basic lookup of multiple tokens") {
    constexpr int64_t vocab_size = 10;
    constexpr int64_t hidden = 4;

    // Build a deterministic fp16 embedding table: table[v][h] = v*10 + h
    std::vector<uint16_t> table(vocab_size * hidden);
    for (int64_t v = 0; v < vocab_size; v++) {
        for (int64_t h = 0; h < hidden; h++) {
            float val = static_cast<float>(v * 10 + h);
            table[v * hidden + h] = atb_llm::Fp32ToFp16(val);
        }
    }

    TestableModel model;
    std::vector<int64_t> input_ids = {2, 5, 7};
    std::vector<uint16_t> output(input_ids.size() * hidden, 0xFFFF);

    model.EmbeddingLookup(input_ids.data(),
                          static_cast<int64_t>(input_ids.size()),
                          table.data(), hidden, vocab_size, output.data());

    // Row 0 (token=2) should equal table[2*hidden .. 2*hidden+hidden)
    for (int64_t h = 0; h < hidden; h++) {
        CHECK(output[0 * hidden + h] == table[2 * hidden + h]);
        CHECK(output[1 * hidden + h] == table[5 * hidden + h]);
        CHECK(output[2 * hidden + h] == table[7 * hidden + h]);
    }

    // And the float values should round-trip back to v*10+h
    CHECK(atb_llm::Fp16ToF32(output[0]) == doctest::Approx(20.0f));
    CHECK(atb_llm::Fp16ToF32(output[hidden]) == doctest::Approx(50.0f));
    CHECK(atb_llm::Fp16ToF32(output[2 * hidden + 3]) == doctest::Approx(73.0f));
}

TEST_CASE("EmbeddingLookup - out-of-bounds token zero-fills") {
    constexpr int64_t vocab_size = 10;
    constexpr int64_t hidden = 4;

    std::vector<uint16_t> table(vocab_size * hidden);
    for (size_t i = 0; i < table.size(); i++) {
        table[i] = atb_llm::Fp32ToFp16(1.0f);  // all ones
    }

    TestableModel model;
    // token 999 is OOB; token 3 is valid
    std::vector<int64_t> input_ids = {3, 999};
    std::vector<uint16_t> output(input_ids.size() * hidden, 0xBEEF);

    // Should not crash
    model.EmbeddingLookup(input_ids.data(),
                          static_cast<int64_t>(input_ids.size()),
                          table.data(), hidden, vocab_size, output.data());

    // Row 0 (valid): should copy the all-ones row
    for (int64_t h = 0; h < hidden; h++) {
        CHECK(output[h] == atb_llm::Fp32ToFp16(1.0f));
    }
    // Row 1 (OOB): zero-filled
    for (int64_t h = 0; h < hidden; h++) {
        CHECK(output[hidden + h] == 0);
    }
}

TEST_CASE("EmbeddingLookup - negative token id zero-fills") {
    constexpr int64_t vocab_size = 5;
    constexpr int64_t hidden = 3;

    std::vector<uint16_t> table(vocab_size * hidden,
                                atb_llm::Fp32ToFp16(7.0f));

    TestableModel model;
    std::vector<int64_t> input_ids = {-1};
    std::vector<uint16_t> output(hidden, 0xCAFE);

    model.EmbeddingLookup(input_ids.data(), 1,
                          table.data(), hidden, vocab_size, output.data());

    for (int64_t h = 0; h < hidden; h++) {
        CHECK(output[h] == 0);
    }
}

TEST_CASE("EmbeddingLookup - boundary tokens (0 and vocab_size-1)") {
    constexpr int64_t vocab_size = 8;
    constexpr int64_t hidden = 2;

    // table[v][h] = (v+1)*100 + h
    std::vector<uint16_t> table(vocab_size * hidden);
    for (int64_t v = 0; v < vocab_size; v++) {
        for (int64_t h = 0; h < hidden; h++) {
            table[v * hidden + h] =
                atb_llm::Fp32ToFp16(static_cast<float>((v + 1) * 100 + h));
        }
    }

    TestableModel model;
    std::vector<int64_t> input_ids = {0, vocab_size - 1};
    std::vector<uint16_t> output(input_ids.size() * hidden, 0);

    model.EmbeddingLookup(input_ids.data(),
                          static_cast<int64_t>(input_ids.size()),
                          table.data(), hidden, vocab_size, output.data());

    // token=0 -> [100, 101]
    CHECK(atb_llm::Fp16ToF32(output[0]) == doctest::Approx(100.0f));
    CHECK(atb_llm::Fp16ToF32(output[1]) == doctest::Approx(101.0f));
    // token=vocab_size-1=7 -> [800, 801]
    CHECK(atb_llm::Fp16ToF32(output[hidden + 0]) == doctest::Approx(800.0f));
    CHECK(atb_llm::Fp16ToF32(output[hidden + 1]) == doctest::Approx(801.0f));
}

TEST_CASE("EmbeddingLookup - empty input (seq_len=0) is a no-op") {
    constexpr int64_t vocab_size = 4;
    constexpr int64_t hidden = 3;

    std::vector<uint16_t> table(vocab_size * hidden,
                                atb_llm::Fp32ToFp16(2.0f));

    TestableModel model;
    // output buffer is empty; should not be touched
    std::vector<uint16_t> output;  // size 0

    model.EmbeddingLookup(nullptr, 0, table.data(), hidden, vocab_size,
                          output.data());

    CHECK(output.empty());
}

// ═══════════════════════════════════════════════════════════════════
// FindImageTokenPositions Tests
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("FindImageTokenPositions - basic scattered matches") {
    std::vector<int64_t> ids = {1, 151655, 2, 151655, 3};
    auto positions = TestableModel::FindImageTokenPositions(
        ids.data(), static_cast<int64_t>(ids.size()), 151655);
    REQUIRE(positions.size() == 2);
    CHECK(positions[0] == 1);
    CHECK(positions[1] == 3);
}

TEST_CASE("FindImageTokenPositions - no matches") {
    std::vector<int64_t> ids = {1, 2, 3, 4, 5};
    auto positions = TestableModel::FindImageTokenPositions(
        ids.data(), static_cast<int64_t>(ids.size()), 151655);
    CHECK(positions.empty());
}

TEST_CASE("FindImageTokenPositions - all tokens are image tokens") {
    constexpr int64_t kImg = 151655;
    std::vector<int64_t> ids = {kImg, kImg, kImg, kImg};
    auto positions = TestableModel::FindImageTokenPositions(
        ids.data(), static_cast<int64_t>(ids.size()), kImg);
    REQUIRE(positions.size() == ids.size());
    for (int64_t i = 0; i < static_cast<int64_t>(ids.size()); i++) {
        CHECK(positions[i] == i);
    }
}

TEST_CASE("FindImageTokenPositions - empty input returns empty") {
    auto positions = TestableModel::FindImageTokenPositions(nullptr, 0, 151655);
    CHECK(positions.empty());
}

TEST_CASE("FindImageTokenPositions - single match at start") {
    std::vector<int64_t> ids = {42, 1, 2, 3};
    auto positions = TestableModel::FindImageTokenPositions(
        ids.data(), static_cast<int64_t>(ids.size()), 42);
    REQUIRE(positions.size() == 1);
    CHECK(positions[0] == 0);
}

TEST_CASE("FindImageTokenPositions - single match at end") {
    std::vector<int64_t> ids = {1, 2, 3, 42};
    auto positions = TestableModel::FindImageTokenPositions(
        ids.data(), static_cast<int64_t>(ids.size()), 42);
    REQUIRE(positions.size() == 1);
    CHECK(positions[0] == 3);
}

TEST_CASE("FindImageTokenPositions - consecutive matches preserved in order") {
    std::vector<int64_t> ids = {1, 99, 99, 99, 2, 99};
    auto positions = TestableModel::FindImageTokenPositions(
        ids.data(), static_cast<int64_t>(ids.size()), 99);
    REQUIRE(positions.size() == 4);
    CHECK(positions[0] == 1);
    CHECK(positions[1] == 2);
    CHECK(positions[2] == 3);
    CHECK(positions[3] == 5);
}

// ═══════════════════════════════════════════════════════════════════
// NOTE on RunPooling
// ═══════════════════════════════════════════════════════════════════
//
// RunPooling needs NPU; covered by E2E tests (test_engine.py /
// test_e2e_full_pipeline.py / benchmark accuracy checks).
// In particular, the LAST_TOKEN + L2-normalize path is what the engine
// uses, and is validated end-to-end against the transformers reference.
// MEAN and CLS strategies currently return ERROR_UNSUPPORTED.
