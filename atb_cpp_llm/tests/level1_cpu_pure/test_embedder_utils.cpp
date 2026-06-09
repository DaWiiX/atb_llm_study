// Level 1: CPU-only unit tests for Embedder utilities
//   1. LAST_TOKEN_BY_MASK pooling
//   2. L2 normalization (RunPooling with normalize=true)
//
// No NPU dependency.  Runs on any host.

#include "families/base_model.h"
#include "atb_llm/types.h"
#include "utils/float_utils.h"
#include "log/logger.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <cmath>

using atb_llm::families::BaseModel;

// Test helper: expose protected methods via using-declarations
class TestablePooling : public BaseModel {
public:
    using BaseModel::RunPooling;

    atb_llm::Status Load(const std::string&, atb_llm::IRuntime*) override {
        return atb_llm::STATUS_OK;
    }
    atb_llm::Status Forward(const atb_llm::InferRequest&,
                             atb_llm::InferResult&) override {
        return atb_llm::STATUS_OK;
    }
    atb_llm::Status ForwardWithTiming(const atb_llm::InferRequest&,
                                       atb_llm::InferResult&,
                                       atb_llm::StageTimings&) override {
        return atb_llm::STATUS_OK;
    }
    const char* GetName() const override { return "TestablePooling"; }
    bool HasVision() const override { return false; }
};

static void FillH16(uint16_t* buf, int64_t n, float val) {
    uint16_t v16 = atb_llm::Fp32ToFp16(val);
    for (int64_t i = 0; i < n; i++) buf[i] = v16;
}

static float H16toF32(uint16_t v) {
    return atb_llm::Fp16ToF32(v);
}

TEST_CASE("RunPooling LAST_TOKEN_BY_MASK") {
    TestablePooling tester;
    const int64_t S = 5, D = 4;
    std::vector<uint16_t> hidden(S * D);

    // Token 0: all 1.0, Token 1: all 2.0, ..., Token 4: all 5.0
    for (int64_t s = 0; s < S; s++)
        FillH16(&hidden[s * D], D, static_cast<float>(s + 1));

    SUBCASE("mask all ones -> last token (S-1)") {
        int64_t mask[5] = {1, 1, 1, 1, 1};
        atb_llm::InferResult r;
        atb_llm::Status st = tester.RunPooling(
            hidden.data(), S, D, false,
            BaseModel::PoolingStrategy::LAST_TOKEN_BY_MASK, r, mask);
        REQUIRE(st == atb_llm::STATUS_OK);

        auto* out = r.As<uint16_t>();
        for (int64_t i = 0; i < D; i++)
            CHECK(H16toF32(out[i]) == doctest::Approx(5.0f));
    }

    SUBCASE("mask [1,1,1,0,0] -> token 2 (index 2)") {
        int64_t mask[5] = {1, 1, 1, 0, 0};
        atb_llm::InferResult r;
        atb_llm::Status st = tester.RunPooling(
            hidden.data(), S, D, false,
            BaseModel::PoolingStrategy::LAST_TOKEN_BY_MASK, r, mask);
        REQUIRE(st == atb_llm::STATUS_OK);

        auto* out = r.As<uint16_t>();
        for (int64_t i = 0; i < D; i++)
            CHECK(H16toF32(out[i]) == doctest::Approx(3.0f));
    }

    SUBCASE("mask [1,0,1,0,1] -> last valid = token 4") {
        int64_t mask[5] = {1, 0, 1, 0, 1};
        atb_llm::InferResult r;
        atb_llm::Status st = tester.RunPooling(
            hidden.data(), S, D, false,
            BaseModel::PoolingStrategy::LAST_TOKEN_BY_MASK, r, mask);
        REQUIRE(st == atb_llm::STATUS_OK);

        auto* out = r.As<uint16_t>();
        for (int64_t i = 0; i < D; i++)
            CHECK(H16toF32(out[i]) == doctest::Approx(5.0f));
    }

    SUBCASE("mask all zeros -> falls back to last token (S-1)") {
        int64_t mask[5] = {0, 0, 0, 0, 0};
        atb_llm::InferResult r;
        atb_llm::Status st = tester.RunPooling(
            hidden.data(), S, D, false,
            BaseModel::PoolingStrategy::LAST_TOKEN_BY_MASK, r, mask);
        REQUIRE(st == atb_llm::STATUS_OK);

        auto* out = r.As<uint16_t>();
        for (int64_t i = 0; i < D; i++)
            CHECK(H16toF32(out[i]) == doctest::Approx(5.0f));
    }
}

TEST_CASE("RunPooling L2 normalization") {
    TestablePooling tester;
    const int64_t D = 4;
    std::vector<uint16_t> hidden(D);

    // Simple fp16 pattern: [3.0, 4.0, 0.0, 0.0] -- L2 norm = 5.0
    hidden[0] = atb_llm::Fp32ToFp16(3.0f);
    hidden[1] = atb_llm::Fp32ToFp16(4.0f);
    hidden[2] = atb_llm::Fp32ToFp16(0.0f);
    hidden[3] = atb_llm::Fp32ToFp16(0.0f);

    int64_t mask[1] = {1};
    atb_llm::InferResult r;
    atb_llm::Status st = tester.RunPooling(
        hidden.data(), 1, D, true,  // normalize=true
        BaseModel::PoolingStrategy::LAST_TOKEN_BY_MASK, r, mask);
    REQUIRE(st == atb_llm::STATUS_OK);

    auto* out = r.As<uint16_t>();
    // Expected: [3/5, 4/5, 0, 0] = [0.6, 0.8, 0, 0]
    CHECK(H16toF32(out[0]) == doctest::Approx(0.6f).epsilon(1e-3f));
    CHECK(H16toF32(out[1]) == doctest::Approx(0.8f).epsilon(1e-3f));
    CHECK(std::fabs(H16toF32(out[2])) < 1e-4f);
    CHECK(std::fabs(H16toF32(out[3])) < 1e-4f);

    // Verify L2 norm ~1.0
    float norm2 = 0;
    for (int64_t i = 0; i < D; i++) {
        float v = H16toF32(out[i]);
        norm2 += v * v;
    }
    CHECK(std::sqrt(norm2) == doctest::Approx(1.0f).epsilon(1e-3f));
}

TEST_CASE("LAST_TOKEN with nullptr mask still works (backward compat)") {
    TestablePooling tester;
    const int64_t S = 3, D = 2;
    std::vector<uint16_t> hidden(S * D);
    FillH16(&hidden[0 * D], D, 10.0f);
    FillH16(&hidden[1 * D], D, 20.0f);
    FillH16(&hidden[2 * D], D, 30.0f);

    atb_llm::InferResult r;
    atb_llm::Status st = tester.RunPooling(
        hidden.data(), S, D, false,
        BaseModel::PoolingStrategy::LAST_TOKEN, r);
    REQUIRE(st == atb_llm::STATUS_OK);

    auto* out = r.As<uint16_t>();
    CHECK(H16toF32(out[0]) == doctest::Approx(30.0f));
    CHECK(H16toF32(out[1]) == doctest::Approx(30.0f));
}
