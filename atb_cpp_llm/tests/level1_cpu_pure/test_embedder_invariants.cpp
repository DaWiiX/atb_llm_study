// Level 1: CPU-only unit test for Qwen3VLEmbedder input/output invariants.
//
// These checks live at the API boundary (before any NPU work happens),
// so they're testable without a model checkpoint.
//
// Covers:
//   - batch_size must be 1
//   - input_ids must be non-null and seq_length > 0
//   - call before Load() returns ERROR_INVALID_PARAM (input validation
//     runs first, so the message is about the actual contract violation,
//     and only after that do we check engine_ readiness)

#include "atb_llm/embedder.h"
#include "atb_llm/types.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <vector>

using atb_llm::Qwen3VLEmbedder;
using atb_llm::InferRequest;
using atb_llm::InferResult;
using atb_llm::Status;
using atb_llm::STATUS_OK;
using atb_llm::ERROR_INVALID_PARAM;
using atb_llm::InputMode;

TEST_CASE("Qwen3VLEmbedder rejects batch_size != 1") {
    Qwen3VLEmbedder emb;
    std::vector<int64_t> ids = {1, 2, 3};

    InferRequest req;
    req.mode = InputMode::TEXT_ONLY;
    req.text.input_ids = ids.data();
    req.text.batch_size = 2;   // ← contract violation
    req.text.seq_length = 3;

    InferResult out;
    CHECK(emb.Encode(req, out) == ERROR_INVALID_PARAM);
}

TEST_CASE("Qwen3VLEmbedder rejects empty input_ids") {
    Qwen3VLEmbedder emb;

    SUBCASE("null input_ids") {
        InferRequest req;
        req.text.batch_size = 1;
        req.text.seq_length = 5;
        req.text.input_ids = nullptr;
        InferResult out;
        CHECK(emb.Encode(req, out) == ERROR_INVALID_PARAM);
    }

    SUBCASE("seq_length == 0") {
        std::vector<int64_t> ids = {1};
        InferRequest req;
        req.text.batch_size = 1;
        req.text.seq_length = 0;
        req.text.input_ids = ids.data();
        InferResult out;
        CHECK(emb.Encode(req, out) == ERROR_INVALID_PARAM);
    }
}

TEST_CASE("Qwen3VLEmbedder rejects Encode before Load") {
    // A valid-looking request, but the embedder hasn't been Load()ed.
    // Input validation runs first (passes), then engine_ == nullptr
    // triggers ERROR_INVALID_PARAM.
    Qwen3VLEmbedder emb;
    std::vector<int64_t> ids = {1, 2, 3};

    InferRequest req;
    req.text.batch_size = 1;
    req.text.seq_length = 3;
    req.text.input_ids = ids.data();

    InferResult out;
    CHECK(emb.Encode(req, out) == ERROR_INVALID_PARAM);
}
