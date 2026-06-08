/**
 * Unit tests for IO/Adapter components using doctest.
 *
 * Covers:
 *   - SafetensorsReader (load, metadata, move semantics, error paths)
 *   - WeightLoader (delegation to reader, error paths)
 *   - Qwen3VLConfig (JSON parsing, defaults, derived fields)
 *   - Qwen3VLPreprocess (SmartResize, BicubicResize)
 *
 * Run: ./test_io_adapters
 * No NPU required — pure CPU/file tests.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "io/safetensors_reader.h"
#include "io/weight_loader.h"
#include "io/json_config.h"
#include "adapters/qwen3vl_embedding/qwen3vl_config.h"
#include "adapters/qwen3vl_embedding/qwen3vl_preprocess.h"

// For creating test safetensors files
#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <array>

// ── Helper: create a minimal safetensors file with known tensors ─────
static std::string CreateTestSafetensors(const std::string& path) {
    safetensors::safetensors_t st;

    // Tensor "weight1": float32 [2, 3]
    {
        float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        size_t data_offset = st.storage.size();
        st.storage.insert(st.storage.end(),
                          reinterpret_cast<uint8_t*>(data),
                          reinterpret_cast<uint8_t*>(data) + sizeof(data));

        safetensors::tensor_t t;
        t.dtype = safetensors::kFLOAT32;
        t.shape = {2, 3};
        t.data_offsets = {data_offset, data_offset + sizeof(data)};
        st.tensors.insert("weight1", t);
    }

    // Tensor "weight2": float16 [4]
    {
        uint16_t data[4] = {
            safetensors::float_to_fp16(1.5f),
            safetensors::float_to_fp16(2.5f),
            safetensors::float_to_fp16(3.5f),
            safetensors::float_to_fp16(4.5f)};
        size_t data_offset = st.storage.size();
        st.storage.insert(st.storage.end(),
                          reinterpret_cast<uint8_t*>(data),
                          reinterpret_cast<uint8_t*>(data) + sizeof(data));

        safetensors::tensor_t t;
        t.dtype = safetensors::kFLOAT16;
        t.shape = {4};
        t.data_offsets = {data_offset, data_offset + sizeof(data)};
        st.tensors.insert("weight2", t);
    }

    // Tensor "layer.0.weight": bfloat16 [2]
    {
        uint16_t data[2] = {
            safetensors::float_to_bfloat16(10.0f),
            safetensors::float_to_bfloat16(20.0f)};
        size_t data_offset = st.storage.size();
        st.storage.insert(st.storage.end(),
                          reinterpret_cast<uint8_t*>(data),
                          reinterpret_cast<uint8_t*>(data) + sizeof(data));

        safetensors::tensor_t t;
        t.dtype = safetensors::kBFLOAT16;
        t.shape = {2};
        t.data_offsets = {data_offset, data_offset + sizeof(data)};
        st.tensors.insert("layer.0.weight", t);
    }

    // Tensor "layer.1.weight": int64 [3]
    {
        int64_t data[3] = {100, 200, 300};
        size_t data_offset = st.storage.size();
        st.storage.insert(st.storage.end(),
                          reinterpret_cast<uint8_t*>(data),
                          reinterpret_cast<uint8_t*>(data) + sizeof(data));

        safetensors::tensor_t t;
        t.dtype = safetensors::kINT64;
        t.shape = {3};
        t.data_offsets = {data_offset, data_offset + sizeof(data)};
        st.tensors.insert("layer.1.weight", t);
    }

    // Add metadata
    st.metadata.insert("format", "pt");

    std::string warn, err;
    bool ok = safetensors::save_to_file(st, path, &warn, &err);
    REQUIRE(ok);
    return path;
}

// ── Helper: create a temporary JSON config file ─────────────────────
static void WriteFile(const std::string& path, const std::string& content) {
    std::ofstream ofs(path);
    REQUIRE(ofs.is_open());
    ofs << content;
    ofs.close();
}

static std::string TmpDir() {
    // Use /tmp for temp files
    return "/tmp/test_io_adapters_" + std::to_string(getpid());
}

static void Mkdir(const std::string& dir) {
    std::string cmd = "mkdir -p " + dir;
    int ret = system(cmd.c_str());
    (void)ret;
}

// ═══════════════════════════════════════════════════════════════════
// SafetensorsReader Tests
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("SafetensorsReader - default construction") {
    atb_llm::SafetensorsReader reader;
    CHECK(reader.NumTensors() == 0);
    CHECK_FALSE(reader.HasKey("anything"));
    CHECK(reader.GetAllKeys().empty());
    CHECK(reader.GetKeysByPrefix("x").empty());
    CHECK(reader.GetTensorData("x") == nullptr);
}

TEST_CASE("SafetensorsReader - LoadFromFile nonexistent") {
    atb_llm::SafetensorsReader reader;
    atb_llm::Status s = reader.LoadFromFile("/nonexistent/path/model.safetensors");
    CHECK(s == atb_llm::ERROR_FILE_NOT_FOUND);
}

TEST_CASE("SafetensorsReader - LoadFromFile valid") {
    std::string tmpdir = TmpDir();
    Mkdir(tmpdir);
    std::string path = tmpdir + "/test.safetensors";
    CreateTestSafetensors(path);

    atb_llm::SafetensorsReader reader;
    atb_llm::Status s = reader.LoadFromFile(path);
    CHECK(s == atb_llm::STATUS_OK);
    CHECK(reader.NumTensors() == 4);

    // Cleanup
    std::remove(path.c_str());
    std::remove(tmpdir.c_str());
}

TEST_CASE("SafetensorsReader - HasKey and GetAllKeys") {
    std::string tmpdir = TmpDir();
    Mkdir(tmpdir);
    std::string path = tmpdir + "/test.safetensors";
    CreateTestSafetensors(path);

    atb_llm::SafetensorsReader reader;
    reader.LoadFromFile(path);

    CHECK(reader.HasKey("weight1"));
    CHECK(reader.HasKey("weight2"));
    CHECK(reader.HasKey("layer.0.weight"));
    CHECK(reader.HasKey("layer.1.weight"));
    CHECK_FALSE(reader.HasKey("nonexistent"));

    auto all_keys = reader.GetAllKeys();
    CHECK(all_keys.size() == 4);

    std::remove(path.c_str());
    std::remove(tmpdir.c_str());
}

TEST_CASE("SafetensorsReader - GetKeysByPrefix") {
    std::string tmpdir = TmpDir();
    Mkdir(tmpdir);
    std::string path = tmpdir + "/test.safetensors";
    CreateTestSafetensors(path);

    atb_llm::SafetensorsReader reader;
    reader.LoadFromFile(path);

    auto layer_keys = reader.GetKeysByPrefix("layer.");
    CHECK(layer_keys.size() == 2);

    auto weight_keys = reader.GetKeysByPrefix("weight");
    CHECK(weight_keys.size() == 2);

    auto none_keys = reader.GetKeysByPrefix("xyz");
    CHECK(none_keys.empty());

    std::remove(path.c_str());
    std::remove(tmpdir.c_str());
}

TEST_CASE("SafetensorsReader - GetTensor metadata") {
    std::string tmpdir = TmpDir();
    Mkdir(tmpdir);
    std::string path = tmpdir + "/test.safetensors";
    CreateTestSafetensors(path);

    atb_llm::SafetensorsReader reader;
    reader.LoadFromFile(path);

    atb_llm::WeightInfo info;

    // weight1: float32 [2, 3]
    CHECK(reader.GetTensor("weight1", info) == atb_llm::STATUS_OK);
    CHECK(info.shape.size() == 2);
    CHECK(info.shape[0] == 2);
    CHECK(info.shape[1] == 3);
    CHECK(info.nbytes == 24);  // 6 * 4 bytes
    CHECK(info.dtype == static_cast<int>(safetensors::kFLOAT32));

    // weight2: float16 [4]
    CHECK(reader.GetTensor("weight2", info) == atb_llm::STATUS_OK);
    CHECK(info.shape.size() == 1);
    CHECK(info.shape[0] == 4);
    CHECK(info.nbytes == 8);  // 4 * 2 bytes

    // nonexistent
    CHECK(reader.GetTensor("no_such_key", info) == atb_llm::ERROR_WEIGHT_LOAD);

    std::remove(path.c_str());
    std::remove(tmpdir.c_str());
}

TEST_CASE("SafetensorsReader - GetTensorData") {
    std::string tmpdir = TmpDir();
    Mkdir(tmpdir);
    std::string path = tmpdir + "/test.safetensors";
    CreateTestSafetensors(path);

    atb_llm::SafetensorsReader reader;
    reader.LoadFromFile(path);

    // weight1: float32 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    const uint8_t* data = reader.GetTensorData("weight1");
    REQUIRE(data != nullptr);
    const float* fdata = reinterpret_cast<const float*>(data);
    CHECK(fdata[0] == doctest::Approx(1.0f));
    CHECK(fdata[1] == doctest::Approx(2.0f));
    CHECK(fdata[5] == doctest::Approx(6.0f));

    // nonexistent
    CHECK(reader.GetTensorData("no_such_key") == nullptr);

    std::remove(path.c_str());
    std::remove(tmpdir.c_str());
}

TEST_CASE("SafetensorsReader - move construction") {
    std::string tmpdir = TmpDir();
    Mkdir(tmpdir);
    std::string path = tmpdir + "/test.safetensors";
    CreateTestSafetensors(path);

    atb_llm::SafetensorsReader reader1;
    reader1.LoadFromFile(path);
    CHECK(reader1.NumTensors() == 4);

    // Move construct
    atb_llm::SafetensorsReader reader2(std::move(reader1));
    CHECK(reader2.NumTensors() == 4);
    CHECK(reader2.HasKey("weight1"));

    std::remove(path.c_str());
    std::remove(tmpdir.c_str());
}

TEST_CASE("SafetensorsReader - move assignment") {
    std::string tmpdir = TmpDir();
    Mkdir(tmpdir);
    std::string path = tmpdir + "/test.safetensors";
    CreateTestSafetensors(path);

    atb_llm::SafetensorsReader reader1;
    reader1.LoadFromFile(path);

    atb_llm::SafetensorsReader reader2;
    reader2 = std::move(reader1);
    CHECK(reader2.NumTensors() == 4);
    CHECK(reader2.HasKey("layer.0.weight"));

    std::remove(path.c_str());
    std::remove(tmpdir.c_str());
}

TEST_CASE("SafetensorsReader - reload overwrites") {
    std::string tmpdir = TmpDir();
    Mkdir(tmpdir);
    std::string path = tmpdir + "/test.safetensors";
    CreateTestSafetensors(path);

    atb_llm::SafetensorsReader reader;
    reader.LoadFromFile(path);
    CHECK(reader.NumTensors() == 4);

    // Reload same file
    atb_llm::Status s = reader.LoadFromFile(path);
    CHECK(s == atb_llm::STATUS_OK);
    CHECK(reader.NumTensors() == 4);

    std::remove(path.c_str());
    std::remove(tmpdir.c_str());
}

// ═══════════════════════════════════════════════════════════════════
// WeightLoader Tests
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("WeightLoader - default construction") {
    atb_llm::WeightLoader loader;
    CHECK(loader.NumTensors() == 0);
}

TEST_CASE("WeightLoader - LoadFromFile nonexistent") {
    atb_llm::WeightLoader loader;
    atb_llm::Status s = loader.LoadFromFile("/nonexistent/model.safetensors");
    CHECK(s == atb_llm::ERROR_FILE_NOT_FOUND);
}

TEST_CASE("WeightLoader - LoadFromFile valid") {
    std::string tmpdir = TmpDir();
    Mkdir(tmpdir);
    std::string path = tmpdir + "/test.safetensors";
    CreateTestSafetensors(path);

    atb_llm::WeightLoader loader;
    atb_llm::Status s = loader.LoadFromFile(path);
    CHECK(s == atb_llm::STATUS_OK);
    CHECK(loader.NumTensors() == 4);

    std::remove(path.c_str());
    std::remove(tmpdir.c_str());
}

TEST_CASE("WeightLoader - GetTensor delegates to reader") {
    std::string tmpdir = TmpDir();
    Mkdir(tmpdir);
    std::string path = tmpdir + "/test.safetensors";
    CreateTestSafetensors(path);

    atb_llm::WeightLoader loader;
    loader.LoadFromFile(path);

    atb_llm::WeightInfo info;
    CHECK(loader.GetTensor("weight1", info) == atb_llm::STATUS_OK);
    CHECK(info.shape[0] == 2);
    CHECK(info.shape[1] == 3);
    CHECK(loader.GetTensor("no_key", info) == atb_llm::ERROR_WEIGHT_LOAD);

    std::remove(path.c_str());
    std::remove(tmpdir.c_str());
}

TEST_CASE("WeightLoader - GetKeysByPrefix delegates") {
    std::string tmpdir = TmpDir();
    Mkdir(tmpdir);
    std::string path = tmpdir + "/test.safetensors";
    CreateTestSafetensors(path);

    atb_llm::WeightLoader loader;
    loader.LoadFromFile(path);

    auto keys = loader.GetKeysByPrefix("layer.");
    CHECK(keys.size() == 2);

    std::remove(path.c_str());
    std::remove(tmpdir.c_str());
}

TEST_CASE("WeightLoader - GetTensorData delegates") {
    std::string tmpdir = TmpDir();
    Mkdir(tmpdir);
    std::string path = tmpdir + "/test.safetensors";
    CreateTestSafetensors(path);

    atb_llm::WeightLoader loader;
    loader.LoadFromFile(path);

    const uint8_t* data = loader.GetTensorData("weight1");
    REQUIRE(data != nullptr);
    const float* fdata = reinterpret_cast<const float*>(data);
    CHECK(fdata[0] == doctest::Approx(1.0f));

    CHECK(loader.GetTensorData("no_key") == nullptr);

    std::remove(path.c_str());
    std::remove(tmpdir.c_str());
}

TEST_CASE("WeightLoader - GetReader accessor") {
    std::string tmpdir = TmpDir();
    Mkdir(tmpdir);
    std::string path = tmpdir + "/test.safetensors";
    CreateTestSafetensors(path);

    atb_llm::WeightLoader loader;
    loader.LoadFromFile(path);

    const atb_llm::SafetensorsReader& reader = loader.GetReader();
    CHECK(reader.NumTensors() == 4);
    CHECK(reader.HasKey("weight1"));

    std::remove(path.c_str());
    std::remove(tmpdir.c_str());
}

TEST_CASE("WeightLoader - CopyToNPU on nonexistent key") {
    // CopyToNPU requires TensorAllocator (NPU), so we test error path only
    std::string tmpdir = TmpDir();
    Mkdir(tmpdir);
    std::string path = tmpdir + "/test.safetensors";
    CreateTestSafetensors(path);

    atb_llm::WeightLoader loader;
    loader.LoadFromFile(path);

    // We can't test CopyToNPU without NPU, but we can verify
    // GetTensor returns proper error for missing keys
    atb_llm::WeightInfo info;
    CHECK(loader.GetTensor("nonexistent_tensor", info) == atb_llm::ERROR_WEIGHT_LOAD);

    std::remove(path.c_str());
    std::remove(tmpdir.c_str());
}

// ═══════════════════════════════════════════════════════════════════
// Qwen3VLConfig Tests
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Qwen3VLConfig - default values") {
    atb_llm::adapters::Qwen3VLConfig cfg;
    CHECK(cfg.image_token_id == 151655);
    CHECK(cfg.text_hidden_size == 2048);
    CHECK(cfg.text_num_heads == 16);
    CHECK(cfg.text_num_kv_heads == 8);
    CHECK(cfg.text_head_dim == 128);
    CHECK(cfg.text_intermediate_size == 6144);
    CHECK(cfg.text_num_layers == 28);
    CHECK(cfg.text_vocab_size == 151936);
    CHECK(cfg.vis_hidden_size == 1024);
    CHECK(cfg.vis_num_heads == 16);
    CHECK(cfg.vis_depth == 24);
    CHECK(cfg.vis_patch_size == 16);
    CHECK(cfg.normalize == true);
    CHECK(cfg.pp_patch_size == 16);
    CHECK(cfg.pp_min_pixels == 4096);
    CHECK(cfg.pp_max_pixels == 1310720);
}

TEST_CASE("Qwen3VLConfig - derived fields") {
    atb_llm::adapters::Qwen3VLConfig cfg;
    // vis_head_dim = vis_hidden_size / vis_num_heads = 1024 / 16 = 64
    CHECK(cfg.vis_head_dim() == 64);
    // num_grid = sqrt(vis_num_position_embeddings) = sqrt(2304) = 48
    CHECK(cfg.num_grid() == 48);
}

TEST_CASE("Qwen3VLConfig - LoadQwen3VLConfig nonexistent dir") {
    atb_llm::adapters::Qwen3VLConfig cfg;
    atb_llm::Status s = atb_llm::adapters::LoadQwen3VLConfig("/nonexistent/dir", cfg);
    CHECK(s == atb_llm::ERROR_FILE_NOT_FOUND);
}

TEST_CASE("Qwen3VLConfig - LoadQwen3VLConfig full config") {
    std::string tmpdir = TmpDir();
    Mkdir(tmpdir);

    // Write config.json
    std::string config_json = R"({
        "model_type": "qwen3vl_embedding",
        "image_token_id": 151655,
        "text_config": {
            "hidden_size": 1536,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 4096,
            "num_hidden_layers": 24,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1000000.0,
            "vocab_size": 151936,
            "rope_scaling": {
                "mrope_section": [16, 24, 24]
            }
        },
        "vision_config": {
            "hidden_size": 768,
            "num_heads": 12,
            "intermediate_size": 3072,
            "depth": 12,
            "in_channels": 3,
            "temporal_patch_size": 2,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "num_position_embeddings": 1024,
            "out_hidden_size": 1536,
            "deepstack_visual_indexes": [3, 7, 11],
            "initializer_range": 0.02
        }
    })";
    WriteFile(tmpdir + "/config.json", config_json);

    // Write preprocessor_config.json
    std::string pp_json = R"({
        "patch_size": 14,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "min_pixels": 3136,
        "max_pixels": 1000000,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225]
    })";
    WriteFile(tmpdir + "/preprocessor_config.json", pp_json);

    atb_llm::adapters::Qwen3VLConfig cfg;
    atb_llm::Status s = atb_llm::adapters::LoadQwen3VLConfig(tmpdir, cfg);
    CHECK(s == atb_llm::STATUS_OK);

    // Verify text config
    CHECK(cfg.text_hidden_size == 1536);
    CHECK(cfg.text_num_heads == 12);
    CHECK(cfg.text_num_kv_heads == 4);
    CHECK(cfg.text_head_dim == 128);
    CHECK(cfg.text_intermediate_size == 4096);
    CHECK(cfg.text_num_layers == 24);
    CHECK(cfg.text_vocab_size == 151936);
    CHECK(cfg.text_rms_norm_eps == doctest::Approx(1e-5f));
    CHECK(cfg.text_rope_theta == doctest::Approx(1000000.0f));
    CHECK(cfg.text_mrope_section.size() == 3);
    CHECK(cfg.text_mrope_section[0] == 16);
    CHECK(cfg.text_mrope_section[1] == 24);

    // Verify vision config
    CHECK(cfg.vis_hidden_size == 768);
    CHECK(cfg.vis_num_heads == 12);
    CHECK(cfg.vis_intermediate_size == 3072);
    CHECK(cfg.vis_depth == 12);
    CHECK(cfg.vis_patch_size == 14);
    CHECK(cfg.vis_spatial_merge_size == 2);
    CHECK(cfg.vis_num_position_embeddings == 1024);
    CHECK(cfg.vis_out_hidden_size == 1536);
    CHECK(cfg.vis_deepstack_visual_indexes.size() == 3);
    CHECK(cfg.vis_deepstack_visual_indexes[0] == 3);
    CHECK(cfg.vis_deepstack_visual_indexes[2] == 11);

    // Verify preprocessor config
    CHECK(cfg.pp_patch_size == 14);
    CHECK(cfg.pp_temporal_patch_size == 2);
    CHECK(cfg.pp_merge_size == 2);
    CHECK(cfg.pp_min_pixels == 3136);
    CHECK(cfg.pp_max_pixels == 1000000);
    CHECK(cfg.pp_image_mean.size() == 3);
    CHECK(cfg.pp_image_mean[0] == doctest::Approx(0.485f));
    CHECK(cfg.pp_image_std[2] == doctest::Approx(0.225f));

    // Cleanup
    std::remove((tmpdir + "/config.json").c_str());
    std::remove((tmpdir + "/preprocessor_config.json").c_str());
    std::remove(tmpdir.c_str());
}

TEST_CASE("Qwen3VLConfig - LoadQwen3VLConfig missing preprocessor") {
    std::string tmpdir = TmpDir();
    Mkdir(tmpdir);

    // Only config.json, no preprocessor_config.json
    std::string config_json = R"({
        "model_type": "qwen3vl_embedding",
        "text_config": {
            "hidden_size": 2048
        },
        "vision_config": {
            "hidden_size": 1024
        }
    })";
    WriteFile(tmpdir + "/config.json", config_json);

    atb_llm::adapters::Qwen3VLConfig cfg;
    atb_llm::Status s = atb_llm::adapters::LoadQwen3VLConfig(tmpdir, cfg);
    CHECK(s == atb_llm::STATUS_OK);

    // Preprocessor defaults should be used
    CHECK(cfg.pp_patch_size == 16);
    CHECK(cfg.pp_min_pixels == 4096);
    CHECK(cfg.pp_image_mean.size() == 3);
    CHECK(cfg.pp_image_mean[0] == doctest::Approx(0.5f));

    std::remove((tmpdir + "/config.json").c_str());
    std::remove(tmpdir.c_str());
}

TEST_CASE("Qwen3VLConfig - LoadQwen3VLConfig minimal config.json") {
    std::string tmpdir = TmpDir();
    Mkdir(tmpdir);

    // Minimal config.json — no text_config or vision_config sub-objects
    std::string config_json = R"({"model_type": "qwen3vl_embedding"})";
    WriteFile(tmpdir + "/config.json", config_json);

    atb_llm::adapters::Qwen3VLConfig cfg;
    atb_llm::Status s = atb_llm::adapters::LoadQwen3VLConfig(tmpdir, cfg);
    CHECK(s == atb_llm::STATUS_OK);

    // Defaults should remain
    CHECK(cfg.text_hidden_size == 2048);
    CHECK(cfg.vis_hidden_size == 1024);

    std::remove((tmpdir + "/config.json").c_str());
    std::remove(tmpdir.c_str());
}

TEST_CASE("Qwen3VLConfig - LoadQwen3VLConfig missing config.json") {
    std::string tmpdir = TmpDir();
    Mkdir(tmpdir);

    // No config.json at all
    atb_llm::adapters::Qwen3VLConfig cfg;
    atb_llm::Status s = atb_llm::adapters::LoadQwen3VLConfig(tmpdir, cfg);
    CHECK(s == atb_llm::ERROR_FILE_NOT_FOUND);

    std::remove(tmpdir.c_str());
}

// ═══════════════════════════════════════════════════════════════════
// Qwen3VLPreprocess Tests
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("SmartResize - already aligned, within limits") {
    int32_t new_h, new_w;
    // factor=32, 64x64 image, area=4096 within [min=4096, max=1310720]
    atb_llm::adapters::SmartResize(64, 64, 32, 4096, 1310720, new_h, new_w);
    CHECK(new_h == 64);
    CHECK(new_w == 64);
}

TEST_CASE("SmartResize - rounds to nearest factor") {
    int32_t new_h, new_w;
    // 100x100, factor=32: round(100/32)*32 = round(3.125)*32 = 3*32=96
    // area = 96*96 = 9216 > min=4096 and < max=1310720
    atb_llm::adapters::SmartResize(100, 100, 32, 4096, 1310720, new_h, new_w);
    CHECK(new_h % 32 == 0);
    CHECK(new_w % 32 == 0);
}

TEST_CASE("SmartResize - small image upscaled to min_pixels") {
    int32_t new_h, new_w;
    // 16x16, factor=32, min_pixels=4096
    // h_bar=32, w_bar=32, area=1024 < 4096 -> upscale
    atb_llm::adapters::SmartResize(16, 16, 32, 4096, 1310720, new_h, new_w);
    CHECK(new_h >= 32);
    CHECK(new_w >= 32);
    CHECK(new_h % 32 == 0);
    CHECK(new_w % 32 == 0);
    CHECK(static_cast<int64_t>(new_h) * new_w >= 4096);
}

TEST_CASE("SmartResize - large image downscaled to max_pixels") {
    int32_t new_h, new_w;
    // 2000x2000, factor=32, max_pixels=1310720
    // h_bar=2016, w_bar=2016, area=4064256 > 1310720 -> downscale
    atb_llm::adapters::SmartResize(2000, 2000, 32, 4096, 1310720, new_h, new_w);
    CHECK(new_h % 32 == 0);
    CHECK(new_w % 32 == 0);
    CHECK(static_cast<int64_t>(new_h) * new_w <= 1310720);
}

TEST_CASE("SmartResize - non-square image") {
    int32_t new_h, new_w;
    atb_llm::adapters::SmartResize(100, 200, 32, 4096, 1310720, new_h, new_w);
    CHECK(new_h % 32 == 0);
    CHECK(new_w % 32 == 0);
    // Aspect ratio should be roughly preserved
    float ratio_in = 200.0f / 100.0f;
    float ratio_out = static_cast<float>(new_w) / new_h;
    CHECK(ratio_out == doctest::Approx(ratio_in).epsilon(0.1));
}

TEST_CASE("BicubicResize - identity (same size)") {
    // 2x2 image, 1 channel, resize to 2x2
    uint8_t input[4] = {10, 20, 30, 40};
    float output[4] = {};

    atb_llm::adapters::BicubicResize(input, 2, 2, 1, 2, 2, output);
    CHECK(output[0] == doctest::Approx(10.0f));
    CHECK(output[1] == doctest::Approx(20.0f));
    CHECK(output[2] == doctest::Approx(30.0f));
    CHECK(output[3] == doctest::Approx(40.0f));
}

TEST_CASE("BicubicResize - 2x2 to 4x4 (integer upscale)") {
    // Expected values are the output of qwen3vl_preprocess.cpp::BicubicResize
    // itself — a Catmull-Rom cubic kernel (a = -0.5) with align_corners=False
    // source mapping `src = (dst + 0.5) * in/out - 0.5` and EDGE-CLAMP
    // boundary handling on the 4x4 neighbourhood.
    //
    // Reproduced in Python (NOT torch.nn.functional.interpolate — that
    // uses a different boundary convention and yields -59.46/+314.46 here):
    //   def cubic(x, a=-0.5):
    //       ax = abs(x)
    //       return ((a+2)*ax**3 - (a+3)*ax**2 + 1) if ax<=1 else
    //              (a*ax**3 - 5*a*ax**2 + 8*a*ax - 4*a) if ax<2 else 0
    //   # for each output pixel:
    //   src = (out + 0.5) * in_dim / out_dim - 0.5
    //   sum_over m,n in [-1..2]: cubic(d-m)*cubic(d-n) *
    //                            input[clamp(s+m,0,in-1), clamp(s+n,0,in-1)]
    //
    // For the 2x2 checkerboard the bicubic kernel overshoots — the all-zero
    // corners drop to -38.38 and the all-255 corners climb to 293.38. The
    // interior sample [5] (row 1, col 1; src coord ≈0.125 in 2x2 space)
    // resolves to 82.55.
    //
    // The Python reproduction lives in
    //   tests/python_reference/gen_cpu_reference.py::_cpp_bicubic_resize
    // and is end-to-end validated by
    //   tests/level1_cpu_pure/test_preprocess_cpu.cpp::TestBicubicVsPython
    //   (case "2x2_to_4x4").
    // Checkerboard pattern
    uint8_t input[4] = {0, 255, 255, 0};
    float output[16] = {};

    atb_llm::adapters::BicubicResize(input, 2, 2, 1, 4, 4, output);

    // Bicubic (align_corners=False) overshoots: negative at all-0 corners, >255 at all-255 corners
    CHECK(output[0] == doctest::Approx(-38.3807f));    // top-left (was 0)
    CHECK(output[3] == doctest::Approx(293.3807f));    // top-right (was 255)
    CHECK(output[12] == doctest::Approx(293.3807f));   // bottom-left (was 255)
    CHECK(output[15] == doctest::Approx(-38.3807f));   // bottom-right (was 0)

    // Center interpolation
    CHECK(output[5] == doctest::Approx(82.5513f));
}

TEST_CASE("BicubicResize - 3-channel") {
    // 2x2 image, 3 channels (RGB)
    uint8_t input[12] = {
        // Channel 0 (R)
        255,
        0,
        0,
        255,
        // Channel 1 (G)
        0,
        255,
        255,
        0,
        // Channel 2 (B)
        128,
        128,
        128,
        128,
    };
    float output[12] = {};

    atb_llm::adapters::BicubicResize(input, 2, 2, 3, 2, 2, output);

    // Channel 0
    CHECK(output[0] == doctest::Approx(255.0f));
    CHECK(output[1] == doctest::Approx(0.0f));
    // Channel 1
    CHECK(output[4] == doctest::Approx(0.0f));
    CHECK(output[5] == doctest::Approx(255.0f));
    // Channel 2: all 128
    CHECK(output[8] == doctest::Approx(128.0f));
    CHECK(output[11] == doctest::Approx(128.0f));
}

TEST_CASE("BicubicResize - downscale 4x4 to 2x2") {
    // Expected values are the output of qwen3vl_preprocess.cpp::BicubicResize
    // itself — a Catmull-Rom cubic kernel (a = -0.5) with align_corners=False
    // source mapping and edge-clamp boundary handling (NOT PyTorch's
    // F.interpolate, which uses different boundary weights). For this
    // linear gradient the 16-tap kernel resolves to 31.875 / 53.125 /
    // 116.875 / 138.125 — values reproduced by
    //   tests/python_reference/gen_cpu_reference.py::_cpp_bicubic_resize
    // and end-to-end validated in
    //   tests/level1_cpu_pure/test_preprocess_cpu.cpp::TestBicubicVsPython
    //   (case "4x4_to_2x2").
    uint8_t input[16] = {
        10, 20, 30, 40,
        50, 60, 70, 80,
        90, 100, 110, 120,
        130, 140, 150, 160};
    float output[4] = {};

    atb_llm::adapters::BicubicResize(input, 4, 4, 1, 2, 2, output);

    // Bicubic 16-neighbour interpolation (align_corners=False)
    CHECK(output[0] == doctest::Approx(31.875f));
    CHECK(output[1] == doctest::Approx(53.125f));
    CHECK(output[2] == doctest::Approx(116.875f));
    CHECK(output[3] == doctest::Approx(138.125f));
}

// ═══════════════════════════════════════════════════════════════════
// PreprocessImage Tests (full pipeline, CPU-only)
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("PreprocessImage - small image basic") {
    // Create a 32x32 image, 3 channels, uniform pixel value 128
    const int32_t H = 32, W = 32, C = 3;
    std::vector<uint8_t> image(C * H * W, 128);

    atb_llm::adapters::Qwen3VLConfig cfg;
    // Use default config: patch_size=16, tp=2, merge_size=2
    // factor = 16*2 = 32

    // Calculate expected output size
    int32_t new_h, new_w;
    atb_llm::adapters::SmartResize(H, W, cfg.pp_patch_size * cfg.pp_merge_size,
                                   cfg.pp_min_pixels, cfg.pp_max_pixels,
                                   new_h, new_w);
    int32_t grid_h = new_h / cfg.pp_patch_size;
    int32_t grid_w = new_w / cfg.pp_patch_size;
    int32_t grid_t = 1;  // single image
    int64_t num_patches = static_cast<int64_t>(grid_t) * grid_h * grid_w;
    int64_t patch_dim = static_cast<int64_t>(C) * cfg.pp_temporal_patch_size *
                        cfg.pp_patch_size * cfg.pp_patch_size;

    std::vector<uint16_t> pixel_values(num_patches * patch_dim, 0);
    int64_t out_num_patches = 0;
    int64_t grid_thw[3] = {};

    atb_llm::Status s = atb_llm::adapters::PreprocessImage(
        image.data(), C, H, W, cfg,
        pixel_values.data(), out_num_patches, grid_thw);

    CHECK(s == atb_llm::STATUS_OK);
    CHECK(out_num_patches == num_patches);
    CHECK(grid_thw[0] == 1);  // grid_t
    CHECK(grid_thw[1] == grid_h);
    CHECK(grid_thw[2] == grid_w);
    CHECK(num_patches > 0);

    // Check that pixel_values are non-zero (uniform input -> same normalized value)
    bool any_nonzero = false;
    for (int64_t i = 0; i < num_patches * patch_dim; i++) {
        if (pixel_values[i] != 0) {
            any_nonzero = true;
            break;
        }
    }
    CHECK(any_nonzero);
}

TEST_CASE("PreprocessImage - output shape consistency") {
    const int32_t H = 64, W = 96, C = 3;
    std::vector<uint8_t> image(C * H * W, 200);

    atb_llm::adapters::Qwen3VLConfig cfg;
    int32_t factor = cfg.pp_patch_size * cfg.pp_merge_size;

    int32_t new_h, new_w;
    atb_llm::adapters::SmartResize(H, W, factor,
                                   cfg.pp_min_pixels, cfg.pp_max_pixels,
                                   new_h, new_w);

    int64_t grid_h = new_h / cfg.pp_patch_size;
    int64_t grid_w = new_w / cfg.pp_patch_size;
    int64_t num_patches = grid_h * grid_w;  // grid_t = 1
    int64_t patch_dim = static_cast<int64_t>(C) * cfg.pp_temporal_patch_size *
                        cfg.pp_patch_size * cfg.pp_patch_size;

    std::vector<uint16_t> pixel_values(num_patches * patch_dim, 0);
    int64_t out_num_patches = 0;
    int64_t grid_thw[3] = {};

    atb_llm::Status s = atb_llm::adapters::PreprocessImage(
        image.data(), C, H, W, cfg,
        pixel_values.data(), out_num_patches, grid_thw);

    CHECK(s == atb_llm::STATUS_OK);
    CHECK(out_num_patches == num_patches);
    CHECK(grid_thw[1] == grid_h);
    CHECK(grid_thw[2] == grid_w);
}
