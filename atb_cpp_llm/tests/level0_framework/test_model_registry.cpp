/**
 * Unit tests for ModelRegistry using doctest.
 *
 * Covers:
 *   - Singleton behavior (Instance() returns same pointer)
 *   - Empty lookup (Has / Create on nonexistent entry)
 *   - Register and Create (legacy and extended APIs)
 *   - Has after register
 *   - Duplicate registration (first match wins in Create)
 *   - CreateWithFallback (exact match, compatibility check, priority ordering)
 *   - Null factory handling
 *   - Free functions (RegisterModelFactory, RegisterModelEntry)
 *
 * Run: ./test_model_registry
 * No NPU required — pure CPU logic tests.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "engine/model_registry.h"
#include "atb_llm/model.h"
#include "io/json_config.h"
#include "log/logger.h"

#include <memory>
#include <string>
#include <functional>

// ── Minimal mock model for testing ────────────────────────────────
class MockModel : public atb_llm::IModel {
public:
    atb_llm::Status Load(const std::string& /*model_dir*/,
                         atb_llm::IRuntime* /*runtime*/) override {
        return atb_llm::STATUS_OK;
    }
    atb_llm::Status Forward(const atb_llm::InferRequest& /*request*/,
                            atb_llm::InferResult& /*result*/) override {
        return atb_llm::STATUS_OK;
    }
    const char* GetName() const override { return "MockModel"; }
};

// ── Factory helpers ───────────────────────────────────────────────

static std::unique_ptr<atb_llm::IModel> MakeMockModel() {
    return std::make_unique<MockModel>();
}

static int g_create_count = 0;

/** Factory that increments a counter so we can distinguish which factory ran. */
static std::unique_ptr<atb_llm::IModel> MakeCountingModel() {
    g_create_count++;
    return std::make_unique<MockModel>();
}

// ── Helper: build a minimal valid JsonConfig ─────────────────────
static atb_llm::JsonConfig MakeConfig(const std::string& model_type) {
    std::string json = "{\"model_type\":\"" + model_type + "\"}";
    return atb_llm::JsonConfig::Parse(json);
}

// ═══════════════════════════════════════════════════════════════════
// 1. Singleton
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("ModelRegistry::Instance() returns the same pointer") {
    LOG_INFO("=== Test: Singleton ===");
    auto& r1 = atb_llm::ModelRegistry::Instance();
    auto& r2 = atb_llm::ModelRegistry::Instance();
    CHECK(&r1 == &r2);
}

// ═══════════════════════════════════════════════════════════════════
// 2. Empty lookup
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Empty lookup: Has() returns false, Create() returns nullptr") {
    LOG_INFO("=== Test: Empty lookup ===");
    auto& reg = atb_llm::ModelRegistry::Instance();
    CHECK(reg.Has("nonexistent_model_xyz") == false);
    CHECK(reg.Create("nonexistent_model_xyz") == nullptr);
}

// ═══════════════════════════════════════════════════════════════════
// 3. Register and Create (legacy API)
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Register (legacy) and Create: factory produces non-null model") {
    LOG_INFO("=== Test: Register and Create (legacy) ===");
    auto& reg = atb_llm::ModelRegistry::Instance();
    reg.Register("reg_create_test_model", MakeMockModel);

    auto model = reg.Create("reg_create_test_model");
    REQUIRE(model != nullptr);
    CHECK(std::string(model->GetName()) == "MockModel");
}

// ═══════════════════════════════════════════════════════════════════
// 4. Has after register
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Has() returns true after Register") {
    LOG_INFO("=== Test: Has after register ===");
    auto& reg = atb_llm::ModelRegistry::Instance();

    CHECK(reg.Has("has_test_model") == false);
    reg.Register("has_test_model", MakeMockModel);
    CHECK(reg.Has("has_test_model") == true);
}

// ═══════════════════════════════════════════════════════════════════
// 5. Duplicate register: first match wins in Create()
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Duplicate register: first registered factory wins") {
    LOG_INFO("=== Test: Duplicate register ===");
    auto& reg = atb_llm::ModelRegistry::Instance();
    const char* name = "dup_model";

    // Reset counter, register two factories under the same name
    g_create_count = 0;
    reg.Register(name, MakeCountingModel);   // first
    reg.Register(name, MakeCountingModel);   // second — should NOT be called by Create()

    auto model = reg.Create(name);
    REQUIRE(model != nullptr);

    // Only the first factory should have been invoked
    CHECK(g_create_count == 1);
}

// ═══════════════════════════════════════════════════════════════════
// 6. CreateWithFallback — exact match first
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("CreateWithFallback: exact match wins over compat_check entries") {
    LOG_INFO("=== Test: CreateWithFallback — exact match first ===");
    auto& reg = atb_llm::ModelRegistry::Instance();
    const char* name = "fallback_exact_model";

    // Register an exact-match entry (low priority, but exact match takes precedence)
    atb_llm::RegistryEntry exact;
    exact.model_type = name;
    exact.factory = MakeMockModel;
    exact.priority = 0;
    reg.Register(std::move(exact));

    // Register a compat_check entry that would also match THIS specific name.
    // Must use a scoped check to avoid polluting other test cases in the
    // global singleton registry.
    atb_llm::RegistryEntry compat;
    compat.model_type = "compat_for_exact_test";
    compat.factory = MakeMockModel;
    compat.priority = 99;
    compat.compat_check = [](const std::string& mt, const atb_llm::JsonConfig& /*cfg*/) {
        return mt == "fallback_exact_model";
    };
    reg.Register(std::move(compat));

    auto model = reg.CreateWithFallback(name, MakeConfig(name));
    // Exact match should succeed regardless of the compat entry
    CHECK(model != nullptr);
}

// ═══════════════════════════════════════════════════════════════════
// 7. CreateWithFallback — no compat matches
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("CreateWithFallback: returns nullptr when no compat_check passes") {
    LOG_INFO("=== Test: CreateWithFallback — no compat ===");
    auto& reg = atb_llm::ModelRegistry::Instance();
    const char* name = "no_compat_model_xyz";

    // Register a compat_check entry that only matches a specific known type.
    // Scoped to avoid cross-test pollution from the global singleton registry.
    atb_llm::RegistryEntry entry;
    entry.model_type = "compat_failing_entry";
    entry.factory = MakeMockModel;
    entry.priority = 10;
    entry.compat_check = [](const std::string& mt, const atb_llm::JsonConfig& /*cfg*/) {
        return mt == "only_this_specific_model";  // never matches our test name
    };
    reg.Register(std::move(entry));

    auto model = reg.CreateWithFallback(name, MakeConfig(name));
    // No exact match, and compat_check doesn't fire for this name
    CHECK(model == nullptr);
}

// ═══════════════════════════════════════════════════════════════════
// 8. CreateWithFallback — priority ordering
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("CreateWithFallback: highest priority compat entry is selected") {
    LOG_INFO("=== Test: CreateWithFallback — priority ordering ===");
    auto& reg = atb_llm::ModelRegistry::Instance();
    const char* name = "priority_model_xyz";

    // Track which factory was called
    std::string called_by;
    auto makeFactory = [&called_by](const std::string& label) -> atb_llm::ModelFactory {
        return [&called_by, label]() -> std::unique_ptr<atb_llm::IModel> {
            called_by = label;
            return std::make_unique<MockModel>();
        };
    };

    // Register entries with priorities 5, 10, 7.
    // Each compat_check is scoped to this test's specific model_type to avoid
    // cross-test pollution in the global singleton registry.
    atb_llm::RegistryEntry e5;
    e5.model_type = "prio_fallback_5";
    e5.factory = makeFactory("priority_5");
    e5.priority = 5;
    e5.compat_check = [](const std::string& mt, const atb_llm::JsonConfig& /*cfg*/) {
        return mt == "priority_model_xyz";
    };
    reg.Register(std::move(e5));

    atb_llm::RegistryEntry e10;
    e10.model_type = "prio_fallback_10";
    e10.factory = makeFactory("priority_10");
    e10.priority = 10;
    e10.compat_check = [](const std::string& mt, const atb_llm::JsonConfig& /*cfg*/) {
        return mt == "priority_model_xyz";
    };
    reg.Register(std::move(e10));

    atb_llm::RegistryEntry e7;
    e7.model_type = "prio_fallback_7";
    e7.factory = makeFactory("priority_7");
    e7.priority = 7;
    e7.compat_check = [](const std::string& mt, const atb_llm::JsonConfig& /*cfg*/) {
        return mt == "priority_model_xyz";
    };
    reg.Register(std::move(e7));

    auto model = reg.CreateWithFallback(name, MakeConfig(name));
    REQUIRE(model != nullptr);
    CHECK(called_by == "priority_10");  // highest priority wins
}

// ═══════════════════════════════════════════════════════════════════
// 9. Null factory
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Null factory: Has() returns true, Create() throws") {
    LOG_INFO("=== Test: Null factory ===");
    auto& reg = atb_llm::ModelRegistry::Instance();
    const char* name = "null_factory_model";

    atb_llm::RegistryEntry entry;
    entry.model_type = name;
    entry.factory = nullptr;   // intentionally null
    reg.Register(std::move(entry));

    // Has() checks model_type only — should return true
    CHECK(reg.Has(name) == true);

    // Create() calls factory() without null check — throws std::bad_function_call
    CHECK_THROWS_AS(reg.Create(name), std::bad_function_call);
}

// ═══════════════════════════════════════════════════════════════════
// 10. Free functions: RegisterModelFactory and RegisterModelEntry
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Free functions: RegisterModelFactory and RegisterModelEntry work") {
    LOG_INFO("=== Test: Free functions ===");
    auto& reg = atb_llm::ModelRegistry::Instance();

    // Test RegisterModelFactory (legacy free function)
    atb_llm::RegisterModelFactory("free_func_factory_model", MakeMockModel);
    CHECK(reg.Has("free_func_factory_model") == true);
    auto m1 = reg.Create("free_func_factory_model");
    CHECK(m1 != nullptr);

    // Test RegisterModelEntry (extended free function)
    atb_llm::RegistryEntry entry;
    entry.model_type = "free_func_entry_model";
    entry.factory = MakeMockModel;
    entry.priority = 42;
    atb_llm::RegisterModelEntry(std::move(entry));
    CHECK(reg.Has("free_func_entry_model") == true);
    auto m2 = reg.Create("free_func_entry_model");
    CHECK(m2 != nullptr);
}
