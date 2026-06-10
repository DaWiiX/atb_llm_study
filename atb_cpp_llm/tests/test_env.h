#pragma once

/**
 * @brief Resolve the model checkpoint directory.
 *
 * Reads @c QWEN3VL_EMB_MODEL_DIR from the environment. @c build_and_test.sh
 * loads `.env` at the repo root before invoking CTest, so this env var is
 * always set in the normal workflow. If a test binary is invoked directly
 * without that bootstrap, the call aborts with a clear error pointing at
 * `.env` — better than silently using a stale hard-coded path.
 *
 * Usage:
 * @code
 *   #include "test_env.h"
 *   config.model_dir = GetModelDir();
 * @endcode
 */

#include <cstdlib>
#include <cstdio>
#include <cstdlib>
#include <string>

inline std::string GetModelDir() {
    const char* env = std::getenv("QWEN3VL_EMB_MODEL_DIR");
    if (env && env[0] != '\0')
        return env;
    std::fprintf(stderr,
        "[test_env] QWEN3VL_EMB_MODEL_DIR is not set.\n"
        "           Add it to <repo>/.env (see .env.example) and re-run via\n"
        "           build_and_test.sh, or `export` it before invoking the\n"
        "           test binary directly.\n");
    std::abort();
}
