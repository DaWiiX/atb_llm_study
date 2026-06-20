#pragma once

/**
 * @brief Resolve the model checkpoint directory.
 *
 * The .env three-tier precedence ( @c std::getenv > @c .env file > default )
 * and the .env discovery / parsing machinery live in the production header
 * @c utils/dotenv.h, shared with @c Is310P()/@c Is910B() so that a bare
 * `./benchmark` invocation picks up @c .env the same way ctest does. This
 * test header now only adds the test-facing @c GetModelDir() convenience.
 *
 * Usage:
 * @code
 *   #include "test_env.h"
 *
 *   config.model_dir = GetModelDir();
 *   // or for arbitrary keys (via dotenv.h):
 *   std::string val = GetEnv("MY_VAR", "fallback");
 * @endcode
 */

#include "utils/dotenv.h"

#include <cstdio>
#include <string>

/**
 * Resolve the model checkpoint directory.
 *
 * Uses @c GetEnv("QWEN3VL_EMB_MODEL_DIR").  If the variable is absent from
 * both the shell environment and the @c .env file, prints a diagnostic to
 * stderr and returns an empty string — the caller is responsible for
 * validation (e.g. printing a LOG_ERROR and exiting cleanly).
 */
inline std::string GetModelDir() {
    std::string val = GetEnv("QWEN3VL_EMB_MODEL_DIR");
    if (!val.empty())
        return val;

    std::fprintf(stderr,
        "[test_env] QWEN3VL_EMB_MODEL_DIR is not set.\n"
        "           Add it to <repo>/.env (see .env.example) and re-run via\n"
        "           build_and_test.sh, or `export` it before invoking the\n"
        "           test binary directly.\n");
    return std::string();
}
