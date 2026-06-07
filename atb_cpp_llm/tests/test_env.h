#pragma once

/**
 * @brief Resolve the model checkpoint directory.
 *
 * Checks the environment variable @c QWEN3VL_EMB_MODEL_DIR first; falls back
 * to the hard-coded default path.  Callers include this header once and use
 * @c GetModelDir() wherever a model directory string is required.
 *
 * Usage:
 * @code
 *   #include "test_env.h"
 *   config.model_dir = GetModelDir();
 * @endcode
 */

#include <cstdlib>
#include <string>

inline std::string GetModelDir() {
    const char* env = std::getenv("QWEN3VL_EMB_MODEL_DIR");
    if (env && env[0] != '\0')
        return env;
    return "/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B";
}