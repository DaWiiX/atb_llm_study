#pragma once

/**
 * @brief Resolve the model checkpoint directory and other config values.
 *
 * Mirrors the Python @c env.py three-tier precedence:
 *   @c std::getenv(name) > @c .env file > default
 *
 * The @c .env file is auto-discovered by walking up from the working
 * directory (and falling back to @c /proc/self/exe), then parsed once into
 * a static cache.  Parsing is intentionally boring — @c KEY=VALUE per line,
 * @c # comments, optional surrounding quotes, no interpolation.
 *
 * Usage:
 * @code
 *   #include "test_env.h"
 *
 *   config.model_dir = GetModelDir();
 *   // or for arbitrary keys:
 *   std::string val = GetEnv("MY_VAR", "fallback");
 * @endcode
 */

#include <cstdlib>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <fstream>
#include <unistd.h>

// ── .env discovery and parsing ──────────────────────────────────────────

static const char* kDotEnvFilename = ".env";
static const int   kSearchDepth     = 6;

/* Return true when @p path exists and is readable. */
inline bool FileExists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

/* Strip leading / trailing whitespace and carriage-return. */
inline std::string Trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && (s[start] == ' ' || s[start] == '\t' || s[start] == '\r'))
        ++start;
    size_t end = s.size();
    while (end > start && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\r'))
        --end;
    return s.substr(start, end - start);
}

/* Remove matching surrounding single or double quotes. */
inline std::string StripQuotes(const std::string& s) {
    if (s.size() >= 2 && s.front() == s.back() && (s.front() == '"' || s.front() == '\''))
        return s.substr(1, s.size() - 2);
    return s;
}

/**
 * Locate the nearest @c .env file.
 *
 * 1. @c ATB_DOTENV_PATH env-var override (exact path).
 * 2. Walk up from the current working directory (at most @c kSearchDepth levels).
 * 3. Walk up from the directory containing @c /proc/self/exe.
 *
 * Returns the absolute path or an empty string when no file is found.
 */
inline std::string FindDotEnv() {
    // 1. Override via environment variable
    const char* override = std::getenv("ATB_DOTENV_PATH");
    if (override && override[0] != '\0') {
        if (FileExists(override))
            return std::string(override);
        return "";
    }

    // 2. Walk up from CWD
    char cwd_buf[4096];
    if (::getcwd(cwd_buf, sizeof(cwd_buf)) != nullptr) {
        std::string dir(cwd_buf);
        for (int i = 0; i < kSearchDepth; ++i) {
            std::string candidate = dir + "/" + kDotEnvFilename;
            if (FileExists(candidate))
                return candidate;
            size_t pos = dir.rfind('/');
            if (pos == std::string::npos || pos == 0)
                break;
            dir = dir.substr(0, pos);
        }
    }

    // 3. Walk up from the executable's directory
    char exe_buf[4096];
    ssize_t len = ::readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
    if (len != -1) {
        exe_buf[len] = '\0';
        std::string exe_path(exe_buf);
        size_t last_slash = exe_path.rfind('/');
        if (last_slash != std::string::npos) {
            std::string exe_dir = exe_path.substr(0, last_slash);
            for (int i = 0; i < kSearchDepth; ++i) {
                std::string candidate = exe_dir + "/" + kDotEnvFilename;
                if (FileExists(candidate))
                    return candidate;
                size_t pos = exe_dir.rfind('/');
                if (pos == std::string::npos || pos == 0)
                    break;
                exe_dir = exe_dir.substr(0, pos);
            }
        }
    }

    return "";
}

/**
 * Parse a @c .env file into a @c {key: value} map.
 *
 * Lines are parsed as @c KEY=VALUE.  Empty lines and lines starting with
 * @c # are skipped.  Surrounding single or double quotes are stripped from
 * values.  No interpolation or @c export keywords are supported — keep it
 * simple.
 */
inline std::unordered_map<std::string, std::string> ReadDotEnv(const std::string& path) {
    std::unordered_map<std::string, std::string> out;
    if (path.empty())
        return out;

    std::ifstream file(path.c_str());
    if (!file.is_open())
        return out;

    std::string line;
    while (std::getline(file, line)) {
        // Strip trailing carriage-return (Windows line-endings on Linux)
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        std::string trimmed = Trim(line);
        if (trimmed.empty() || trimmed[0] == '#')
            continue;

        size_t eq = trimmed.find('=');
        if (eq == std::string::npos)
            continue;

        std::string key   = Trim(trimmed.substr(0, eq));
        std::string value = Trim(trimmed.substr(eq + 1));
        value = StripQuotes(value);

        if (!key.empty())
            out[key] = value;
    }
    return out;
}

/**
 * Return the parsed @c .env contents, lazy-loaded and cached.
 *
 * The file is discovered once via @c FindDotEnv() on first call; subsequent
 * calls return the cached map.  Thread-safe under C++11.
 */
inline const std::unordered_map<std::string, std::string>& GetDotEnvCache() {
    static const std::unordered_map<std::string, std::string> cache = []() {
        std::string dotenv_path = FindDotEnv();
        return ReadDotEnv(dotenv_path);
    }();
    return cache;
}

// ── Public API ──────────────────────────────────────────────────────────

/**
 * Resolve a configuration value with three-tier precedence:
 *   @c std::getenv(name)  >  @c .env file  >  @p default_value
 *
 * @param name          Environment-variable / dotenv key.
 * @param default_value  Fallback returned when neither source defines the key.
 *                       Pass @c nullptr (the default) to get an empty string.
 * @return The resolved value (never @c nullptr — always a @c std::string).
 */
inline std::string GetEnv(const char* name, const char* default_value = nullptr) {
    // 1. Shell environment (highest priority)
    const char* env = std::getenv(name);
    if (env && env[0] != '\0')
        return std::string(env);

    // 2. Cached .env file
    const auto& cache = GetDotEnvCache();
    auto it = cache.find(name);
    if (it != cache.end())
        return it->second;

    // 3. Caller-supplied default
    if (default_value != nullptr)
        return std::string(default_value);
    return std::string();
}

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
