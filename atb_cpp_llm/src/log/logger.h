#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <ctime>

namespace atb_llm {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
    NONE = 4
};

namespace detail {

inline LogLevel& MutableLogLevel() {
    static LogLevel level = []() {
        const char* env = std::getenv("LOG_LEVEL");
        if (!env) return LogLevel::WARN;
        int val = std::atoi(env);
        if (val < 0) return LogLevel::DEBUG;
        if (val > 3) return LogLevel::NONE;
        return static_cast<LogLevel>(val);
    }();
    return level;
}

inline LogLevel GetLogLevel() {
    return MutableLogLevel();
}

inline void SetLogLevel(LogLevel level) {
    MutableLogLevel() = level;
}

inline const char* LogLevelStr(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERROR: return "ERROR";
        default:              return "?????";
    }
}

inline void LogMessage(LogLevel level, const char* file, int line, const char* fmt, ...) {
    if (level < GetLogLevel()) return;

    // Extract filename only
    const char* filename = file;
    const char* p = file;
    while (*p) {
        if (*p == '/' || *p == '\\') filename = p + 1;
        ++p;
    }

    // Timestamp
    time_t now = time(nullptr);
    struct tm tm_buf;
    localtime_r(&now, &tm_buf);
    char time_str[32];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm_buf);

    // Format user message into buffer first to avoid interleaved output
    char msg_buf[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg_buf, sizeof(msg_buf), fmt, args);
    va_end(args);

    // Single fprintf call — atomic from other threads' perspective
    fprintf(stderr, "[%s][%s][%s:%d] %s\n", time_str, LogLevelStr(level), filename, line, msg_buf);
    fflush(stderr);
}

} // namespace detail
} // namespace atb_llm

// ── Logging macros ───────────────────────────────────────
#define LOG_DEBUG(fmt, ...) \
    atb_llm::detail::LogMessage(atb_llm::LogLevel::DEBUG, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) \
    atb_llm::detail::LogMessage(atb_llm::LogLevel::INFO, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) \
    atb_llm::detail::LogMessage(atb_llm::LogLevel::WARN, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) \
    atb_llm::detail::LogMessage(atb_llm::LogLevel::ERROR, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
