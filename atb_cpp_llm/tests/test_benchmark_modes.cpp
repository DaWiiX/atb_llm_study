/**
 * Smoke test for the benchmark binary's output format.
 *
 * Spawns the benchmark subprocess, runs `--mode text`, and parses the
 * combined stdout+stderr to verify that:
 *   - the machine-readable `BENCH_RESULT:` line is emitted with all
 *     expected fields (mode, resolution, S, vis, e2e_mean), and
 *   - the human-readable stage table (ReportStages via LOG_INFO) is
 *     present, including the "Stage" header, "Preprocess" and "E2E" rows.
 *
 * Prevents regression of §9.8 (benchmark not in CTest, output quality
 * never validated).
 *
 * Requires NPU + full model checkpoint (QWEN3VL_EMB_MODEL_DIR).
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cstdio>
#include <string>
#include <vector>

// Capture a benchmark subprocess's combined stdout+stderr.
// `args` is appended after the binary name, e.g. "--mode text --iter 1".
// Uses "./benchmark" (not "benchmark") because the shell does not search the
// current directory by default; CTest runs this test with build/ as cwd,
// where the benchmark binary lives.
static std::string capture_benchmark(const std::string& args) {
    std::string cmd = "./benchmark " + args + " 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    std::string output;
    char buf[512];
    while (fgets(buf, sizeof(buf), pipe)) {
        output += buf;
    }
    pclose(pipe);
    return output;
}

TEST_CASE("benchmark --mode text 输出格式") {
    std::string out = capture_benchmark("--mode text --iter 1 --warmup 0");

    // 1. Exit code is indirect: on failure benchmark still emits errors,
    //    so non-empty output is a weak liveness signal.
    CHECK_FALSE(out.empty());

    // 2. Machine-readable line present.
    CHECK(out.find("BENCH_RESULT:") != std::string::npos);

    // 3. Machine line fields complete: mode=text resolution=N/A S= vis=
    CHECK(out.find("mode=text") != std::string::npos);
    CHECK(out.find("resolution=N/A") != std::string::npos);
    CHECK(out.find("S=") != std::string::npos);
    CHECK(out.find("vis=") != std::string::npos);
    CHECK(out.find("e2e_mean=") != std::string::npos);

    // 4. Human-readable stage table (ReportStages uses LOG_INFO → stderr,
    //    merged into stdout by 2>&1). Check for header + key stage rows.
    CHECK(out.find("Stage") != std::string::npos);
    CHECK(out.find("Preprocess") != std::string::npos);
    CHECK(out.find("E2E") != std::string::npos);
}
