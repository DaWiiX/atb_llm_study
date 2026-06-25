# 踩坑经验集（Lessons Learned）

> **按主题组织，不按时间。** 做事前按主题检索，避免同类坑复发。每条带「触发关键词」——当你脑海中冒出这些词，先查这里。
>
> 维护规则：踩了新坑就归并进对应主题（没有就加主题），同主题只留最强的那条；检查是否有更弱的旧条目可替换。跨会话需主动 recall 的规则同步到 `.claude/.../memory/`。
>
> 来源标记：`refactor §4.x` = refactoring-plan §4；`arch §9.x` = architect-assessment §9；`audit` = audit-fix-plan；`310p` = platform-310p。

---

## 主题 1：精度调试

**触发关键词**：cosine 低、NaN、精度下降、block 输出异常、fp16 二进制、跨语言数值、grid_t、actual_patches、隐式领域事实、单图多帧、temporal 折入

1. **跨语言 fp16 二进制必须确认位解释一致**
   `np.frombuffer(raw, dtype=np.float16)` 正确；`struct.unpack` + `dtype=np.uint16` 再 `.astype(np.float16)` 是错的（那是数值 cast 不是比特重解释）。— `refactor §4.1`

2. **精度下降第一步永远是 `git stash` 回基线**
   基线通过→问题在新代码；基线不过→问题在原有代码。不要边改边猜。— `refactor §4.9`

3. **NPU sync 不是"加了更安全"**
   H2D/D2H 已自带同步，额外 `Synchronize()` 会破坏异步流水导致 cosine 断崖式下降。`ASCEND_LAUNCH_BLOCKING=1` 不保证 byte-exact 确定性。— `refactor §4.10`

4. **同一功能的双路径必然分叉**
   性能优化后必须删旧路径，让 `Forward` 成为 `ForwardWithTiming` 的薄 wrapper，benchmark compare 确认精度无损后立即删。留着两条路，改 bug 只动一边。— `refactor §4.3`

5. **多框架 benchmark 必须用完全相同输入**
   C++ 用裸 token、Python 走 chat template → cosine 只有 0.2–0.3。统一 token 生成脚本，所有框架加载同一批 `.bin`。— `refactor §4.7`

6. **【2026-06-23 benchmark路径C Reviewer】隐式领域事实（如 grid_t=1）被多个口径隐式依赖，改动逼其显式化时要警惕视频/多帧场景**
   Qwen3VL 单图推理：`temporal_patch_size` 折入 `patch_dim`，实际 `grid_t=1`（`actual_patches = grid_h*grid_w`）。但 benchmark 既有代码用 `grid_t=2` 算 `num_patches`（仅作 host buffer 2× 超分配），靠 `PreprocessImage` 回填 `actual_patches` 隐式修正 `vis_tokens`。路径 C 不调 `PreprocessImage`，被迫显式 `actual_patches = grid_h*grid_w`——正确，但 `vis_tokens` 现与 PreprocessImage 回填解耦、无运行时一致性校验。**当前单图 grid_t=1 恒成立无 bug；但视频/多帧场景 grid_t=2 时会静默错算 vis_tokens（input_ids 与引擎 patch 数不匹配）。** 教训：单图→多帧适配时，grep 所有 `grid_t`/`actual_patches`/`num_patches` 用法，确认每个是否假设 grid_t=1；buffer 超分配用的 grid_t 与语义用的 grid_t 不能混。— benchmark 路径C Reviewer MINOR

---

## 主题 2：平台差异（910B / 310P）

**触发关键词**：310P、910B、NZ mask、FRACTAL_NZ、TransdataOperation、setup fail、status 4、ASCEND_PLATFORM、GQA

1. **910B 模拟 310P 不可信，必须真机验证**
   GQA 在 910B 模拟 310P 模式失败，真 310P 完全正常。— `310p`

2. **310P mask 必须 NZ 格式 + FRACTAL_NZ format tag（二者缺一不可）**
   PA_ENCODER 下 910B 用 ND、310P 用 NZ。数据 layout 是 NZ 还不够，NPU tensor 的 format tag 也必须是 `ACL_FORMAT_FRACTAL_NZ`(29)，否则 ATB 内部 ND→NZ Transdata 失败："call operation setup fail"。— `310p` / `arch §9.6`

3. **C++ 和 Python 平台逻辑必须同步（含 fallback 链）**
   C++ 改了 NZ mask 转换，Python `engine.py` 也必须改，不一致会一方通过一方失败。**同理适用于容错机制**：C++ 参考生成器有 ATB→transformers-CPU fallback（`test_stage_reference.py`），但 Python 测试参考侧 `load_tf_ref` 长期没有——910B 上不暴露，到 310P transformers 参考撞 ArgMaxWithValue/Conv3d 就硬 FAIL。一侧加的容错/退化逻辑必须同步到对侧，否则跨平台测试必然暴露缺口。— `310p` / 2026-06-18 fallback 事件

4. **`ASCEND_PLATFORM` 配错会静默 fallback**
   `is_310p()` 依赖 `.env` 的 `ASCEND_PLATFORM=310P`；配错则返回 False，静默用 910B 的 ND mask，310P 上图编译失败。正确行为隐式依赖人工配置——配错不报错最危险。— `arch §9.6`

   **根治（2026-06-20）**：根因比"配错"更深——`Is310P()` 只读 `std::getenv` **不读 `.env`**，而 ctest 单测靠 CMake `ENVIRONMENT` + `build_and_test.sh` source .env 把 `ASCEND_PLATFORM` 烤进进程环境才过；裸启动 `./benchmark` 读不到 `.env` → `Is310P()=false` → 310P 撞 `TransdataOperation ... inDims is not support`。修复：把 `.env` 三级解析抽到 `src/utils/dotenv.h`，`Is310P()`/`Is910B()` 改用 `GetEnv`（getenv > .env > 默认）；并加 `aclrtGetSocName()` 启动期硬件探针自检，配置与硬件不符 `LOG_ERROR` 提醒（不静默）。教训：**配置读取入口必须全代码库统一**（生产代码与 test 不能各读各的），否则"单测过、裸二进制挂"的静默缺口必然在跨平台暴露。

5. **GQA 在 310P 原生支持，无需 Is310P() 守卫**
   实测 cos=1.0。早期文档说"310P 不支持 GQA"是错的，guard 已全部移除。— `310p`

6. **AllocNpuFloat16 默认 ND 格式**
   310P 上创建 mask tensor 必须显式设 `format = ACL_FORMAT_FRACTAL_NZ`。— `310p`

---

## 主题 3：二进制/脚本发布前验证

**触发关键词**：对外二进制、benchmark、CLI mode、冒烟测试、编译通过、输出可读性、配置漂移

1. **【核心，已复发两次】未注册自动化测试的二进制，提交前必须冒烟每个 mode/flag**
   benchmark 不在 CTest → 所有 mode 输出从未被验证 → §9.6 静默崩溃、§9.8 输出丑陋+分辨率漂移，两次都是 PR 后用户实测才发现。**编译通过 ≠ 功能正确**，只验证语法不验证输出语义。对外可执行二进制提交前在真实环境跑每个 mode 组合，人工确认输出（不只看退出码）。— `arch §9.6 + §9.8`（已合并，原两条同型复发）

2. **配置项保持单一数据源，避免多处硬编码漂移**
   `--mode bench` 和 `--mode compare` 各自硬编码 resolutions 数组，时间久了必然不一致。提取为文件级 `kBenchmarkResolutions[]` 共用，比测试断言更彻底。— `arch §9.8 (M5)`

3. **静态审查 + 编译验证只覆盖语法层**
   输出质量（可读性、数值正确性、配置一致性）必须运行时验证。reviewer 不会主动运行 `--mode compare` 看输出长什么样。— `arch §9.8`

4. **静态全局变量 `std::abort()` 会杀死进程在 main() 之前**
   `GetModelDir()` 在静态初始化阶段 abort，所有 LOG 宏从未执行 → 静默退出无日志。环境依赖不能放在静态初始化里。— `arch §9.6`

5. **"测试通过"的语义会膨胀**
   `run_all.py --benchmarks` 退出码 0 被解读为"benchmark 功能正常"，实际只验证了默认 `--mode mm` 路径。测试通过 ≠ 功能完整。— `arch §9.6`

---

## 主题 4：测试反模式

**触发关键词**：静态审查、运行时测试、测试覆盖、refdata、跳过、阈值、假阳性、参数对齐、引擎真实参数、smart_resize factor、AA 降采样、抗锯齿守卫、恒等尺度、参考链真伪、bit-exact probe、unbound 方法、stand-in、精度 margin、输入分布、worst-case、自然图 vs 噪声、token 同源、chat template、official full gate、self-consistency、diagnostic、testing guide

1. **静态审查 ≠ 运行时测试**
   API 行为误解、运行时竞态、性能回归、兼容性、数值精度——这五类只有实际运行才能发现。审查通过 = 静态审查零问题 + 单元测试 PASS + 复现确认问题消失。— `arch §9.4`

2. **needs_refdata 标签必须完整（已复发 2 次）**
   `test_io_adapters` 读 `preprocess_*.bin` 但不在 needs_refdata 列表 → `--no-refdata` 时静默跳过精度验证，CTest 仍报 PASS（假阳性）。新增读 `/tmp/*.bin` 的测试必须登记到 CMakeLists REFDATA_DEPENDENT_TESTS。— `audit C1`。**复发 1**（P10-B spike）：`test_aclnn_bicubic_spike` 同样漏登，被 Reviewer BLOCKER-2 抓出。**复发 2** 提醒：这是高频坑，新增任何读 `/tmp/*.bin` 参考的测试，第一反应就是 grep REFDATA_DEPENDENT_TESTS 确认已登记。— `audit C1 + P10-B spike BLOCKER-2`

3. **参考实现不能自证**
   BicubicResize 用自己的实现当参考 = 没有独立参考。参考实现必须是同输入下的黄金标准（本项目为 transformers）。— `test-fix-plan P7`

4. **测试阈值不能放松来"通过"**
   C++ 和 Python 相同输入 cosine < 0.99 就是 bug，定位根因，不放宽阈值（CLAUDE.md 测试精度原则）。— 项目原则

5. **重复测试要合并**
   `test_consistency.cpp` 是 `test_accuracy.cpp` 子集、多个 NZ 验证脚本冗余——删除子集，保留最全的。— `test-fix-plan P9/P11`

6. **【2026-06-23 P10-B spike BLOCKER-1】测试用例的参数必须对齐引擎真实参数，自己编的参数会让闸口结论失效**
   spike gen 脚本 smart_resize 用 `factor=28`，引擎实际 `factor=32`（patch_size=16×merge_size=2）、`min_pixels=4096`、`max_pixels=1310720`（来自 `preprocessor_config.json`）。结果 1080×1920 在 spike 里是 1092×1932（近恒等），引擎实际 832×1504（大幅降采样）——**在错误尺寸上验证的 cos=0.999 全是假象**，纠正参数后 cos 立刻翻成 0.987/0.958 闸口失败。**任何验证引擎某阶段的测试，其输入参数（resize factor/像素上下限/dtype/shape）必须从引擎实际读取的 config 源对齐，不能自己编。** 先 grep 引擎代码确认真实参数来源（`preprocessor_config.json`/`config_`），再写测试。这是 7.9"测试覆盖生产分布"的延伸：不只覆盖生产分辨率，还要用生产的真实参数。

7. **【2026-06-23 P10-B 工程化 BLOCKER-1】AA/抗锯齿算子只在降采样时有益，恒等/上采样时反而破坏精度**
   `aclnnUpsampleBicubic2dAA`（含抗混叠预滤波）在降采样时 vs PIL cos=0.99999（rescues P10-B），但在恒等尺度（416×672→416×672）下 vs CPU cos=0.950——AA 预滤器对不需缩放的图像做了不必要的平滑。**抗锯齿 resize 算子必须加降采样条件守卫**：`downsample = (new_h<height || new_w<width)`，仅降采样时用 AA，恒等/上采样用非 AA。这是 AA 算子的固有语义陷阱：AA 是为降采样防混叠设计的，对非降采样是噪声源。注意：bicubic 核本身在恒等下是单位冲激（见 #2 证伪记录），问题出在 AA 的额外预滤步骤。— P10-B 工程化 BLOCKER-1

8. **【2026-06-24 max_pixels 对齐官方】参考链"看起来对"≠"等于官方"，必须用真实官方类做 bit-exact probe**
   `gen_official_pixel_values.py` 最初用 `do_resize=True` + processor config 的 1310720（torchvision BICUBIC AA 链），cos 0.9999 过 gate——但这是"processor-config 链"非"官方 Qwen3VLEmbedder 链"，对降采样分辨率 grid 都不一样（4888 vs 6944 patches），虚假宣称 official。Reviewer 用真实 `Qwen3VLEmbedder` 的 unbound 方法（`format_model_input` + `_preprocess_inputs`）+ stand-in（只带必要属性、不加载 2B 权重）对同一输入产出 pixel_values，与参考 bin 对比——**4 分辨率 max_diff=0.0 bit-exact**，才把"看起来对"升级为"证明等于官方"。**可复用范式**：重实现参考链后，用真实官方类的 unbound 方法 + stand-in 做 bit-exact probe，比 cos 阈值更硬。cos 阈值只能证"接近"，bit-exact 才证"等同"。— max_pixels 对齐 Reviewer

9. **【2026-06-24 max_pixels 对齐官方】精度 margin 依赖输入分布，估算勿混用自然图与 worst-case**
   同一管线（NPU fp16 + bicubic-AA vs 官方 fp32）在自然图 cos≈0.99999，但在 worst-case 随机噪声输入 cos=0.999878（margin 减半）。计划用自然图估的 0.99999 当预期，实测噪声图 0.999878 看着像回归其实不是。**精度闸口的预期值和验收输入必须明确**：用代表性分布还是 worst-case？勿在估算里混用以免误判回归。降采样 bicubic 对高频噪声最敏感，是天然 worst-case。— max_pixels 对齐 Reviewer

10. **【2026-06-25 official full embedding gate】端到端参考不只要同图，还必须 token/chat template 同源**
   full embedding gate 若 C++ 端手写 input_ids 或复用旧 token bin，就算图像 pixel_values 与官方一致，也可能因 system prompt、chat template、image token 个数或默认 instruction 漂移导致 embedding 对不上。阶段2生成器通过 wrapper 捕获 `Qwen3VLEmbedder.process()` 内部同一次 `_preprocess_inputs` 的 `input_ids` 写 token bin，并用 public embedding vs captured-chain embedding `max_diff=0` 证明 embedding 与 token 同源。**规则**：端到端 vs 官方 gate 的 token 必须从同一次官方 preprocess 捕获，不能手写；必须 guard 默认 prompt（本模型为 `Represent the user's input.`）、max_pixels、image token 数；否则就是"同图不同问题"的假参考。— 阶段2 official full embedding gate

11. **【2026-06-25 testing-guide 沉淀】测试文档必须区分 official gate / self-consistency / diagnostic**
   阶段0–2 暴露的核心盲区不是“没有测试”，而是测试语义被混用：`benchmark --mode compare`、`test_accuracy`、path C 自比对能证明跨语言或路径一致，却不能证明 vs official；diagnostic 只打印 cosine 更不能当 gate。阶段3把这条纪律沉淀到 `evergreen/testing-architecture.md` 和 `evergreen/testing-guide-dev.md`：新增测试必须声明 `(平台 × 分辨率 × 路径 × 参考)`，写清是否 vs official，gate 必须退码，diagnostic 必须标 `not gated`。— testing-guide 阶段3

12. **【2026-06-25 build_and_test.sh 定位补漏】总入口脚本的语义必须写进 testing guide，不能只把底层命令散列出来**
   阶段3 testing guide 只列了 official gate、C++ 全量、benchmark 等命令，但没有解释 `build_and_test.sh` 的入口语义、refdata 三态和 fallback 风险，导致后来需要追问“它还重要吗”。根因是写文档时按“测试类型”组织，却没有按“开发者实际入口”组织；把脚本头部已有说明当作足够，忽略了 guide 才是开发者查验收路径的第一入口。**规则**：凡是有总入口脚本（build/test/deploy/benchmark driver），testing guide 必须明确写 4 件事：①它覆盖什么；②不覆盖什么；③常用模式；④什么日志意味着不是完整通过。尤其 `build_and_test.sh` 出现 `Falling back to --no-refdata` 或 `Excluding ... needs_refdata` 时，不能把最后的 PASS 解读为完整 C++ 全量 PASS。— build_and_test.sh guide 补漏

13. **【2026-06-25 official reference fallback】official reference 不是平台专属，平台 skip 只能是待修 gap**
   看到 `test_engine_vs_official` 在 310P skip，就把 `gen_official_embedding.py` 也误判成 910B-only，这是把“当前被测实现过不了”错误外推到“官方真相源不适用”。官方 `Qwen3VLEmbedder` reference 是全平台目标；如果 torch_npu 官方链在 310P 撞不支持算子，应 fallback 到 CPU 官方链生成 golden refdata，而不是跳过 reference。**规则**：平台限制只能体现在被测 C++ 实现是否达标，不能削弱 official reference；任何 `skip` 必须写成 tracked gap，最终目标仍是全平台 vs official cosine ≥ 0.99。— 310P official embedding generator 事件

14. **【2026-06-25 .env template root/src】配置变量语义必须与消费代码一致，并在脚本侧校验**
   `.env.example` 把 `QWEN3VL_EMB_SRC` 写成 `/path/to/Qwen3-VL-Embedding/src`，但 `gen_official_embedding.py` 又对该值追加 `/src`，导致用户按模板配置后变成 `/src/src` 并报 `ModuleNotFoundError: No module named 'models'`。这是模板和消费代码语义漂移，不是用户环境问题。**规则**：路径型 env var 必须明确是 repo root 还是 import root；模板、`env.py` 注释、消费脚本三处一致；脚本应打印最终解析路径并接受/校验常见输入，错误要指向配置语义而不是裸 import failure。— 310P official embedding generator 事件

15. **【2026-06-25 Reviewer BLOCKER-1】gate 测试必须守卫空 case 列表，防止假阳性**
   用户可能误配 `OFFICIAL_EMBED_CASES` 为空（如全空白、纯括号），导致 C++ gate 的 for 循环 0 次执行，failures=0 并打印 PASS——CTest 将其视为真 PASS，但实际没有测试任何 embedding。Python 侧 `parse_cases()` 有显式空列表检查抛出 `ValueError`，C++ 侧的等效守卫必须同步存在。**规则**：任何从用户配置解析 case 列表的 gate，必须在循环前显式检查列表非空，空列表就是 FAIL。— OFFICIAL_EMBED_CASES Reviewer

16. **【2026-06-25 Reviewer MAJOR 1/2/3】跨语言（Python/Shell/C++）的配置解析必须格式一致且无效值必须告警**
   `OFFICIAL_EMBED_CASES` 被 Python gen/Shell sentinel/C++ gate 三处独立解析，其中 bracket 剥离逻辑不一致会导致某处崩溃而别处通过。C++ 侧对不含 `x` 的 token 静默跳过无日志，用户不会知道某 case 被排除。**规则**：同一个配置项在多种语言中独立解析时，必须约定且文档化唯一格式（如 `HxW,HxW` 不含括号），且无效 token 必须 emit 日志（至少 WARN）。— OFFICIAL_EMBED_CASES Reviewer

---

## 主题 5：构建与 CMake

**触发关键词**：CMake、set_tests_properties、RESOURCE_LOCK、OOM、warning、-Wno、SYSTEM include、codemod、import、PRIVATE 链接、链接传递、undefined reference、显式重链

1. **CMake 顺序敏感，set_tests_properties 必须在所有 add_test 之后**
   RESOURCE_LOCK 声明引用的 target 必须已存在，放错位置 configure 报 `target not found`。— `refactor §4.13`

2. **显存 OOM 先用 RESOURCE_LOCK 串行化，别调小 buffer pool**
   先确认是"并发抢"还是"单个不够"。前者串行化（CPU 测试照常并发，性能损失小）；后者再调 pool 大小。调小 pool 会让 ATB Setup 因 workspace 不够失败。— `refactor §4.14`

3. **warning 抑制分清"我们的代码" vs "第三方头"**
   第三方头用 `target_include_directories(... SYSTEM)`；只有测试代码的弱实践（fread 不检查返回值等）才用 `target_compile_options(test_target -Wno-*)`。生产代码的 warning 必须能看到。— `refactor §4.15`

4. **codemod 改路径必须同步改 import 路径**
   批量替换 hard-coded path 后，依赖这些路径的 short-form import 也必须改。`python -c "import py_compile"` 不够，必须真正 import 一次。— `refactor §4.11/§4.17`

5. **环境变量用 `${VAR:?message}` 而非 `${VAR:-default}`**
   该被 `.env` 配置的变量，给默认值会让新人 silently 用错路径。`build_and_test.sh` source `.env` 后用 `${QWEN3VL_EMB_MODEL_DIR:?...}` 强制要求设置。— `refactor §4.16`

6. **`set_atb_buffer_size` 不能跨进程复用**
   `gen_all.py` 必须用 `subprocess.call([sys.executable, script])` 而非 `import`，每个生成器独立子进程，否则第二个起 ATB 图输出全 0。— `refactor §4.12`

7. **【2026-06-23 P10-B spike MAJOR-5】CMake `PRIVATE` 链接不传递到下游 target，下游直接用该库符号要显式重链**
   `opapi` 加入 `atb_llm_engine` 的 `PRIVATE` link libraries 后，所有 `add_atb_test` 注册的测试 link `atb_llm_engine`，但 PRIVATE 不传递——测试若直接调 opapi 符号（如 spike 直接调 `aclnnUpsampleBicubic2dAA`）会 undefined reference。引擎内部用 opapi（通过 engine.so 的 NEEDED 间接加载）没问题，但下游 target 直接用就要 `target_link_libraries(test PRIVATE opapi)` 显式重链。**判据：target 直接调用某库符号 = 必须显式 link 该库，不能依赖上游 PRIVATE 传递。** PRIVATE 只保证"上游用"，不保证"下游也能用"。PUBLIC/INTERFACE 才传递。新增测试直接调 aclnn/ATB 算子时 grep 确认链接。— P10-B spike MAJOR-5（路径C阶段已移除冗余 opapi 显式链接，因 spike 改走 engine wrapper 不再直接调）

---

## 主题 6：跨语言契约（C++ / Python）

**触发关键词**：token 命名、.bin 文件、跨语言接口、生成器、消费者

1. **C++ 和 Python 文件命名约定必须一致**
   C++ 读 `tokens_chat_mm_{W}x{H}.bin`、Python 生成 `tokens_mm_{w}x{h}.bin` → 不同步时 cosine 莫名其妙低，代码表面看不出。逐文件对照检查。— `refactor §4.8` / `arch §9.6`

2. **所有 .bin 由一个脚本统一生成，格式契约固化**
   格式：`[int32 count][element_type * count]`。生成者和消费者用同一份格式定义。— `refactor §4.8`

3. **跨语言工作流要端到端验证，不只分别测两端**
   生成→消费→对比全流程跑通，而非 C++ 测 C++、Python 测 Python。— `arch §10.2 第7项`

4. **【2026-06-25 CPU PIL 对齐 Batch C Reviewer】重实现一侧的参考算法后，所有同源参考生产者必须同步对齐，否则全管线级 cos 跌破；单阶段窄口径是钝指标**
   C++ CPU `BicubicResize`/`PreprocessImage` 重写为 bit-exact PIL 8bpc AA 后，vs 官方 transformers cos=1.0、Level-1 bit-exact max_diff=0 全绿——但 `test_stage_precision` IMAGE_ONLY 端到端 cos 跌到 **0.978341 FAIL**。根因不是 C++ 引擎（同测试 engine-only 诊断喂 Python 参考 pixel_values cos=0.999929），而是 **Python 引擎 `atb_python_qwen3vl_embedding/preprocess.py:85` 仍用 torch `F.interpolate(mode='bicubic')`（Mitchell a=−0.75，无 AA），未对齐 PIL**，其生成的 stage 参考 bin 过时。**教训一**：当某算法在一个语言/路径上重对齐了官方真实现，所有产同源参考的生产者（另一语言引擎、stage 参考生成器、e2e 基线）必须同步迁移，否则"单点对齐官方"会与"未迁移的参考链"在全管线级持续漂移——C++ 对了反而让 vs-Python 参考的 gate 红。**教训二**：单阶段 pixel_values 的窄口径 gate（`vision_stages` L0=0.999908，720→704 仅轻度降采样、只比单阶段）对全管线精度问题是**钝指标**，给出假安全；真问题要靠端到端嵌入级 gate（28 层放大预处理差异）+ engine-only 诊断（隔离是预处理还是引擎）才暴露。判定回归是否触发后续修复，不能只看最钝的单阶段 cos。— CPU PIL 对齐 Batch C Reviewer（触发 Batch D）

---

## 主题 7：工作流与流程

**触发关键词**：bug 报告、复现、基线、Developer/Reviewer、已知问题、猜测、配置漂移、知识孤岛、闸口验证、不熟悉的 API、官方文档、插桩试错、生产分辨率、工具耗时、基线直觉、平台支持矩阵、solo 开发、省 token、看起来小、派单门槛、七步前置、审查发现归档、Reviewer 发现、对标官方、官方推理运行时、config 默认值 vs embedder 常量、max_pixels 分裂、覆盖守卫、footgun

0. **【最高优先级，已复发两次，2026-06-22 P10-B 再犯】非平凡任务必须派单 Dev→Reviewer→Re-review，禁止 solo 写到底**
   WORKFLOW.md §3 + `dispatch-not-solo-dev` 记忆明文要求：中等及以上任务派 Developer agent 实现、Reviewer agent（独立实例）破坏者审查、Re-review 循环到零问题。**solo 开发跳过破坏者审查，建设者偏见（验证能工作而非找边界）必然让 bug 漏网**。P10-B 全程 solo：写 spike 测试 + 改 CMake + 改 wrapper + debug double-free + 改 gen 脚本，没派任何 agent。结果正是预言的——double-free、退化尺寸测试不覆盖生产场景、盲等 5 分钟慢脚本、盲信旧文档平台结论，**这些有 Reviewer 第一轮就会被挑出来**（"测试为什么没有生产分辨率？""gen 跑 5 分钟正常吗？""double-free 查官方文档了吗？"）。上一轮 Python NPU→CPU fallback 已因 solo 被用户纠错过（见 `dispatch-not-solo-dev` origin），这次是同型复发未闭环。— WORKFLOW §3 / `dispatch-not-solo-dev` 记忆

   **复发根因（必须内化，不止"知道"）**：
   - **门槛判断错误**：把 spike 错归类为"architect 前置复现、trivial、inline 最快"。实际它演变成多文件中等任务。**判断门槛用"改了几个文件/会不会有边界问题"，不用"我觉得简不简单"**。改 >1 文件、或涉及不熟 API、或要 debug —— 就是中等任务，派单。
   - **被"省 token/快"带偏**：solo 短期省 token，但 bug 漏网 → 用户纠错 → 返工，总成本更高。这是短期收益压倒长期风险的典型。
   - **跳过七步前置**：没走 WORKFLOW §2 七步（尤其第 7 步"编写工作范围→developer 不能自己猜"），直接写代码。七步本身就是派单的触发器。
   - **教训没泛化**：`dispatch-not-solo-dev` 只在"明显大任务"时想起，"看起来小的任务"就忘。教训必须覆盖**所有非平凡任务**，不分大小。

   **How to apply**：动手前先过 WORKFLOW §2 七步；非平凡任务（改 >1 文件 / 不熟 API / 要 debug / 有精度或边界问题）→ 派 Developer，实现完派 Reviewer 破坏者审查，Re-review 到零问题。trivial 仅限单行 typo/明显 bug/纯查文件。不确定算不算 trivial 时，**默认派单**。

1. **【核心】收到 bug 报告第一步是复现，不是读代码**
   先在可用硬件运行用户报告的确切命令建立基线，再读代码。"测试全部通过"不能替代"实际运行出问题的那个命令"。架构师曾直接分析代码跳过复现，导致第三个根因完全遗漏。— `arch §9.6/§9.7`

2. **Developer → Reviewer → Re-review 循环直到零问题**
   Developer 有建设者偏见（验证能工作），Reviewer 有破坏者视角（找边界情况），必须用不同 agent 实例（上下文隔离）。Reviewer prompt 要求"辩证挑刺"而非"验证正确性"。— `arch §9.1/§9.3`

3. **修复前先搜 docs 中的已知问题记录**
   `refactoring-plan` 早就记录了 token 命名不匹配，但从未纳入 test-fix-plan 可操作项 → 知识孤岛，缺陷持续 6 个月。修复前 grep `docs/*.md` 的"症状/根因/教训"。— `arch §9.6/§10.2 第8项`

4. **审计文档之间要联动，已知问题必须转入可操作追踪**
   任何审计/回顾文档标记的已知问题必须同步到 STATUS.md 待办。定期扫描 docs 里的"症状/根因"关键词提取未修复项。— `arch §10.2 第8项`

5. **不清楚需求就问，禁止猜测**
   Architect 不清楚时用 grill-me skill 向用户彻底问清楚；developer agent 遇到不明确汇报给 architect，不自己猜。— `arch §9.7`

6. **同型 bug 复发说明教训未闭环**
   §9.6 发现"benchmark 不在 CTest"，§9.7 加了工作流，但只用于"修 bug 时复现"，没扩展到"提交前冒烟所有 mode"→ §9.8 重犯。教训必须泛化为通用规则，而非针对单次事件。— `arch §9.8`

7. **【核心，2026-06-22 P10-B 复发】精度闸口未验证前禁止铺工程化**
   闸口类任务（"先试 X 能不能用，能用再工程化"）的正确顺序：写最小验证代码 → 跑通精度/可行性闸口 → 通过才铺开管线。P10-B 反过来：在 `aclnnUpsampleBicubic2d` 精度未验证前就展开了 `engine.h::GetRuntime()`、`PreprocessImageNpu` 头声明、整套 wrapper —— 闸口万一不过这些全是废代码。**闸口验证代码要小到能一次扔掉**（一个测试函数 + 现有 .bin），不是先搭脚手架再测。这是 7.5「禁止猜测」在工程节奏上的具体化：用假设（"aclnn 语义=torch 应该对齐"）代替了验证。

8. **【核心，2026-06-22 P10-B 复发】遇到不熟悉的 API，第一动作是查官方文档/示例，不是猜+插桩试错**
   `NpuBicubicResize` double-free 折腾多轮（调 free 顺序、加 sync、跳过 free），全在凭经验猜。用户发 CANN 9.0.0 官方文档，一眼看出官方示例**根本不调 `aclDestroyAclOpExecutor`** —— executor 是 stream 生命周期管理，手动 destroy 必然 double-free。官方示例还展示了正确的 `aclCreateTensor` 用法（`ACL_FORMAT_ND` + 显式 strides，非 `nullptr` stride + `NCHW`）。**遇到没用过的 API，先花 5 分钟查官方文档/头文件签名/官方 sample，再写第一行代码**。猜试错的 token 成本远高于读文档。这是 7.3「修复前先搜已知问题」向"外部权威源"的延伸：docs 内的已知问题 + 外部官方文档都要查。

9. **【2026-06-22 P10-B】测试 case 必须覆盖生产输入分布，退化尺寸结论不可外推**
   spike 初版只用 2×2/4×4/8×8 退化尺寸就下"通过闸口"结论。这些尺寸 boundary band 占比极高，与生产图（1080×1920，interior 主导）输入分布完全不同，结论不可外推。被用户当场指出。补生产分辨率后 cos 从小图的 0.987–0.999 稳定到生产的 0.9993–0.9997。**验证类测试的 case 选择要问"真实场景输入长什么样"，覆盖生产分辨率/规模，不是挑最容易跑的尺寸**。这是主题 3.1"编译通过≠功能正确"在测试设计层的延伸。

10. **【2026-06-22 P10-B】对工具链耗时要有基线直觉，异常慢立即排查而非盲等**
    `gen_cpu_reference.py` 跑了 5+ 分钟还在转，一直 `sleep` 等没质疑。PIL resize 是 C 实现，1080×1920 秒级；纯 Python 标量循环（`_cpp_bicubic_resize` 4×4=16 tap 逐像素）才会几分钟。根因是生产 case 复用了慢的 C++算法翻译参考，而 spike 只需 PIL 参考。**启动一个可能耗时的生成/构建步骤前，先估算"正常该多久"；超出 2–3 倍立即查代码或看实时输出流，不要盲等**。且首跑用前台实时输出，别用 `| tail` 管道缓冲掉进度。

11. **【2026-06-22 P10-B】文档里的平台/能力结论是易过期事实，使用前必须回查权威源**
    roadmap 写"aclnn 仅 910b kernel、无 310P"，当约束用了，设计了"910b 特化"定位。用户发 CANN 文档一看：310P（Atlas 推理系列）也 √ 支持，定位错了。**文档记录的"某算子支持哪些平台/数据类型/格式"是会随 CANN 版本变化的事实，使用前必须回查当前 CANN 版本的官方文档或实测**，不能当不变约束。这把 7.3"搜已知问题"的义务反过来：文档里的能力结论本身也可能是错的，需要核实而非照搬。

12. **【元规则，2026-06-23】Reviewer 探查出的每个 BLOCKER/MAJOR（及有泛化价值的 MINOR）必须归档进本文件对应主题**
    Reviewer 破坏者审查发现的 bug/陷阱是开发中真实遇到的困难，不归档就会同型复发。architect 在 Re-review 闭环、提交前完成归档：同主题已有更强条目则合并/替换，没有则新增，并补触发关键词。判据：边界/所有权/精度/内存/并发/平台类发现一律归档；纯个案（如某变量名拼写）可不归档。流程定义见 [WORKFLOW.md](./WORKFLOW.md) §3.3「审查发现归档纪律」。本条是元规则——它管"其他教训怎么进来"，不是某个具体坑。

13. **【2026-06-24 max_pixels 对齐官方】"对标官方"必须对标官方推理运行时，不是 model card / config 文件的默认值**
    官方 `Qwen3VLEmbedder` 推理时 `max_pixels=1843200`（`qwen3_vl_embedding.py:28` 硬编码注入 content dict + `do_resize=False`），但模型 `preprocessor_config.json` 写的是 `1310720`（image processor 通用默认，do_resize=False 时不参与 resize）。我们引擎读 config 的 1310720，导致降采样分辨率 grid 与官方不一致（4888 vs 6944 patches），embedding 不在同一序列空间，无法对标官方。**根因**：config 文件是 image processor 通用默认，被上层 embedder 的常量覆盖；"官方推理"看的是 embedder 运行时，不是 config。**教训**：对标官方时先查官方推理入口（`Qwen3VLEmbedder.__init__`/`process`）实际注入的参数，而非 model card / preprocessor_config 默认值——后者是通用基线，常被业务层覆盖。修复用 embedder 常量 `kQwen3VLEmbeddingMaxPixels` 覆盖 config，所见即所做（避免未来 dev 看到 config field=1310720 又"修复"回去，footgun 复发——本条是 7.6"config field 必须有消费代码"的延伸：被覆盖的 config field 要用覆盖守卫证明"故意忽略"而非"读取失败"）。— max_pixels 对齐 BLOCKER（Reviewer MAJOR-1 升级）

---

## 主题 8：代码设计

**触发关键词**：config 字段、debug dump、DEPRECATED、wrapper、死代码、返回值、接口契约、host/device 指针、裸指针契约、数据位置绑死、张量抽象、D2H/H2D 往返、调用方 sync、free-safety sync、所有权转移、Detach 静默、double-free、头注释合约漂移、stale 合约、goto-cleanup、资源泄漏、提前 return 泄漏、多资源分配

1. **每个 config 字段必须有消费代码**
   新增字段问"谁读它？怎么验证它确实被读了？"用 `grep -rn "\.field_name"` 验证写→读链路。— `refactor §4.2`

2. **debug dump 抽成工具函数，不污染业务路径**
   调用方一行 `debug::DumpNpuFp16(rt, tensor, count, path)`，不要在 production 路径散布 `if (getenv(...))` 块。— `refactor §4.4`

3. **DEPRECATED 标签必须配套 `[[deprecated]]` 或 LOG_WARN，否则删掉**
   标了 DEPRECATED 但代码实际优先读它 = 误导。— `refactor §4.5`

4. **Wrapper 类从第一天起就要有区别于底层的 contract**
   `Encode` 校验 batch_size==1、输出 shape=={hidden_size}，不能只是 pass-through。— `refactor §4.6`

5. **返回值必须检查**
   16 处 `CopyToDevice`/`CopyToHost` 返回值未检查 → 静默失败。统一 `Status s = ...; if (s != STATUS_OK) { LOG_ERROR; return s; }`。— `arch G2`

6. **【2026-06-23 P10-B】接口契约不要用裸 host 指针绑死数据位置，跨 host/device 流水会被迫往返搬运**
   `PreprocessedImage.pixel_values` 用 `const void*`（CPU 指针）作为预处理结果的交付契约，是引擎公共接口（`InferRequest.preprocessed`），贯穿 `PreprocessImageNpu`→`ForwardWithTiming`→`RunVision` 整条链。后果：NPU preprocess 在 device 上算完 `transpose_out`，却因契约要求交付 CPU 指针被迫 D2H（~15MB），`RunVision` 拿到 CPU 指针又 H2D（~15MB）回 NPU 喂 PatchEmbed——数据已在 NPU 上却绕一圈 CPU，~30MB 往返纯浪费。改 device tensor 输出要动 6 处（结构 + 两实现 + Forward 传参 + RunVision 签名 + benchmark 对比路径 + 所有权），因为"数据在 host"这个实现细节被契约扩散到了链上每个环节。

   **根因**：用裸指针做张量契约，把"数据当前在哪一侧（host/device）"这个本该运行时决定的事，写死进了类型签名。

   **避免模式**：用 host/device 双栖的张量视图做契约（轻量描述符，同时记 `device_ptr`(可空)/`host_ptr`(可空)/shape/dtype + "数据当前在哪侧"状态）。生产方填自己产出的那侧，消费方按需取，需对侧才搬运。类似 PyTorch `Tensor` 可 `.cuda()/.cpu()` 或 ACL `aclTensor` 带 deviceData。

   **何时引入（不要过早）**：等真正出现"数据已在 device 却被迫搬回 host 再搬回 device"的可测收益触发点再做。P10-B 当前收益仅几 ms（搬运非瓶颈，bicubic compute 是主体），且 `PreprocessedImage` 是已稳定的对外公共接口（`PREPROCESSED` 模式对外暴露），改它有破坏性。记为长期方向，等 NPU preprocess 接入主路径后若大图卡搬运再引入。**别为一个还没上线路径的搬运优化，提前做跨 6 处的公共契约改造。** — P10-B device tensor 输出评估

7. **【2026-06-23 路径C MINOR-1 + spike MAJOR-3】NPU 函数的 sync 责任要明确且不重复——wrapper 内别强制 sync 破坏异步流水，调用方别重复 sync**
    sync 语义的两个正反面陷阱：① **wrapper 内强制 sync 破坏异步流水**（spike MAJOR-3）：`NpuBicubicResize`/`NpuBicubicResizeAA` 每次 Execute 后强制 `aclrtSynchronizeStream`——spike 场景可接受,但若放进 preprocess 热路径,每次 resize 都 sync 会抵消 NPU 加速、且意外等待上游异步 work（与主题 1 第 3 条"H2D/D2H 已自带同步,额外 Synchronize 破坏异步流水"同源）。wrapper 应让调用方管 sync,或文档明确"sync 由本函数负责"。② **调用方重复 sync 是空操作 + 误导**（路径C MINOR-1）：`PreprocessImageNpuInternal` 末尾为 free-safety 已无条件 sync,调用方又加 `if (!SkipTimingSyncs()) Synchronize()` 并注释"capture true NPU time"——空操作,注释把功劳归错。**规则:NPU 函数的 sync 责任在头注释声明清楚(自带 sync / 调用方管);调用方加 sync 前确认被调函数是否已 sync(尤其有 free 中间 tensor 的函数必有 free-safety sync)。** free-safety sync 是异步瓶颈,要异步需 deferred-free intermediates。— spike MAJOR-3 + 路径C MINOR-1

8. **【2026-06-23 路径C Reviewer】所有权转移 API（Detach/erase/release）对未跟踪/已转移对象必须显式告警，不能静默 no-op**
    `TensorAllocator::Detach` 用 `allocations_.erase(key)`，对未跟踪 tensor 返回 0、静默 no-op——可能掩盖 double-Detach 或 Detach 非 allocator tensor 的误用。所有权转移是 double-free 高危区，静默吞掉误用最危险。**所有"从跟踪/管理中移除"语义的 API（Detach/untrack/release），未命中时必须 LOG_WARN 或 debug assert，不能静默。** 正确调用方不受影响（总是操作 tracked 对象），误用时立即暴露。— 路径C MINOR-2

9. **【2026-06-23 P10-B spike BLOCKER-3 + 工程化 Reviewer】头文件注释/合约必须与实现同步，改实现必改头注释**
    `aclnn_bicubic_resize.h` 注释写"910B-only NPU bicubic resize"、"Returns ERROR_UNSUPPORTED on 310P"，但 `.cpp` 已改为跨平台（无平台守卫，永不返回 ERROR_UNSUPPORTED）。调用方若按 header 合约写 `if (Is910B()) ... else fallback`，在 310P 上会走多余 fallback；更糟的是 stale 合约是**错误信息**，误导未来 developer。根因：改了实现（去掉平台守卫）没同步改头注释。**改函数行为/平台支持/返回码语义时，头文件 doxygen 注释必须同步更新；合约漂移比没注释更危险。** Reviewer 审查时对照头声明与实现是固定检查项。— P10-B spike BLOCKER-3

10. **【2026-06-23 P10-B 工程化 MAJOR-1】多资源分配函数必须用 goto-cleanup / RAII，提前 return 会泄漏已分配资源**
    `PreprocessImageNpu` 分配 5 个 device tensor（input/resized/normalize_tmp/mean_bc/inv_std_bc）+ workspace，每步失败都 `return`——中间任一步失败，前面已分配的 tensor 全泄漏（device memory 不像 host 有 RAII 自动回收）。Reviewer 列出 8 条泄漏路径。**分配多个资源的函数，统一用 goto-cleanup 模式（或 RAII wrapper）：所有资源前置声明为空，错误路径 goto cleanup，cleanup 段逆序释放非空资源；正常路径也走 cleanup 或显式释放。** 判据：函数内 alloc >1 个 device/host 资源 + 有多个失败点 = 必须统一 cleanup。这是 C 资源管理的通则，device memory 更甚（泄漏不 crash 但耗尽 HBM）。— P10-B 工程化 MAJOR-1

---

## 关联

- 工作流细则（审查标准、agent briefing 模板）：`archive/architect-assessment-2026-06-16.md` §9.7
- 预防机制检查清单（13 项）：同上 §10.2
- 开发检查清单（8 项）：`archive/refactoring-plan.md` §6
- 跨会话记忆：`.claude/.../memory/benchmark-modes-smoke-test.md`（主题 3 第 1 条的 memory 镜像）
