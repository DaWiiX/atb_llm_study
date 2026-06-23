# 踩坑经验集（Lessons Learned）

> **按主题组织，不按时间。** 做事前按主题检索，避免同类坑复发。每条带「触发关键词」——当你脑海中冒出这些词，先查这里。
>
> 维护规则：踩了新坑就归并进对应主题（没有就加主题），同主题只留最强的那条；检查是否有更弱的旧条目可替换。跨会话需主动 recall 的规则同步到 `.claude/.../memory/`。
>
> 来源标记：`refactor §4.x` = refactoring-plan §4；`arch §9.x` = architect-assessment §9；`audit` = audit-fix-plan；`310p` = platform-310p。

---

## 主题 1：精度调试

**触发关键词**：cosine 低、NaN、精度下降、block 输出异常、fp16 二进制、跨语言数值

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

---

## 主题 2：平台差异（910B / 310P）

**触发关键词**：310P、910B、NZ mask、FRACTAL_NZ、TransdataOperation、setup fail、status 4、ASCEND_PLATFORM、GQA

1. **910B 模拟 310P 不可信，必须真机验证**
   GQA 在 910B 模拟 310P 模式失败，真 310P 完全正常。— `310p`

2. **310P mask 必须 NZ 格式 + FRACTAL_NZ format tag（二者缺一不可）**
   PA_ENCODER 下 910B 用 ND、310P 用 NZ。数据 layout 是 NZ 还不够，NPU tensor 的 format tag 也必须是 `ACL_FORMAT_FRACTAL_NZ`(29)，否则 ATB 内部 ND→NZ Transdata 失败："call operation setup fail"。— `310p` / `arch §9.6`

3. **C++ 和 Python 平台逻辑必须同步**
   C++ 改了 NZ mask 转换，Python `engine.py` 也必须改，不一致会一方通过一方失败。— `310p`

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

**触发关键词**：静态审查、运行时测试、测试覆盖、refdata、跳过、阈值、假阳性

1. **静态审查 ≠ 运行时测试**
   API 行为误解、运行时竞态、性能回归、兼容性、数值精度——这五类只有实际运行才能发现。审查通过 = 静态审查零问题 + 单元测试 PASS + 复现确认问题消失。— `arch §9.4`

2. **needs_refdata 标签必须完整**
   `test_io_adapters` 读 `preprocess_*.bin` 但不在 needs_refdata 列表 → `--no-refdata` 时静默跳过精度验证，CTest 仍报 PASS（假阳性）。新增读 `/tmp/*.bin` 的测试必须登记到 CMakeLists REFDATA_DEPENDENT_TESTS。— `audit C1`

3. **参考实现不能自证**
   BicubicResize 用自己的实现当参考 = 没有独立参考。参考实现必须是同输入下的黄金标准（本项目为 transformers）。— `test-fix-plan P7`

4. **测试阈值不能放松来"通过"**
   C++ 和 Python 相同输入 cosine < 0.99 就是 bug，定位根因，不放宽阈值（CLAUDE.md 测试精度原则）。— 项目原则

5. **重复测试要合并**
   `test_consistency.cpp` 是 `test_accuracy.cpp` 子集、多个 NZ 验证脚本冗余——删除子集，保留最全的。— `test-fix-plan P9/P11`

---

## 主题 5：构建与 CMake

**触发关键词**：CMake、set_tests_properties、RESOURCE_LOCK、OOM、warning、-Wno、SYSTEM include、codemod、import

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

---

## 主题 6：跨语言契约（C++ / Python）

**触发关键词**：token 命名、.bin 文件、跨语言接口、生成器、消费者

1. **C++ 和 Python 文件命名约定必须一致**
   C++ 读 `tokens_chat_mm_{W}x{H}.bin`、Python 生成 `tokens_mm_{w}x{h}.bin` → 不同步时 cosine 莫名其妙低，代码表面看不出。逐文件对照检查。— `refactor §4.8` / `arch §9.6`

2. **所有 .bin 由一个脚本统一生成，格式契约固化**
   格式：`[int32 count][element_type * count]`。生成者和消费者用同一份格式定义。— `refactor §4.8`

3. **跨语言工作流要端到端验证，不只分别测两端**
   生成→消费→对比全流程跑通，而非 C++ 测 C++、Python 测 Python。— `arch §10.2 第7项`

---

## 主题 7：工作流与流程

**触发关键词**：bug 报告、复现、基线、Developer/Reviewer、已知问题、猜测、配置漂移、知识孤岛、闸口验证、不熟悉的 API、官方文档、插桩试错、生产分辨率、工具耗时、基线直觉、平台支持矩阵、solo 开发、省 token、看起来小、派单门槛、七步前置、审查发现归档、Reviewer 发现

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

---

## 主题 8：代码设计

**触发关键词**：config 字段、debug dump、DEPRECATED、wrapper、死代码、返回值、接口契约、host/device 指针、裸指针契约、数据位置绑死、张量抽象、D2H/H2D 往返、调用方 sync、free-safety sync、所有权转移、Detach 静默、double-free

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

7. **【2026-06-23 路径C Reviewer】调用方加 sync 前先确认被调函数是否已 sync，避免空操作 + 误导注释**
    `PreprocessImageNpuInternal` 末尾为 free-safety 已无条件 `runtime->Synchronize()`（释放 AsStrided/Transpose 中间 tensor 前必须等异步算子完成）。调用方 `ForwardWithTiming` 又加了 `if (!SkipTimingSyncs()) Synchronize()` 并注释"capture true NPU preprocess time"——实际是空操作，注释把功劳归错。根因：调用方不知道被调函数内部已 sync。**调一个 NPU 函数后再加 sync，先确认该函数是否已自带 sync（尤其有 free 中间 tensor 的函数必有 free-safety sync）；已 sync 则调用方不要再加，注释说明计时由谁保证。** 附带限制：free-safety sync 是异步瓶颈，要异步需 deferred-free intermediates。— 路径C MINOR-1

8. **【2026-06-23 路径C Reviewer】所有权转移 API（Detach/erase/release）对未跟踪/已转移对象必须显式告警，不能静默 no-op**
    `TensorAllocator::Detach` 用 `allocations_.erase(key)`，对未跟踪 tensor 返回 0、静默 no-op——可能掩盖 double-Detach 或 Detach 非 allocator tensor 的误用。所有权转移是 double-free 高危区，静默吞掉误用最危险。**所有"从跟踪/管理中移除"语义的 API（Detach/untrack/release），未命中时必须 LOG_WARN 或 debug assert，不能静默。** 正确调用方不受影响（总是操作 tracked 对象），误用时立即暴露。— 路径C MINOR-2

---

## 关联

- 工作流细则（审查标准、agent briefing 模板）：`archive/architect-assessment-2026-06-16.md` §9.7
- 预防机制检查清单（13 项）：同上 §10.2
- 开发检查清单（8 项）：`archive/refactoring-plan.md` §6
- 跨会话记忆：`.claude/.../memory/benchmark-modes-smoke-test.md`（主题 3 第 1 条的 memory 镜像）
