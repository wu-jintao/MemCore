# 公开数据检索评测

本工具衡量公开上游数据的**证据召回**，不是 AML 成绩、答案正确率、官方 Smoke 或 Full。实现为独立编写的 Python 标准库程序，没有直接调用付费 embedding、回答模型或评分模型。

数据版本、下载 URL、SHA256、作者、许可证和指标口径见 `../public_data_manifest.json`。LongMemEval-S cleaned 为 MIT 标注的公开版本；LoCoMo-Refined 及其原始 LoCoMo 材料为 CC BY-NC 4.0，保留下载的 LICENSE 与 NOTICE，只作本地非商业研究，不把数据打包进服务或开源发布。底层数据来源仍有各自使用条款。

## 准备

在仓库根目录运行：

```bash
python3 competition/eval_prepare.py --download
python3 competition/eval_test.py
```

默认写入 `.local/public-data/`。已存在文件先核对固定大小和 SHA256；原始、准备后的数据都不写入公开源代码目录。默认下载约 280 MB：500 个 LongMemEval 构造历史和 10 个 LoCoMo-Refined 会话。

准备后的每个 split 有三个独立文件：`histories.jsonl` 只有历史，`queries.jsonl` 只有问题，`gold.jsonl` 保存评分侧答案与证据标注。程序重建 Add 时只选择角色、原文、公开说话者、来源 session 日期和不包含数据集名称的标识符，不发送 question、answer、has_answer、evidence、event_summary、observation 或 session_summary。Search 只发送原始问题及隔离字段，不注入 question_date 或答案。

LoCoMo 按完整会话固定 6/2/2 分为 dev、validation、heldout。同一个会话的全部问题始终在一个 split。LongMemEval 没有持续真实 user ID；按整个构造历史分组，并把复用已标注证据 session 的实例连在一起再分组。上游 filler session 可跨构造历史复用，不能声称整个原始语料完全隔离；summary 会披露跨 split 复用数量。

LoCoMo 默认排除 `is_multi_modality=true` 题，证据编号无法严格解析/定位的题也整题排除。LongMemEval 的 30 道拒答题没有对应召回金标，按其上游检索评测做法排除。上游少量空白 filler 消息没有可检索文字，准备时省略并统计；若空白消息被标为证据，则报错而非伪造内容。这些排除形成一个公开文本代理子集，无法复现 AML 私有文本套件采样。

来源日期没有时区时，Unix 毫秒采用 UTC 作为编码约定，原始日期字符串保留在每条历史中。两人 LoCoMo 对话按 speaker_a → user、speaker_b → assistant 包装，speaker 姓名保留在内容。历史文字不改写。分块上限 20 消息/2,000 空格词，是本工具的可复现近似，不等同官方 Adapter tokenizer。

## 开发集运行

自动启动默认词法服务，使用新数据库与随机 token，结束时关闭这个本地服务：

```bash
python3 competition/eval_retrieval.py --dataset longmemeval_s --split dev --max-histories 20 --launch-local --model-label lexical-context1
python3 competition/eval_retrieval.py --dataset locomo_refined --split dev --max-histories 1 --launch-local --model-label lexical-context1
```

自动启动默认显式设置 `MEMORY_EMBEDDING_PROVIDER=disabled`，不会启用语义提供方。已安装 CPU 模型时可以显式比较同一个开发子集：

```bash
.local/embedding-venv/bin/python competition/eval_retrieval.py --dataset locomo_refined --split dev --max-histories 1 --launch-local --embedding-provider local --embedding-model-dir .local/models/multilingual-e5-small --model-label hybrid-e5small-context1
```

local 模式仅加载已经下载的 E5，强制离线，threads=2、batch=8、concurrency=1，没有付费 HTTP provider 选项。当前工具清理继承的 MEMORY_EMBEDDING 配置/外部密钥，显式固定窗口 384、重叠 64、query/passage 前缀、段数上限与队列/操作时限，记录安全的实际启动环境。`--context-radius 0` 可单独验证邻接窗口的收益；其他配置相同才能比较。自行连接服务时必须明确说明 backend；本工具不推断该服务是否会调用付费模型。

`--semantic-weight 0.5` 可在同一开发子集降低全局语义 RRF 权重；词法权重固定 1。权重不按用户、问题或数据集变化。启用 local 而权重为 0 仍会在 Add 构建语义索引，因此只做 Search 消融时不能把它当作完整词法成本。

连接已经启动的本地服务：

```bash
export MEMORY_API_TOKEN='使用服务实际鉴权值，不要把真实密钥写进文件'
python3 competition/eval_retrieval.py --dataset locomo_refined --split dev --max-histories 1 --base-url http://127.0.0.1:8080 --model-label 明确的方法与版本
```

结果写入每次唯一的 `results/<run_id>/`，包括 `summary.json`、逐问题 `retrieval.jsonl`、本地 service 日志和启动时的代码快照。原文命中、延迟和失败数量均来自实际运行；配置记录数据/hash、服务代码开始/结束 hash、并发及预算。若服务文件中途改变，`server_source_changed_during_run=true`，不能将该次运行当固定版本比较。results 的 `.gitignore` 默认排除这些实际数据衍生结果，不把检索证据自动纳入公开仓库。

本地数据库在 `.local/public-data/runs/<run_id>/`。比较不同方法使用同一 prepared split 与同一 max-histories、max-queries、top-k、预算、并发；每次使用独立运行范围。只运行小开发集不能证明整套效果，也不能把 macOS 速度当作云机容量。

## 指标与限制

- Recall-any：至少完整返回一个标注证据 turn 的问题比例。
- Recall-all：完整返回全部标注证据 turn 的问题比例。
- Macro turn recall：每题覆盖证据 turn 的比例，再按题平均。
- Source-verified turn recall：匹配原始 Add 来源位置/稳定消息 ID，同时完整包含原始 turn。只匹配 metadata 或 session 不计证据命中。
- 无来源字段时，只有在该历史中唯一出现的完整证据文字能作为命中；重复的“好的”等泛泛文字须有正确来源，否则不计。
- Session recall 单独报告，属于粗指标；它不能替代真实证据 turn 覆盖。
- API 失败留在全部请求问题的分母中，不默默剔除。按返回前缀计算 k=5/10/20/50/100；如果设置字符预算，只保留预算内完整条目，明确记为字符而不是 token。

已有运行可以离线比较相同 k 和字符预算，不必重新发送模型或检索请求：

```bash
python3 competition/eval_compare.py --runs competition/results/词法运行ID competition/results/混合运行ID --output competition/results/comparisons/同子集对比.json
```

程序检查完全相同的题目 ID、split 和 prepared 数据 hash，然后对已返回证据的完整条目前缀重新评分。默认同时报告不限字符及 8,000/16,000/32,000/64,000 字符预算；这些数值是字符预算，不模拟官方 Answer。它还报告每档相对首个运行新增/丢失完整证据覆盖的题数。

邻接窗口中的每个 `[source {...}]` 段分别检查原始内容与来源，不将 item ID 误当整个窗口的来源。目标答案字符串不参与召回判断，不凭“Paris”等答案词给完整 turn 加分。标注 turn 的完整覆盖仍不保证回答模型答对；答案评测在取得模型额度后独立接入，不从这些召回值计算虚构的 AML Overall。

评分口径 `public-full-turn-body-only-v2` 仅在实际正文段中检查完整原文，生成的 source JSON 和邻接结构标签不参与命中。已知原文中的合法 source 形状引用保留，每个来源正文独立匹配，禁止跨两个 wrapper 拼接成虚假的完整 turn。17 项离线正确性回归包含 metadata-only 短句、伪造已知 anchor ID、邻接标签、合法引用、跨正文拼接和继承环境配置污染；这些自造用例不作为数据集成绩。

早期未记录 grader_version 的运行使用存在 metadata-inclusive 缺陷的旧口径。保留其原始 summary 和 retrieval 不改写；`eval_compare.py` 对原始检索结果按当前正文口径重新评分，另存报告并列出命中计数变化。汇总见 `RESULTS.md`，以其标注的新评分口径为准。没有因这次修复重新发送 embedding、检索或付费请求。

已完成且 WAL 为空的本地语义开发运行可做只读配置核验，不加载模型、不读题目/正文、不请求 API：

```bash
python3 competition/eval_audit.py --model-dir .local/models/multilingual-e5-small --runs competition/results/语义运行ID --output competition/results/comparisons/semantic-index-audit.json
```

核对固定 revision 的模型文件 SHA、索引 fingerprint/维度/float32 长度，以及 16 个确定顺序存储向量的 L2 范数样本。这是模型配置审计，不是新效果实验或负载证明。

validation 用于方法选择；heldout 保留最终确认。运行 heldout 必须显式 `--final-heldout`，且不能只挑子集；一次尝试会写锁，失败也保留审计记录。不要删除锁反复调参。当前工作的开发集运行不会访问留出集检索结果。

## 2026-10-06 原生身份审计更新

全量 history-session 连接审计发现，470 道 LongMemEval 题属于一个连通分量；当前全部只作为开发集，没有独立 LME validation。上述早期 evidence-session 分组是历史流程说明，不构成全语料隔离。LoCoMo 的完整会话 547/183/129 题划分保持不变。见 `../eval_validation.py` 和 `CAPABILITY_COVERAGE.json`。
