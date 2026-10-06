# 扩展公开 dev：固定两候选比较

在冻结版本和相同 token 预算下，等权 E5 混合是当前领先候选：LoCoMo 六个 dev 会话的完整证据召回从 89.58% 提升到 92.14%，固定 LME60 从 91.67% 提升到 95.00%。这支持继续验收混合方案；正式使用仍需云端容量、接口和最终配置通过。这里没有执行 Answer/Judge、官方 Smoke/Full，也没有官方成绩。

## 范围与约束

只比较词法 context1 和离线 multilingual-e5-small 等权混合 context1，top_k=100、串行 Search、评估器后处理字符预算为 0；没有扫权重或新增方法。LoCoMo 使用全部六个已固定的公开 dev 会话（547 题、3,616 条历史消息、673 条 gold turn 标注）；LME 使用准备后的固定 dev 顺序前 60 个历史（60 题、29,242 条历史消息、126 条 gold turn 标注）。没有读取 validation/heldout 检索结果，没有标签、问题或答案进入 Add，没有付费请求。

每个候选使用独立数据库和原始 run ID。Lo 每个候选 245 次 Add、547 次 Search 全部成功；LME 每个候选 4,095 次 Add、60 次 Search 全部成功。原文与派生向量同步提交，未将未完成索引计作 Add 成功。服务仅本机 loopback，完成后评测服务关闭，完整公开 dev 数据库保留。

All 表示某题的全部 gold 证据 turn 完整命中；Macro 表示各题证据 turn 召回的均值。计分只匹配实际来源正文，排除包装 metadata 和邻接结构标记，session 命中不能代替 turn 命中；API 失败仍计入请求分母。完整证据的 Recall-all 可以诊断检索，无法衡量回答准确率、无关证据干扰或官方 Overall。

## 相同返回条目上限

表中 All/Macro 均为百分比。“新增/丢失”按同一题配对完整证据覆盖，相对词法基线。k=100 是请求上限，服务可能返回更少条目。

| 数据 | k | 词法 All/Macro | 混合 All/Macro | 混合新增/丢失题 |
|---|---:|---:|---:|---:|
| Lo6 | 5 | 62.16/65.55 | 59.23/62.43 | +44/−60 |
| Lo6 | 10 | 69.84/73.23 | 67.09/70.73 | +39/−54 |
| Lo6 | 20 | 75.69/79.43 | 75.87/79.43 | +41/−40 |
| Lo6 | 50 | 82.45/86.14 | 86.29/89.53 | +42/−21 |
| Lo6 | 100 | 89.58/92.55 | 92.14/95.08 | +19/−5 |
| LME60 | 5 | 56.67/68.61 | 58.33/73.50 | +4/−3 |
| LME60 | 10 | 68.33/78.36 | 66.67/79.94 | +4/−5 |
| LME60 | 20 | 78.33/86.11 | 80.00/88.06 | +4/−3 |
| LME60 | 50 | 86.67/92.00 | 83.33/94.17 | +0/−2 |
| LME60 | 100 | 91.67/95.17 | 95.00/98.22 | +2/−0 |

Lo6 的六个会话在 k=100 的 All 均提高，按会话等权均值为 89.81%→92.42%；题目加权表为 89.58%→92.14%。Lo6 k=5/10 的混合 All 下降，LME60 k=10/50 也下降，因此不能把大预算改善扩展为任意较小 k 均获益。六个 LoCoMo 会话内题目相关，固定前 60 个 LME 历史也可能不能代表整个基准分布；这里不作显著性或通用优越性结论。

## 相同 token 预算

使用官方开源 tiktoken 0.14.0；主表用 o200k_base，并用 cl100k_base 作敏感性检查。公开配置未披露生产 Answer 的确切模型和 prompt，因此模型家族、序列化和指令预留均为明确代理假设。主表预留未知指令 2,048 token、chat 格式 64 token，逐题实际编码问题与选项；保留完整返回条目的有序前缀，既不截断正文，也不跳过大条目。候选序列化为 `Memory {rank}:` 后接原始返回 content 和换行。

按 [官方 API 规则](https://agentmemoryleaderboard.ai/api-guide)，128k 窗口扣除 8,192 输出和 2,048 安全空间得到 117,760 输入 token；该输入上限仍包含 prompt、问题与选项。本表实际 memory 上限取 `min(指定预算, 117760 − 问题/选项代理 − 指令预留 − chat预留)`，所以“输入窗口”并不表示可以塞入 117,760 token 的纯记忆。实际官方 prompt/末条目处理未知，生产结果可能不同。

| 数据 | 记忆预算 | 词法 All/Macro | 混合 All/Macro | 混合新增/丢失题 |
|---|---:|---:|---:|---:|
| Lo6 | 16k | 80.44/84.13 | 82.08/85.45 | +38/−29 |
| Lo6 | 32k | 87.02/90.13 | 89.21/92.49 | +30/−18 |
| Lo6 | 64k | 89.58/92.55 | 92.14/95.08 | +19/−5 |
| Lo6 | 输入窗口代理 | 89.58/92.55 | 92.14/95.08 | +19/−5 |
| LME60 | 16k | 76.67/86.25 | 80.00/88.39 | +4/−2 |
| LME60 | 32k | 85.00/91.72 | 85.00/94.25 | +2/−2 |
| LME60 | 64k | 90.00/94.33 | 95.00/98.22 | +3/−0 |
| LME60 | 输入窗口代理 | 91.67/95.17 | 95.00/98.22 | +2/−0 |

两种编码、三档未知指令预留（512/2,048/8,192）的网格中，两数据集 All/Macro 均未因这些假设而改变。输入窗口代理没有删除当前返回条目；两种编码下完整响应最大值 Lo6 为 52,780 token，LME60 为 72,330 token。这是当前响应的稳定性，不能确认未知官方序列化完全一致。完整 JSON 同时保留所有预算、token 使用量和配对得失。

## 服务输出上限与下一次固定实验

本轮 server 冻结版本仍有总计 250,000 字符的 whole-item cap 和单个邻接窗口 16,000 字符限制。LME60 词法平均返回 75.53 条、215,064 字符，混合平均 91.37 条、245,113 字符；o200k_base 全文平均约 58,745/67,064 token。条目不足 100 单独不能证明总 cap 触发，候选数也可能不足；但当前返回远低官方输入窗口，服务 cap 可能提前舍弃可容纳的证据，离线重评分无法恢复这些来源。

最有价值的后续是复用这两个完整 LME60 数据库，只重跑同样查询的 Search，将输出 cap 提高后按同 token 预算再次比较。保持模型、权重、上下文和查询集合不变；新 run ID 记录工程源 hash，并保留 original ingestion_run_id 重建 user/source 命名空间。无需重新 4,095 次 Add 或 embedding。此报告未执行该新实验，也不推定提高 cap 必然提高回答成绩。

## 适配器修复与失败审计

原 LME60 词法 run `20261005T130751Z-54713212` 有两个 ingestion 失败：同一历史内重复原生 session_id 使旧适配器 request_id 碰撞，服务器正确返回 409。该旧报告仍保留 60 题分母（58 次 Search 成功），不覆盖、不删失败。原 E5 run `20261005T130857Z-4b1c2d99` 在诊断后安全停止，11 个历史已完成写入，尚未进入 Search；partial audit、数据库和源码快照保留，不与完整方法作比较。

修复 `session-occurrence-v2` 使同一历史内每个会话 occurrence/分块的 source/request ID 唯一且重试稳定；gold 以旧 sid、position、完整来源正文唯一映射，歧义时拒绝；23 项离线回归通过。LME 两个候选在新 dev root 独立完整重跑。Lo6 没有重复 session_id，新旧 history/query/gold 文件逐字节相同、旧 Add/source/id_lookup 映射相同；Lo 两个检索 run 保持旧执行事实，只做兼容来源版本的离线 token 复核，指标全部一致，不称作新适配器重跑。

## 复现与资源边界

离线 E5 固定模型 `intfloat/multilingual-e5-small`、revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`、MIT 许可、384 维；query/passage 前缀、L2 归一化、384-token 分段、64-token overlap、512 模型 maxseq。线程 2、batch 8、推理并发 1，完整配置见 JSON；没有外部 embedding 调用。模型卡见 [Hugging Face 官方仓库](https://huggingface.co/intfloat/multilingual-e5-small)。

只读索引审计确认 10 个固定模型文件的 manifest 校验一致；Lo6 为 3,616 messages/3,616 segments，LME60 为 29,242 messages/39,374 segments，均为匹配 fingerprint 的 384 维 float32，16 个范数样本通过。独立审计文件 hash 在 aggregate JSON 中记录。

四个正式比较 run：

| 数据 | 词法 run | 混合 run | Add/Search 每候选 |
|---|---|---|---:|
| Lo6 | `20261005T130717Z-61de7412` | `20261005T130703Z-626026a8` | 245/547 |
| LME60 | `20261005T132705Z-70916815` | `20261005T132713Z-ce592f8e` | 4095/60 |

两个 LME run 都保持 server 起止 SHA `5a2d049d857ebc00ee7af51877af8b91681ba952eb8f2fee1d22a817dd9ec0c9`、semantic SHA `00d18e0e4d96d383b5cc6ff0302f17e3d97d4afe26b1d9658ca6d6e128179a4f`。LME 新适配器 eval_prepare/retrieval/compare SHA 为 `61989dc2897f8c4c97e10ced3c6da75d11019074e3c92e8907b980c1cbd39ed2` / `02dec9d3f77898aa6f09a8297c55b0dc03e91c0f46a6c4ba40e0ea32a791c3e8` / `63a130c94c835405cf6ec41d3568d5eb0e0f581e7bcf8cb951e097b485a29e5f`。Lo 原 run 的对应源 hash、数据 hash、原始响应 hash 和 token helper hash 均在 aggregate JSON 中逐 run 记录。

本地 Mac 诊断耗时 Lo6 词法/混合约 13.32/45.32 秒，LME60 约 117.77/1729.80 秒；部分进程并行且环境不同于云机，不能用这些数证明纯串行方法成本、最低云资源或 16 并发验收。正式混合方法仍需 Linux 容量/网络测试。正式任务数据及索引生命周期由提交与部署说明约束；这里保留的是公开 dev 复现数据。

公开 aggregate 不含历史原文、题目、答案、个人路径或单条 history/query 标识；原始数据、检索 JSONL、DB、进程日志保留在 ignored 本地文件中。此前 RESULTS/TOKEN_BUDGETS 报告均未覆盖。

[完整 aggregate JSON](EXPANDED_DEV.json) · [官方 tiktoken](https://github.com/openai/tiktoken) · [公开 Answer 模型配置](https://github.com/AML-memory/agent-memory-leaderboard/blob/main/api_config.py)
