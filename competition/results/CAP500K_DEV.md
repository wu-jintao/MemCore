# 输出 cap500k：公开 dev Search-only 复用实验

固定 LME60 的检索代理支持将 whole-item 总字符上限从 250k 提高到 500k：在相同输入窗口 token 预算下，等权 E5 混合的完整证据召回由 95.00% 提升到 96.67%（新增 1 题、丢失 0），词法仍为 91.67%。正式候选倾向等权 E5 context1 + cap500k；仍须通过 Linux 云端容量、网络接口和最终配置验收，召回代理不是回答准确率或官方 Overall。

## 固定范围与唯一改动

复用 [扩展公开 dev 比较](EXPANDED_DEV.md) 中已完成的两个独立 LME60 数据库。样本、原文、向量、模型 revision/fingerprint、权重、context1、top_k=100 均保持，两个新 run 各只发 60 次 Search，Add=0，没有重新生成向量。SQLite 备份后的原消息 ID/正文、请求和向量逻辑 identity 一致；原数据库物理 SHA 前后未变。私有服务快照唯一 AST 改动为 250000→500000，SHA `c1ce11b330ca543894bbb413c3828f8106049a44c3beb46429bda036ddcdf020`。

两候选 60/60 Search 全部成功，服务/端口已关闭。新 run ID 仅识别本次 Search，user/source 使用 original ingestion_run_id：

| 候选 | 新 Search-only run | 原 ingestion run |
|---|---|---|
| 词法 context1 | `20261005T140351Z-searchonly-219a1d76` | `20261005T132705Z-70916815` |
| E5 等权 context1 | `20261005T140352Z-searchonly-5260e89d` | `20261005T132713Z-ce592f8e` |

本轮只处理相同固定前 60 个公开 dev 历史，没有 validation/heldout、官方 Smoke/Full、付费模型或 Answer/Judge 调用。问题/选项/gold 仍只用于离线评分，不进入 Add。

## 相同 token 预算下两个候选

All 是一题的全部 gold 证据 turn 完整命中率；Macro 是各题 turn 召回均值。包装 metadata/session 命中不代替完整正文，失败保留在请求分母。表中数值为百分比，配对得失相对同 cap500k 的词法。

| 记忆预算 | 词法 All/Macro | E5 混合 All/Macro | 混合新增/丢失题 |
|---|---:|---:|---:|
| 16k | 76.67/86.25 | 80.00/88.39 | +4/−2 |
| 32k | 85.00/91.72 | 85.00/94.25 | +2/−2 |
| 64k | 90.00/94.33 | 95.00/98.22 | +3/−0 |
| 110k | 91.67/95.17 | 96.67/98.56 | +3/−0 |
| 输入窗口代理 | 91.67/95.17 | 96.67/98.56 | +3/−0 |

## E5 自身 cap 增量

| 记忆预算 | 原 cap250k All/Macro | 新 cap500k All/Macro | 新增/丢失题 |
|---|---:|---:|---:|
| 16k | 80.00/88.39 | 80.00/88.39 | +0/−0 |
| 32k | 85.00/94.25 | 85.00/94.25 | +0/−0 |
| 64k | 95.00/98.22 | 95.00/98.22 | +0/−0 |
| 输入窗口代理 | 95.00/98.22 | 96.67/98.56 | +1/−0 |

E5 gold 完整 turn 命中数从 122/126 提高到 123/126；source-verified Macro 与完整正文 Macro 均为 98.56%，没有无来源歧义计分。16k/32k/64k 的 E5 完整证据得失均为 +0/−0；改善出现在较大预算。LoCoMo 沿用此前六个会话的固定两候选事实，本次没有新增 Lo 检索实验。

## token 代理与输出量

使用官方 tiktoken 0.14.0：主表 o200k_base，另检查 cl100k_base；实际编码原问题/选项，未知指令预留 512/2,048/8,192、chat 格式 64。whole-item 有序前缀不截断正文、不跳过条目。按 [官方 API 规则](https://agentmemoryleaderboard.ai/api-guide) 的 117,760 输入 token 预算，实际 memory 上限还扣除问题、选项与预留。生产 Answer 模型/prompt 未公开固定，因此这些序列化/编码均为明确代理，不能声称精确复现官方 prompt。

两种编码和三档指令预留均未改变 All/Macro；输入窗口代理没有删除当前返回条目。所有编码下全文最大为 88,520 token。主 o200k_base 词法平均/最大约 63,713/87,765 token，E5 为 74,144/88,055 token。cap500k 词法平均返回 81.12 条、233,555 字符，E5 平均 100 条、271,300 字符；较大输出可能增加无关证据和网络流量，未运行 Answer/Judge 来衡量这些影响。

## 可复现性与界限

完整 aggregate 保存两个方法的每档 token 指标、E5 的 cap 增量配对、原 ingestion/source 和新工程 hash、备份逻辑 identity、原数据库未变化证明。token helper 用 ingestion_run_id 和声明的 session-occurrence-v2 映射重新计分；若盲用新 Search run ID 重建来源，会使来源核验失真。当前 helper SHA `bed152129da7f1c792f0abc2780ecf25c51285ba7740ac915a2b5e11215077b2`。

这只是一个固定公开 dev cap 工程消融：60 个历史未必代表全部基准，新增 1 题不构成通用或统计显著性结论；也无法证明官方 Answer 不受噪声干扰。Search-only 耗时不能视作 Add、重新 embedding 或云端并发性能。没有开始额外权重/模型搜索。原报告、完整 DB、原始检索和服务快照保留；公开 aggregate 不含题目、历史原文、答案、私人路径或单条 source ID。

[完整 aggregate JSON](CAP500K_DEV.json) · [此前冻结250k实验](EXPANDED_DEV.md) · [官方 tiktoken](https://github.com/openai/tiktoken)
