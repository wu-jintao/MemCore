# RRF 词法 rank tie-break：公开 dev Search-only 比较

新通用 tie-break 没有改变当前匹配 token 预算下的完整证据召回：Lo547 输入窗口代理仍为 92.14%，LME60 仍为 96.67%，16k/32k/64k/110k/窗口代理的逐题 All 配对均为 +0/−0。唯一观察到的条目预算收益是 LME60 原始 k=10 的 All 从 66.67% 提升到 68.33%（新增 1 题、丢失 0）；这不是大窗口或官方成绩提高。

## 实際改动与比较边界

原 RRF 同分只按 stable message_id 排序；新规则先按词法 rank、再按 stable message_id 排序，缺少词法 rank 视为无穷大。RRF 分数、constant60、词法/语义权重均未改。它是通用的词法 rank 优先规则，查询和数据集无分支，不是词法/向量对称 tie-break。固定的排名准则可以减少同分时对 opaque source ID 排序的依赖；这批 dev 数据不能证明回答模型或整个基准获益。

固定样本、模型 revision/fingerprint、context1 和原始索引，两组新 run 只有 Search、Add=0。原始数据库不变、SQLite 备份逻辑 identity 一致，新服务和端口已清理，没有 validation/heldout、Answer/Judge、付费模型或官方 Smoke/Full。

| 数据 | 基线 | 新 Search-only run | 直接消融 |
|---|---|---|---|
| Lo6 / 547题 | 原 E5 `20261005T130703Z-626026a8` | `20261005T142025Z-searchonly-ccdd9f0e` | 原执行cap250k；两版547题都返回100条且全文字符均<250k，cap未触发，差异可归tie |
| LME60 / 60题 | cap500k E5 `20261005T140352Z-searchonly-5260e89d` | `20261005T142027Z-searchonly-29c1ea40` | 两版均cap500k；独立AST核对除_fuse以外相同 |

原 Lo run 仍是旧250k版本执行事实，本次不称作原run500k重跑；LME 不以250k作tie基线，避免把此前cap收益误算为tie收益。两组新来源命名空间均使用原 ingestion_run_id，分别为 `20261005T130703Z-626026a8` 和 `20261005T132713Z-ce592f8e`。

## 相同 token 预算

All/Macro 均为百分比。All 要求完整覆盖一题全部gold证据turn；Macro为各题turn召回均值。失败仍在请求分母，source正文计分不把metadata或session命中算作完整证据。

| 数据 | 记忆预算 | 原规则 All/Macro | 新规则 All/Macro | 新增/丢失All题 |
|---|---|---:|---:|---:|
| Lo547 | 16k | 82.08/85.45 | 82.08/85.45 | +0/−0 |
| Lo547 | 32k | 89.21/92.49 | 89.21/92.49 | +0/−0 |
| Lo547 | 64k | 92.14/95.08 | 92.14/95.08 | +0/−0 |
| Lo547 | 输入窗口代理 | 92.14/95.08 | 92.14/95.08 | +0/−0 |
| LME60 | 16k | 80.00/88.39 | 80.00/88.39 | +0/−0 |
| LME60 | 32k | 85.00/94.25 | 85.00/94.25 | +0/−0 |
| LME60 | 64k | 95.00/98.22 | 95.00/98.22 | +0/−0 |
| LME60 | 输入窗口代理 | 96.67/98.56 | 96.67/98.56 | +0/−0 |

使用同一官方 tiktoken 0.14.0：主表o200k_base，cl100k_base敏感性；原问题/选项实际编码，未知指令预留512/2,048/8,192、chat64，保留whole-item有序前缀。输入窗口代理按117,760扣除问题/选项与预留，实际官方模型/prompt仍未知。两种编码和三档指令预留没有改变All/Macro；当前窗口代理无条目裁切，Lo这对E5响应最大51,894token，LME最大88,520token。

All逐题配对无得失、Macro聚合一致不意味着所有原始排名、每题部分证据或下游回答完全相同；这里只陈述经过离线检查的指标。Lo各k5/10/20/50/100的All无新增/丢失；LME只在k10新增1题，其余这些k无新增/丢失。

## 来源与正式版本界限

私有method snapshot server SHA为 `319e263b05cb775873f3d300b6f5a47fae43fc98f6886defb2886173d7e6bdd6`；新_fuse AST SHA为 `91d91cd8af28fe5f2a1f57ffe3aa3c78956abfc9e0bd09b07e70e707a3e2f465`。LME既有500k snapshot SHA为 `c1ce11b330ca543894bbb413c3828f8106049a44c3beb46429bda036ddcdf020`；独立读取两份runtime源确认SHA匹配summary，AST只存在_fuse差异。两份method snapshots使用semantic SHA `00d18e0e4d96d383b5cc6ff0302f17e3d97d4afe26b1d9658ca6d6e128179a4f`，没有移植后续HTTP网络处理或容量错误分类改动。

因此这份结果检验的是排名规则，不能代替最终canonical代码的Linux模型容量、公网接口、同步写入、恢复和Smoke验收。新的永久输入超限→422/临时故障→503是独立契约修复，不把它混作此处召回实验的评分来源。

完整aggregate含原始/新source与响应hash、原ingestion命名空间、备份identity、各条目/token预算和配对计数；公开文件不含题目、历史原文、答案或私人路径。旧EXPANDED_DEV、CAP500K_DEV和原始结果均未覆盖。无需为这次tie结果再新增模型或权重搜索；正式候选的主要证据仍来自等权E5+context1+cap500k的大预算比较，不能把tie本身称作已证实的大窗口提分。

[完整aggregate JSON](TIE_DEV.json) · [cap500k消融](CAP500K_DEV.md) · [原扩展dev比较](EXPANDED_DEV.md) · [官方API预算](https://agentmemoryleaderboard.ai/api-guide) · [官方tiktoken](https://github.com/openai/tiktoken)
