#!/usr/bin/env python3
"""Offline token-prefix evidence analysis of completed PUBLIC DEV runs only.

No service, Answer, Judge or embedding calls are made. The only tokenizer is the
official open-source tiktoken package. Production Answer model/prompt identifiers
are not publicly fixed, so the default encoding is an explicit proxy assumption.
Outputs contain aggregate metrics and hashes, never questions or source text.
"""
import argparse
from collections import Counter, defaultdict
import importlib.metadata
import json
from pathlib import Path
import statistics

from eval_prepare import DEFAULT_ROOT, LEGACY_SOURCE_MAPPING_VERSION, file_hash, normalized
from eval_retrieval import GRADER_VERSION, aggregate, evidence_hits, requests_for_history

ANSWER_INPUT_TOKENS = 117760
DEFAULT_BUDGETS = (16000, 32000, 64000, 110000, ANSWER_INPUT_TOKENS)


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def token_count(encoding, text):
    # Source strings resembling special-token markers remain ordinary evidence.
    return len(encoding.encode(text, disallowed_special=()))


def question_text(query):
    text = "Question:\n" + query["query"] + "\n"
    if query.get("options"):
        text += "Options:\n" + "\n".join(query["options"]) + "\n"
    return text


class TokenPrefixes:
    """Keep only complete returned items, in order, under an exact proxy count.

    Candidate serialization is our declared proxy, not a reconstructed production
    prompt: 'Memory {rank}:\n' + original returned content + '\n\n'. Every source
    wrapper and neighboring turn already in content consumes the token budget.
    Prefix counts are cached; source text is never cut or joined for grading.
    """
    def __init__(self, data, encoding, k=100):
        self.data = data[:k]
        self.encoding = encoding
        self.text = "".join("Memory " + str(i + 1) + ":\n" + item["content"] + "\n\n"
                            for i, item in enumerate(self.data))
        self.ends = [0]
        end = 0
        for i, item in enumerate(self.data):
            end += len("Memory " + str(i + 1) + ":\n" + item["content"] + "\n\n")
            self.ends.append(end)
        self.counts = {0: 0}

    def count(self, n):
        if n not in self.counts:
            self.counts[n] = token_count(self.encoding, self.text[:self.ends[n]])
        return self.counts[n]

    def select(self, budget):
        if budget is None:
            n = len(self.data)
        elif budget <= 0:
            n = 0
        else:
            # Nonempty, labeled candidate units make these whole-item prefixes
            # increasing. Do not skip a large item to select a later candidate.
            low, high = 0, len(self.data)
            while low < high:
                mid = (low + high + 1) // 2
                if self.count(mid) <= budget:
                    low = mid
                else:
                    high = mid - 1
            n = low
        return self.data[:n], self.count(n)


def percentiles(values):
    ordered = sorted(values)
    if not ordered:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {"mean": statistics.mean(ordered), "p50": statistics.median(ordered),
            "p95": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)], "max": ordered[-1]}


def load_runs(run_directories):
    groups = defaultdict(list)
    for folder in run_directories:
        summary = json.loads((folder / "summary.json").read_text())
        # Gate before opening retrieval data: validation/heldout are out of scope.
        if summary.get("split") != "dev":
            raise ValueError("Token comparison only consumes PUBLIC dev runs")
        if summary.get("server_source_changed_during_run"):
            raise ValueError("Server changed during a run; audit before comparison")
        records = list(jsonl(folder / "retrieval.jsonl"))
        if len(records) != summary["question_count"]:
            raise ValueError("Incomplete retrieval result file")
        query_ids = {record["query_id"] for record in records}
        if len(query_ids) != len(records):
            raise ValueError("Duplicate query IDs in a result file")
        groups[summary["dataset"]].append((folder, summary, records))
    for runs in groups.values():
        _, reference, reference_records = runs[0]
        ids = {record["query_id"] for record in reference_records}
        for _, summary, records in runs:
            if {record["query_id"] for record in records} != ids:
                raise ValueError("Runs must contain the exact same development questions")
            if summary["configuration"]["prepared_data"] != reference["configuration"]["prepared_data"]:
                raise ValueError("Runs use different prepared data")
            if summary["configuration"]["top_k"] != 100 or summary["configuration"]["character_budget"] != 0:
                raise ValueError("This analysis requires untrimmed top_k=100 development responses")
    return groups


def prepared_inputs(data_root, dataset, runs):
    reference = runs[0][1]
    folder = data_root / "prepared" / dataset / "dev"
    for name in ("histories", "queries", "gold"):
        expected = reference["configuration"]["prepared_data"]["files"][name]["sha256"]
        if file_hash(folder / (name + ".jsonl")) != expected:
            raise ValueError("Prepared development data has changed")
    ids = {record["query_id"] for record in runs[0][2]}
    labels = {row["query_id"]: row for row in jsonl(folder / "gold.jsonl") if row["query_id"] in ids}
    queries = {row["query_id"]: row for row in jsonl(folder / "queries.jsonl") if row["query_id"] in ids}
    if set(labels) != ids or set(queries) != ids:
        raise ValueError("Missing prepared query/gold mapping")
    sample_ids = {row["sample_id"] for row in labels.values()}
    histories = {row["sample_id"]: row for row in jsonl(folder / "histories.jsonl") if row["sample_id"] in sample_ids}
    if set(histories) != sample_ids:
        raise ValueError("Missing development history")
    frequencies = {}
    for sid, history in histories.items():
        texts = {normalized(unit["text"]) for gold in labels.values() if gold["sample_id"] == sid
                 for unit in gold["gold_units"]}
        frequencies[sid] = Counter({text: sum(text in normalized(message["content"])
                                            for session in history["sessions"] for message in session["messages"])
                                   for text in texts})
    return labels, queries, histories, frequencies


def analyze(run_directories, data_root, encodings, budgets, instruction_reserves, chat_overhead):
    groups = load_runs(run_directories)
    result = {
        "status": "public_dev_token_proxy_not_official_aml", "grader_version": GRADER_VERSION,
        "tiktoken_version": importlib.metadata.version("tiktoken"),
        "official_answer_model_confirmed": False,
        "assumptions": {
            "main_encoding": encodings[0].name,
            "main_model_family_proxy": "gpt-4o / gpt-4o-mini" if encodings[0].name == "o200k_base" else "explicit encoding",
            "sensitivity_encodings": [encoding.name for encoding in encodings[1:]],
            "answer_input_tokens_including_question_options_instructions": ANSWER_INPUT_TOKENS,
            "instruction_token_reserves": instruction_reserves, "chat_format_token_reserve": chat_overhead,
            "question_options_serialization": "Question:\\n{original_query}\\n + optional Options:\\n{original_options joined by newline}\\n",
            "candidate_serialization": "Memory {1-based rank}:\\n{exact returned content}\\n\\n",
            "memory_budget_rule": "min(requested memory tokens, 117760 - query/options proxy tokens - instruction reserve - chat reserve)",
            "cut_rule": "Keep complete returned item prefix; never truncate source text or skip a candidate",
            "requested_memory_token_budgets": budgets,
        },
        "source_urls": ["https://agentmemoryleaderboard.ai/api-guide",
                        "https://raw.githubusercontent.com/AML-memory/agent-memory-leaderboard/main/api_config.py",
                        "https://github.com/openai/tiktoken"],
        "code_sha256": {name: file_hash(Path(__file__).parent / name)
                        for name in ("eval_token_budget.py", "eval_retrieval.py", "eval_prepare.py")},
        "limitations": [
            "Production Answer model and exact prompt/serialization are not disclosed by the public model configuration; o200k_base is an assumption, not verified official matching.",
            "Fixed instruction/chat reserves and whole-item prefix boundaries are proxy choices; actual production packing or a partially retained final item may differ.",
            "This only regrades explicitly selected existing PUBLIC dev runs; no validation/heldout, Answer/Judge, paid request, embedding or retrieval rerun.",
            "Evidence recall cannot measure answer accuracy, distraction from irrelevant evidence, or official Overall.",
            "Question observations in one LoCoMo conversation are correlated; no significance or general superiority is claimed.",
            "Service-side whole-item and neighbor-window limits constrain what is available; declared whole-item caps are recorded per run and excluded content cannot be recovered offline.",
        ],
        "datasets": [],
    }
    for dataset, runs in sorted(groups.items()):
        labels, queries, histories, frequencies = prepared_inputs(data_root, dataset, runs)
        dataset_result = {"dataset": dataset, "split": "dev", "question_count": len(labels),
                          "history_count": len(histories), "runs": []}
        baseline = {}
        for run_index, (folder, summary, records) in enumerate(runs):
            # Search-only replays retain the original Add/source namespace.
            ingestion_run_id = summary.get("ingestion_run_id", summary["run_id"])
            source_mapping_version = summary["configuration"].get("source_mapping_version", LEGACY_SOURCE_MAPPING_VERSION)
            sources_by_sample = {sid: requests_for_history(history, ingestion_run_id, source_mapping_version)[2:]
                                 for sid, history in histories.items()}
            run_result = {"run_id": summary["run_id"], "model_label": summary["model_label"],
                          "ingestion_run_id": ingestion_run_id,
                          "source_mapping_version": source_mapping_version,
                          "service_whole_item_character_cap": summary["configuration"].get("service_whole_item_character_cap"),
                          "summary_sha256": file_hash(folder / "summary.json"),
                          "retrieval_sha256": file_hash(folder / "retrieval.jsonl"), "metrics": {}}
            for encoding in encodings:
                prefixes = {record["query_id"]: TokenPrefixes(record["retrieved"], encoding)
                            for record in records if record["status"] == "ok"}
                question_tokens = {qid: token_count(encoding, question_text(query)) for qid, query in queries.items()}
                run_result.setdefault("query_options_tokens", {})[encoding.name] = percentiles(list(question_tokens.values()))
                for instruction_reserve in instruction_reserves:
                    for requested in [None] + budgets:
                        key = encoding.name + "/instructions" + str(instruction_reserve) + "/memory" + ("unlimited" if requested is None else str(requested))
                        rows, question_all = [], {}
                        used_tokens, limits, trimmed = [], [], 0
                        for record in records:
                            qid, sid = record["query_id"], record["sample_id"]
                            if record["status"] != "ok":
                                rows.append({"status": record["status"]})
                                question_all[qid] = False
                                continue
                            limit = None if requested is None else max(0, min(requested, ANSWER_INPUT_TOKENS - question_tokens[qid] - instruction_reserve - chat_overhead))
                            prefix, count = prefixes[qid].select(limit)
                            sources, source_ids = sources_by_sample[sid]
                            metrics = {**evidence_hits(prefix, labels[qid], sources, source_ids, frequencies[sid]),
                                       "returned_items": len(prefix), "returned_characters": sum(len(item["content"]) for item in prefix)}
                            rows.append({"status": "ok", "metrics": {"100": metrics}})
                            question_all[qid] = metrics["recall_all"]
                            used_tokens.append(count)
                            if limit is not None:
                                limits.append(limit)
                                trimmed += len(prefix) < len(prefixes[qid].data)
                        scored = aggregate(rows, 100, len(records))
                        scored["memory_tokens"] = percentiles(used_tokens)
                        scored["effective_memory_token_budget"] = percentiles(limits)
                        scored["questions_with_removed_items"] = trimmed
                        if run_index == 0:
                            baseline[key] = question_all
                        else:
                            first = baseline[key]
                            scored["paired_complete_coverage_changes_vs_first_run"] = {
                                "gained_questions": sum(question_all[qid] and not first[qid] for qid in question_all),
                                "lost_questions": sum(first[qid] and not question_all[qid] for qid in question_all)}
                        run_result["metrics"][key] = scored
            dataset_result["runs"].append(run_result)
        result["datasets"].append(dataset_result)
    return result


def markdown(result):
    lines = ["# 公开开发集离线 token 前缀诊断", "",
             "只复用明确选定的既有开发响应；未调用检索、embedding、Answer、Judge 或正式评测。报告是证据召回代理，不是回答准确率或官方成绩。", "",
             "官网声明 Answer 可用 117,760 输入 token，包含问题、选项、指令。公开模型配置未披露 production Answer 标识；主分析使用官方 tiktoken 的 `o200k_base`（假设 GPT-4o 家族），另以 `cl100k_base` 检查敏感性。不能声称已精确匹配官方模型或提示词。", "",
             "默认预留未知指令 2,048 token 和聊天格式 64 token，问题与选项按公开原文实际编码另扣；并列 512、8,192 指令预留作敏感性分析。候选代理格式为 `Memory {rank}:` 标签与完整返回正文，原有 source/邻接包装均计入；保留整条排序前缀，超出时停止，不截断正文、不跳选。117,760 一档扣除了上述非证据开销，不是给记忆独占整个输入窗口。", ""]
    main = result["assumptions"]["main_encoding"]
    default_reserve = 2048 if 2048 in result["assumptions"]["instruction_token_reserves"] else result["assumptions"]["instruction_token_reserves"][0]
    shown = result["assumptions"]["requested_memory_token_budgets"] + [None]
    headings = ["输入窗口代理上限" if budget == ANSWER_INPUT_TOKENS else "{:g}k token".format(budget / 1000)
                for budget in shown[:-1]] + ["返回全文"]
    for dataset in result["datasets"]:
        lines += ["## " + dataset["dataset"], "", str(dataset["question_count"]) + " 题 / " + str(dataset["history_count"]) + " 个历史。每格为 All / Macro；括号为相对首个词法基线新增 / 丢失完整覆盖题。", "",
                  "| 方法 | " + " | ".join(headings) + " |",
                  "| --- | " + " | ".join("---" for _ in headings) + " |"]
        for run in dataset["runs"]:
            cells = []
            for budget in shown:
                key = main + "/instructions" + str(default_reserve) + "/memory" + ("unlimited" if budget is None else str(budget))
                metric = run["metrics"][key]
                cell = "{:.2f}% / {:.2f}%".format(100 * metric["full_turn_recall_all"], 100 * metric["full_turn_recall_macro"])
                paired = metric.get("paired_complete_coverage_changes_vs_first_run")
                if paired:
                    cell += " (" + str(paired["gained_questions"]) + "/" + str(paired["lost_questions"]) + ")"
                cells.append(cell)
            lines.append("| " + run["model_label"].replace("|", "/") + " | " + " | ".join(cells) + " |")
        lines += ["", "| 方法 | 全文 token 均值 / P95 / 最大 | 窗口上限剔除条目的题数 |", "| --- | --- | --- |"]
        for run in dataset["runs"]:
            prefix = main + "/instructions" + str(default_reserve) + "/memory"
            whole = run["metrics"][prefix + "unlimited"]["memory_tokens"]
            clipped = run["metrics"][prefix + str(ANSWER_INPUT_TOKENS)]
            lines.append("| " + run["model_label"].replace("|", "/") + " | " + " / ".join("{:,.0f}".format(whole[key]) for key in ("mean", "p95", "max")) + " | " + str(clipped["questions_with_removed_items"]) + " |")
        lines.append("")
    encoding_metric_differences = 0
    instruction_metric_differences = 0
    largest_return_tokens = 0
    maximum_window_removed_questions = 0
    for dataset in result["datasets"]:
        for run in dataset["runs"]:
            for encoding in [main] + result["assumptions"]["sensitivity_encodings"]:
                prefix = encoding + "/instructions" + str(default_reserve) + "/memory"
                largest_return_tokens = max(largest_return_tokens, run["metrics"][prefix + "unlimited"]["memory_tokens"]["max"] or 0)
                for reserve in result["assumptions"]["instruction_token_reserves"]:
                    window_key = encoding + "/instructions" + str(reserve) + "/memory" + str(ANSWER_INPUT_TOKENS)
                    maximum_window_removed_questions = max(maximum_window_removed_questions,
                                                           run["metrics"][window_key]["questions_with_removed_items"])
                    for budget in result["assumptions"]["requested_memory_token_budgets"]:
                        key = encoding + "/instructions" + str(reserve) + "/memory" + str(budget)
                        score = run["metrics"][key]
                        main_score = run["metrics"][main + "/instructions" + str(reserve) + "/memory" + str(budget)]
                        reserve_score = run["metrics"][encoding + "/instructions" + str(default_reserve) + "/memory" + str(budget)]
                        metric_names = ("full_turn_recall_all", "full_turn_recall_macro")
                        encoding_metric_differences += any(score[name] != main_score[name] for name in metric_names)
                        instruction_metric_differences += any(score[name] != reserve_score[name] for name in metric_names)
    lines += ["## 选择含义", "",
              "这批响应在两种编码下的全文最大值为 {:,} token。三档未知指令预留下，输入窗口代理上限最多使 {} 道题删除后续条目。因而这批数据的官方大窗口推断主要受检索质量和下游噪声影响，较小字符预算表不能直接代表窗口上限。".format(largest_return_tokens, maximum_window_removed_questions), "",
              "方法选择应结合表中的完整证据 All、Macro 和配对新增/丢失，比较相同 token 预算。召回代理未执行 Answer/Judge，不能据此确认回答正确率或官方成绩；正式选择还需要独立验证和部署容量验收。", "",
              "## 敏感性与适用范围", "",
              "本次预算网格中，两种编码造成 All/Macro 变化的聚合单元数为 {}，不同指令预留造成变化的单元数为 {}。这支持当前小集预算结论对这些假设稳定，不证明未知官方 prompt 一定相同。".format(encoding_metric_differences, instruction_metric_differences), "",
              "完整 JSON 保存两种编码、三档未知指令预留和各 token 预算的聚合指标与配对得失，且记录源文件/原始响应哈希。API 错误仍计入请求分母；指标按完整来源正文 v2 重新计算，session 命中不代替证据 turn。", "",
              "token 和字符不是同一个单位。这里不能恢复服务此前因 whole-item 总上限或邻接窗口限制排除的来源；如运行摘要声明总字符上限，JSON 按 run 记录其值，未声明则保留 null。也无法衡量大量无关证据对回答模型的干扰。LoCoMo 同会话题目彼此相关，样本量以表中历史数为准；需要固定公开开发分组及独立验证来检验候选。", "",
              "[官网 API 预算规则](https://agentmemoryleaderboard.ai/api-guide) · [公开模型配置](https://raw.githubusercontent.com/AML-memory/agent-memory-leaderboard/main/api_config.py) · [官方 tiktoken](https://github.com/openai/tiktoken)", ""]
    return "\n".join(lines)


def self_test(encodings):
    for encoding in encodings:
        data = [{"id": "a", "content": "First complete fact."},
                {"id": "b", "content": "中文事实 <|endoftext|> " * 100},
                {"id": "c", "content": "Third fact."}]
        prefix = TokenPrefixes(data, encoding)
        for budget in (0, 1, prefix.count(1), prefix.count(2) - 1, prefix.count(2), prefix.count(3)):
            rows, count = prefix.select(budget)
            oracle = max(n for n in range(4) if prefix.count(n) <= budget)
            assert rows == data[:oracle] and count <= budget
        assert prefix.select(None)[0] == data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="Explicit PUBLIC dev run directories; baseline first within each dataset")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--encodings", default="o200k_base,cl100k_base")
    parser.add_argument("--memory-token-budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--instruction-reserves", default="2048,512,8192")
    parser.add_argument("--chat-overhead", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True, help="Aggregate-only JSON; Markdown uses the same basename")
    args = parser.parse_args()
    import tiktoken
    encodings = [tiktoken.get_encoding(name) for name in args.encodings.split(",")]
    budgets = [int(value) for value in args.memory_token_budgets.split(",")]
    reserves = [int(value) for value in args.instruction_reserves.split(",")]
    if not budgets or ANSWER_INPUT_TOKENS not in budgets or not reserves or min(budgets) < 1 or min(reserves) < 0 or args.chat_overhead < 0:
        parser.error("Invalid token budgets")
    self_test(encodings)
    result = analyze(args.runs, args.data_root, encodings, budgets, reserves, args.chat_overhead)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "datasets": [{"dataset": row["dataset"], "questions": row["question_count"], "runs": len(row["runs"])} for row in result["datasets"]], "tokenizer": result["tiktoken_version"]}))


if __name__ == "__main__":
    main()
