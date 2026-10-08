# Optional public-data research tools

These modules are separate from inference. No dataset, raw source, questions,
reference labels, raw retrieval results or local model weights are bundled.
The 19 historical public aggregate/method reports in `results/` retain their
original dated configurations and bytes; they do not establish current v4 QA,
Smoke or Full. Obtain
only authorized public data using the pinned manifests; observe original data
licenses. Outputs are local research artifacts and should stay private unless
independently reduced to an appropriate aggregate.

- `eval_prepare.py` separates historical messages, queries and grader labels,
  checks pinned files and records occurrence-aware source identities.
- `eval_retrieval.py` measures explicit Add/Search runs. `eval_compare.py`,
  `eval_audit.py` and `eval_token_budget.py` check source provenance and compare
  evidence availability. Token-prefix analysis is not answer accuracy.
- `eval_v4_budget.py` creates a plan by default. `--execute` alone permits model
  requests and private credential reading. One durable token/request/deadline
  ledger hard-stops future calls on failures or exhaustion; proxies, redirects
  and implicit retries are disabled.
- `eval_v4_query_modes.py` defaults to a dry plan. It first tests eight stored
  document segments using strict cosine >=0.999999 and maximum absolute
  difference <=0.00001. Failure stops query-model calls. Its B/C query-only
  replay uses independent database copies, original user/ingestion scopes and
  cached query vectors, forbids document embedding/Add, and opens grader labels
  only after both replay arms complete. It cannot import a permissive external
  gate result. Native query variants are experiments, not runtime configuration.
- `eval_persona.py` and `eval_persona_pair.py` preserve exact history cutoffs and
  paired complete public-dev groups. `eval_answer.py`, `eval_calibrate.py`,
  `eval_validate_answers.py`, `eval_validation.py`, `eval_research.py` and
  `eval_rerank.py` remain optional independent proxy/research tools. Their presence
  does not select any reranker or LLM for the memory service.

Use each module's `--help` for exact arguments. Budgeted v4 tools require explicit
limits, a fresh private output directory and a private literal dotenv file.
Query-mode plans require a completed compatible baseline with its original
configuration, database, source ledger and prepared public-dev provenance;
these run artifacts are deliberately absent from the repository. Do not weaken
strict gates or relabel a failed probe as equivalent. Historical default paths
are local schema conventions, not included data or portable completed runs.

For already completed PUBLIC dev runs, a fixed offline comparison can use:

```sh
python3 competition/eval_compare.py --runs .local/run-a .local/run-b   --data-root .local/public-data-occurrence-v2 --ks 5,10,20,50,100   --character-budgets 0 --output .local/paired.json
python3 competition/eval_token_budget.py --runs .local/run-a .local/run-b   --data-root .local/public-data-occurrence-v2 --encodings o200k_base   --memory-token-budgets 32000,117760 --instruction-reserves 2048   --chat-overhead 64 --output .local/token-budget.json
```

The tokenizer must already be cached for a strictly offline run. The tools
compare matching query/history identities and retain full denominators;
failed/unfinished requests are not dropped. Do not interpret cached Search
latencies as live embedding-inclusive latency. Correlated conversation questions,
body-only evidence recall and local Answer/Judge results remain distinct from
official Overall and platform answer accuracy. This v1.1 source release makes no current v4 QA-completion or official
evaluation claim.
