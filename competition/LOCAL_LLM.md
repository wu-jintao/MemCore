# Local LLM development tests

`eval_answer.py` can compare two frozen public-dev retrieval runs using a
self-hosted OpenAI-compatible vLLM service. This is an independent Answer/Judge
proxy, not an official AML evaluation. Synthetic adapter tests alone do not
establish answer accuracy; actual development runs are recorded separately below.

The public repository at commit `1b8142bfe0f20f1c5218d6b554aa0012de34e504`
does disclose family-specific Answer/scoring contracts. The production model,
tokenizer, memory-to-family mapping, context budget and Streaming orchestration
remain unverified. The frozen date-aware `eval_answer.py` v3 is retained as a
diagnostic profile. The new `eval_contracts.py` / `eval_calibrate.py` profile
uses independently written instructions for the published LongMemEval/LoCoMo
time, list and preference rules, one user message, and no added question date.
It is not byte-identical to the upstream prompts. Historical rejudging changes
the measurement only and must not be reported as a memory-method improvement.
[Public contract](https://github.com/AML-memory/agent-memory-leaderboard/blob/1b8142bfe0f20f1c5218d6b554aa0012de34e504/data/longmemeval-s/pipeline.py).

## Verified H20 deployment, October 6, 2026

The development worker now serves `Qwen/Qwen3-32B` at revision
`9216db5781bf21249d130ec9da846c4624c16137`. All 26 model/tokenizer files were
checked against official Hugging Face LFS SHA-256 or Git blob SHA-1 values. The
weights are BF16 and unchanged. An isolated vLLM 0.14.1 / Transformers 4.57.6
environment reuses the worker's Torch 2.9.1 installation.

Four H20 GPUs serve the model with tensor parallelism 4; approximately 38.7 GiB
was allocated on each during the deployment check. Four other GPUs remain free
at that check. The service listens only on `127.0.0.1:18080`, uses the model alias
`aml-qwen3-32b`, and accepts one sequence at a time. It is a manually started
development process, not a persistent service with restart guarantees.

The configured total context is 65,536 tokens, using YaRN factor 2 over the
model's native 32,768 context. For this vLLM version the override is
`rope_parameters`, and the original model configuration was not edited. A
synthetic middle-needle request with 61,106 input tokens returned the expected
answer in 15.9 seconds; the tokenizer and service reported identical input
counts. This is one capacity probe, not a benchmark or a general guarantee of
long-context accuracy. See the [official model card](https://huggingface.co/Qwen/Qwen3-32B).

The corrected five-question LongMemEval development pilot uses original public
question dates, a 32,000-token memory budget, temperature-zero non-thinking
generation, and the same Qwen model for Answer and Judge. Baseline scored 2/5
and BGE-reranked evidence scored 4/5, with two gains, no losses and no failures.
The 20 requests took 101.8 seconds and reported 318,978 input / 891 output tokens.
API fees were zero; GPU and other compute costs are not included. This small
pilot is a gate for the fixed 60-question comparison, not evidence of a stable
40-point improvement.

The corrected 60-question comparison completed in 1152.15 seconds: baseline
36/60 (60.00%), BGE-reranked evidence 41/60 (68.33%), with seven gains and two
losses. One baseline answer failed the nonempty/complete-output check and stayed
in the denominator; the candidate also answered that question incorrectly.
There were 239 calls, all with usage, reporting 3,813,103 input and 12,447 output
tokens. The remaining 59 pairs were successfully judged on both sides. The
adapter's SHA was unchanged during the run, and an independent reconstruction
matched all 120 Answer prompt hashes.

Temperature zero does not guarantee identical generated text across runs. A
five-question repeat preserved the packing, status and correctness on both
sides, but only six of ten answers were byte-identical. New implementation
candidates therefore run a fresh matched baseline rather than treating an
earlier baseline score as a fixed constant.

The result is diagnostic: post-hoc review found preference-Judge inconsistencies,
so 0/4 on preference questions should not be treated as proof of failed memory
retrieval. Raw judgments remain unchanged. See
[the aggregate report](results/LOCAL_LLM_LME60_DEV.json) and
[capability coverage](results/CAPABILITY_COVERAGE.json) for category counts, limitations and
the first source-preserving evidence-grouping candidate.

Earlier no-question-date v2 pilots and an interrupted v2 60-question attempt
remain private and separate. Their scores cannot be merged with the corrected
comparison. No official AML Smoke or Full has been launched by this work.

## Public-contract calibration and broader coverage

The new independently worded single-user/no-question-date profile completed a
fresh matched LongMemEval development comparison: baseline **41/60**, BGE
**43/60**, five gains and three losses, no failures. All 240 calls reported
usage: 3,820,206 input / 6,087 output tokens, 861.75 seconds for execution.
The API fee was zero; compute costs were not accounted. These within-profile
results cannot be compared to the historical date-aware v3 totals as a memory
implementation gain. See [CALIBRATED_LME60_DEV.json](results/CALIBRATED_LME60_DEV.json).

Rejudging unchanged historical answers with the public task rules changed
36/41 to 37/45, using 119 Judge calls and no Answer calls. All five grade flips
were preference cases. The old raw grades and source remain intact; this is
measurement calibration only.

Identity auditing puts all 470 prepared public LongMemEval questions in one
development component because native history sessions are shared. There is
no independent LME validation in this manifest. LoCoMo retains complete
conversation splits: development 547/6, validation 183/2, holdout 129/2.
`eval_validate_answers.py` accepts only the full original validation partition,
verifies exact source bytes/serial-ingestion metadata and unchanged candidate
id/content multisets, and defaults to a no-network plan. Execution is bounded
at 732 requests for the matched 183-question comparison. The holdout is not
read for retrieval or generation.

The complete 183-question/two-conversation LoCoMo validation comparison has
finished: baseline **142/183 (77.60%)**, BGE **148/183 (80.87%)**, 17 gains,
11 losses and no failures. All 732 requests reported usage: 11,741,115 input /
11,886 output tokens. Generation/grading took 2621.40 seconds; the wrapper,
including planning, took 2819.62 seconds. Read-only reconstruction independently
matched all 366 Answer messages, 366 Judge messages, 732 actual prompt-token
counts and 366 complete-item prefix boundaries. The two frozen validation
conversations are separate from development; public data and the same-family
Judge still limit this result. The 129-question holdout remains untouched.
See [LOCOMO183_VALIDATION.json](results/LOCOMO183_VALIDATION.json).

A deployment probe on the worker CPU took 290.70 seconds to rerank one fixed
query's 100 candidates/157 window pairs, with 4.85 GiB peak RSS. Its four logical
CPUs span two physical cores with SMT; this is not actual ECS or concurrent
capacity. Keep BGE as a research candidate until a faster serving route is
measured. No reranker was installed in the public service.

Synthetic behavior/provenance checks passed 26/26 steps and 14/14 searches for
both grouping modes. A separate ten-case model check of the reserved default
scored 9/10, with zero failures. The user-preference answer selected the
assistant's conflicting statement even though both correctly attributed
original messages entered Answer. This is a role-attribution diagnosis,
not missing retrieval evidence, a manual regrade, or an official Streaming score.

`eval_persona.py` prepares pinned PersonaMem-v1 32k data offline: 589 questions,
20 persona groups and 222 exclusive-cutoff prefixes, with 356/122/111 questions
in dev/validation/heldout. An independent audit matched 35,918 native messages.
The native query already appears verbatim at the history end in 64 cases; the
adapter does not append CSV questions or later assistant replies. Future runs
must isolate each persona/context/cutoff user scope. No actual PersonaMem
retrieval or model score has been measured. See
[CAPABILITY_COVERAGE.json](results/CAPABILITY_COVERAGE.json).

## Evidence-grouping optimization trial

A fresh Search candidate removed the pre-reservation of lower-ranked anchors
from adjacent context. It preserved the index, original top100 anchor pool,
source bodies and scopes. With the same BGE configuration, the actual Qwen
32k prefix improved complete annotated-evidence coverage from 57/60 to 58/60.

The matched Answer/Judge trial nevertheless regressed: old grouping plus BGE
scored 41/60, emitted-source grouping plus BGE 36/60, with three gains and eight
losses. Both sides had zero failures. The 240 calls reported 3,812,599 input and
11,997 output tokens and took 1129.50 seconds. API fees were zero; compute costs
were not accounted. All input, adapter, server-state and model-manifest hashes
were unchanged. See [the aggregate report](results/PACKING_LME60_DEV.json).

The candidate was not selected as the default. The root service uses
`context_grouping="reserved"`; `--context-grouping emitted` or
`MEMORY_CONTEXT_GROUPING=emitted` enables the research alternative. All 32
service tests passed, and 64 default-response comparisons exactly matched the
permanent old Search implementation while preserving the newer date helpers.
The frozen trial files remain separate from this optional root integration.

The five-question gate had already shown a loss despite intact evidence. Its
post-hoc audit found time/event confusion and uneven treatment of extra dates
by the Judge. These observations explain what to investigate; the recorded
five- and sixty-question scores are not manually changed. A broader fixed
validation and the specified competition stack are needed before promotion.
