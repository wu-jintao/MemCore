# MemCore Coding v1 validation

All local validation uses public or synthetic data. No CAMBench private input,
AML evaluation request, answer model, judge, gold answer, or held-out annotation
was inspected or used.

## Tests

- 99 Python unit/integration/deployment-tool checks passed on the final candidate
  lineage (92 general/Coding checks plus 7 isolated deployment-tool checks).
- The test suites cover the original Textual behavior, semantic transport,
  query-capacity recovery, Coding features, compact schemas, public embedding
  cache integrity, dense proxy fusion, real MemoryStore public proxy replay, and
  loopback runtime recovery.

## Public proxy

- Source: public SWEContextBench Lite Parquet files.
- 300 historical engineering experiences and 99 related tasks.
- Four adversarial noise levels: 4, 16, 48, and 96 distractors.
- Dense cache: 300 document inputs, 99 query inputs, 495 complete UTF-8 segments,
  `text-embedding-v4`, 2048 dimensions.
- Model calls while evaluating cached rankings: zero.
- Final high-noise real-service result: Recall@1 76.77%, Recall@3 87.88%,
  Recall@8 96.97%, Recall@12 97.98%, and MRR 83.07%.
- At the selected 12-result / 180,000-character budget, all relevant public
  experiences were returned for 95/99 tasks, mean returned content was about
  103K characters, and p95 was about 139K characters.

## Runtime verification

The real loopback HTTP server passed 64 unique Adds at concurrency 16, 32
concurrent identical request-ID retries, 64 Searches at concurrency 16, exact
user isolation, deterministic Search equality, SIGKILL restart with identical
results, one 2048-dimensional fingerprint, and SQLite `quick_check=ok`.

## Cross-user embedding reuse

The high-noise public proxy contains 9,604 scoped memory rows but only 300 unique
public engineering experiences. Content-addressed reuse reduced document
embedding inputs to 300 across 30 cached backend operations while preserving
13,592 per-user vector rows and all retrieval metrics. The cache stores only the
fixed semantic fingerprint, exact compact-representation SHA-256, segment index,
dimension, and vector; it stores no source text and never participates directly
in Search ranking.

## Isolated cloud candidates

An earlier Coding core ran on an independent loopback-only cloud candidate with
`text-embedding-v4`, 2048 dimensions, and no AML or public route. A 16-way real
model load check completed all Adds and Searches successfully: Add p95/max was
132.91 seconds, Search p95/max was 10.02 seconds, and 32 concurrent identical
request-ID retries had p95 18.9 milliseconds. This establishes provider and
concurrency stability; it predates the final vector-reuse and request-limit core.

The final core from commit `e2ae204f3e411967e561e9df6fbedc93e90f8eb7`
was then deployed as fresh loopback-only candidate `v1-20261013-c`, pinned by
`CODING_CORE_MANIFEST.json`. A 9,437,436-byte authenticated Add succeeded, while
a declared 67,108,865-byte body was rejected with HTTP 413. The same synthetic
engineering memory was written under two users: the first Add took 0.454 seconds
and created one cache row; the second took 0.008 seconds, created no cache row,
and still created an independent scoped vector. Each user retrieved exactly its
own record and an unrelated user retrieved none. SQLite `quick_check` remained
`ok`. These timings are point checks rather than formal latency claims.

The final isolated Coding capacity configuration raises only the Coding request
body ceiling to 134,217,728 bytes. Candidate `v1-20261013-e` accepted a
68,157,747-byte synthetic Add in 39.8 seconds and rejected a declared
134,217,729-byte body with HTTP 413. The test used no public route or AML call,
retained no request body, and left SQLite `quick_check=ok`; 76.8 GB remained
available on the persistent data disk after the check. The AML documentation
allows one Add or Search request to run for up to 30 minutes.

The clean final baseline was prepared from commit
`a1078996f86d38623bec4c1fdb035e8041ef3018`. Candidate `v1-20261013-f`
started with zero request, message, vector, and reuse-cache rows. A controlled
process kill changed its PID and systemd restored health after the configured
five-second delay; `NRestarts` advanced to one, all four database counts remained
zero, and SQLite `quick_check` remained `ok`. The Textual and legacy services and
Caddy remained active with unchanged process identities during this check.

## Limits

Public proxy retrieval quality does not establish an official CAMBench score.
The disclosed formal suite size does not predict the largest single Coding
trajectory. The 128 MiB ingress check provides headroom but is not an official
CAMBench capacity measurement. No Coding Smoke or Full has been started by this
source validation.

## Publication scope

The published Coding source contains no Parquet data, proxy database, vector
cache, model credential, AML credential, evaluation input, raw per-task output,
or deployment address. Public data is separately downloaded and verified through
`CODING_PUBLIC_DATA_MANIFEST.json`; only aggregate proxy evidence is retained in
`CODING_PUBLIC_PROXY_SUMMARY.json`.
