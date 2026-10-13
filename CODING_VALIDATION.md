# MemCore Coding v1.1 validation

All validation used public or synthetic data. No CAMBench private input, AML raw
evaluation request, answer model, judge, gold answer, credential, or private user
identifier was inspected or used.

## Formal public benchmark replay

- Public source: SWEContextBench revision
  `12ad6ab14e18e9378e1e293c9edbc3f7ce43d27b`.
- 1,007 public histories and 357 public related tasks.
- Development: 99 public Lite tasks.
- Independent validation: 258 Full-minus-Lite tasks.
- Fixed cache: 1,007 document identities, 357 query identities, 1,776 complete
  segments, `text-embedding-v4`, 2048 dimensions.
- Grid selection and implementation replay made zero model, AML, answer, or judge
  calls; they reused the fixed public-only cache.

The selected method is maximum query/document segment-pair cosine with post-RRF
weight 0.5. Independent validation improved Recall@1 from 70.93% to 72.09%,
Recall@3 from 85.66% to 87.21%, Recall@8 from 95.74% to 96.12%, and MRR from
80.09% to 80.96%. Recall@12 remained 98.06%; complete related-experience coverage
remained 253/258; mean returned characters remained exactly 117,422.64.

## Implementation equivalence

The final `MemoryStore` implementation replayed all 357 Full tasks through the
real 1.58 GB Coding v1 database. All 357 tasks passed each invariant:

- final order matched an independent segment-pair/RRF implementation;
- selected evidence ID set was unchanged;
- source content was unchanged;
- returned character total was unchanged;
- v1 baseline matched the prior research harness;
- v1.1 candidate matched the prior research harness.

The database completed `PRAGMA quick_check` with `ok`. Three duplicate query-text
groups contained six task identities and small cached-vector numerical variation;
all variants produced identical baseline order, v1.1 order, and metrics.
Aggregate details and fixed hashes are in
`CODING_V11_PUBLIC_VALIDATION.json`. Per-task outputs, caches, databases, and raw
public source records remain excluded from publication.

## Tests

- 86 dependency-free core, Coding, Textual, transport, capacity, and runtime tests
  passed with the system Python.
- 8 public benchmark/cache tests passed with the optional `numpy` and `pyarrow`
  research dependencies.
- 7 isolated Coding deployment-tool tests are included in those totals and pin
  the 0.5 rerank weight, 12-result limit, 180,000-character budget, source hashes,
  private environment hashes, and systemd unit hash.

## Runtime and deployment limits

A fresh Coding v1.1 candidate was deployed on October 13, 2026 to an independent
loopback listener with a fresh database. It was not added to Caddy and made no AML
request. A bounded real-model probe completed 30/30 requests, including synchronous
visibility, idempotent retry and conflict handling, 16 concurrent Searches, and
complete user-ID isolation. Search p50 was 4.76 seconds and p95 was 17.88 seconds
in this small synthetic run.

A 10,919-byte semantic query exercised two complete `text-embedding-v4` query
segments and returned 12 evidence items in 4.00 seconds. A controlled SIGKILL
changed the process ID, advanced systemd `NRestarts` from zero to one, and restored
an identical Search response. The candidate database retained 33 messages and
vectors with `PRAGMA quick_check=ok`. The existing public Coding candidate stayed
empty, all existing MemCore services stayed active, and the Caddy configuration
hash remained unchanged.

Public proxy retrieval quality does not establish an official CAMBench score.
The cloud checks are bounded synthetic operational evidence, not Full-scale
capacity certification. No Coding Smoke or Full is claimed by this validation.
