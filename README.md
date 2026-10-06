# MemCore

MemCore is a textual agent memory service: Add stores original conversation
messages and synchronous retrieval indexes; Search returns attributable evidence
for an answer model. It combines scoped SQLite storage, BM25, optional dense
retrieval and source-preserving context windows. The online service does not
generate final answers. Offline tools support retrieval and answer-quality
experiments on separately obtained public datasets.

This repository publishes a **research source snapshot dated October 6, 2026**
under the [MIT license](LICENSE). It includes source, tests, method attribution,
deployment templates and aggregate research reports. Dataset histories, raw
retrieval/Answer/Judge outputs, model weights and credentials are not bundled.
See [third-party notices](THIRD_PARTY_NOTICES.md) for their separate terms.

The current E5 configuration is a development baseline. The
[competition FAQ, question 05](https://agentmemoryleaderboard.ai/competition/)
requires `text-embedding-v4` for Academic embeddings and `gpt-4o-mini` for LLM
components. The optional v4 HTTP adapter has passed fake-provider tests, but a
live v4 index, quality and capacity verification remain pending. There is no
official Smoke, Full or leaderboard score. Publication alone does not establish
Academic eligibility or deployment readiness.

[Source status](SOURCE_STATUS.md) distinguishes this snapshot from the earlier
deployed E5 version. [Validation](VALIDATION.md) describes local evidence and
remaining gates; [public-network diagnostics](deploy/VALIDATION.md) retain failed
and incomplete requests as well as passes.

## Run the lexical baseline

Python 3.9+; no third-party packages are needed when embeddings are disabled.

```sh
export MEMORY_API_TOKEN='local-demo-only-change-this'
export MEMORY_EMBEDDING_PROVIDER=disabled
python3 server.py --host 127.0.0.1 --port 8080 --db .local/memory.sqlite3
```

The token above is a public local demonstration value. Use a dedicated random
secret for deployment. The server refuses to start without a token; the
`--allow-insecure` experiment flag only permits a loopback listener. The caller
uses `Authorization: Bearer <token>`. GET `/health` is unauthenticated and returns
storage status only. Never commit tokens, runtime environment files, model keys,
SSH keys, evaluation keys, or benchmark histories.

The service token authorizes the evaluation caller, not individual end users.
Anyone holding it can request any complete `user_id`; keep it restricted to the
authorized evaluation system.

## Add and Search

POST `/add` and `/search` accept UTF-8 JSON with
`Content-Type: application/json`. The request-body limit is 8 MiB. Unsupported
fields/types return 400, invalid authentication 401, incomplete authenticated
body timeout 408, conflicting retries 409, declared semantic input-capacity
exhaustion 422, retryable storage/model failure 503, and other operation failures
500. Headers have a 30-second socket deadline; authenticated body reads allow
up to 1,800 seconds between incoming reads.

```json
{
  "request_id": "request-001",
  "user_id": "run-001/user-7",
  "session_id": "session-1",
  "messages": [
    {"role": "user", "content": "I will visit Star River Books on Saturday.", "timestamp": 1735689600123}
  ]
}
```

Identifiers must be nonempty strings and are preserved exactly, including case,
Unicode and surrounding whitespace. Roles are `user` or `assistant`; textual
content must be nonempty. Optional `timestamp` is a signed 64-bit Unix-millisecond
integer; zero is valid. Missing timestamps remain missing.

Add commits the request, original messages, scoped corpus/term statistics and all
configured segment vectors in one transaction before returning:

```json
{"success":true,"request_id":"request-001","user_id":"run-001/user-7","session_id":"session-1"}
```

The idempotency key is `(complete user_id, request_id)`. An identical canonical
JSON body returns the same acknowledgement without duplicating records or
calling an embedding model again. Object-field order and formatting whitespace
do not affect equality. A changed body, session, timestamp or message order at
the same key returns 409 without replacing the committed record.

```json
{"query":"Which bookstore will I visit?","options":["Star River Books","West Books"],"user_id":"run-001/user-7","top_k":3}
```

`options` is an optional array of strings. `top_k` must be an integer from 1 to
100. Every SQL read of messages, lexical statistics or vectors matches the exact
complete `user_id`; sessions within that user can be searched together. Dataset
names, shortened IDs and other runs do not select shared memory. Blank queries
and options return `{"data":[]}`. Unknown users return no evidence.

Search returns `{"data":[{"id":"msg_…","content":"…","score":0.5}]}`.
Each content item includes a compact `[source {...}]` line and the original
message verbatim. Source information includes role, session, request, original
message index, stable ID and received sequence. Original milliseconds are
retained, with UTC time only when representable. Received sequence is an
ingestion-order tie-breaker, not a fabricated event date. Neighboring messages
retain their own source lines and IDs. Source text and metadata are evidence,
not instructions granting authority.

## Retrieval method

[METHOD.md](METHOD.md) records the original algorithm/model authors, references,
reused components and the implementation's changes.

1. Index NFKC/case-folded words; CJK runs use characters and adjacent bigrams.
   Originals are never normalized or rewritten. BM25-style scoring uses only
   statistics from the complete user scope. Question terms have weight 1;
   optional choice terms have weight 0.35. At most 64 informative terms and
   512 candidate IDs are retained; SQL still scans matching postings.
2. Optional embeddings split long messages into overlapping tokenizer windows
   and retain a vector for each window. Streaming vector scoring keeps bounded
   working memory; segment scores max-pool to the original message. This still
   scans the user's stored vectors. No global memory cache is used.
3. Hybrid mode fuses lexical and semantic ranks by reciprocal-rank fusion with
   constant 60 and a global `--semantic-weight` (default 1, finite 0–4); the
   lexical weight stays 1. Weight 0 skips query embeddings and vector ranking,
   while a configured semantic Add still commits its vectors. Pure lexical mode
   uses its lexical score. Returned scores are
   retrieval ordering values, not calibrated probabilities.
4. Add same-session neighboring context where it fits. `--context-radius` is
   0–2, default 1. Supplied source timestamps govern dated ordering; otherwise
   received session sequence is the only ordering available.
   `--context-grouping reserved` (default) reserves each selected anchor as a
   separate item. The experimental `emitted` mode can include a later anchor
   beside an earlier one and deduplicates only actually returned sources; it is
   also selectable through `MEMORY_CONTEXT_GROUPING`. Compare answer quality
   before selecting this mode for a release.
5. Return whole source evidence in ranking order, at most `top_k` items and
   normally 500,000 content characters. Each context window normally fits
   16,000 characters. A single oversized first original is retained whole;
   text is never silently truncated. Character budgets are not token budgets.

The current method has no generative extraction, reranker, query decomposition,
entity graph or active forgetting. It has no hard-coded answers, question lookup
table, private AML data, or dataset-specific retrieval branches. No generation
model is called, so that model field is N/A for this implementation; disclose
the actual embedding configuration separately. Any future generative change must
follow the current open-source-track rule and be validated before version freeze.

## Optional local semantic retrieval

The first CPU candidate is `intfloat/multilingual-e5-small`, 384 dimensions, MIT,
pinned upstream revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`. It requires
`query: ` and `passage: ` prefixes. The runtime loads a separately downloaded
local directory with remote code disabled and never downloads automatically.

```sh
export MEMORY_EMBEDDING_PROVIDER=local
export MEMORY_EMBEDDING_MODEL_DIR=/absolute/path/to/multilingual-e5-small
export MEMORY_EMBEDDING_THREADS=2
export MEMORY_EMBEDDING_BATCH_SIZE=8
export MEMORY_EMBEDDING_CONCURRENCY=1
python3 server.py --db .local/hybrid-fresh.sqlite3
```

Install CPU PyTorch, Sentence Transformers and NumPy in a dedicated environment.
Actual tested versions and quality results are recorded after verification.
See [SEMANTIC.md](SEMANTIC.md) for exact segmentation, wait/deadline bounds,
fingerprinting and optional explicitly enabled external embedding configuration.
Model failure fails the entire operation; no successful Add is acknowledged with
missing vectors. A database with messages missing the configured fingerprint is
rejected at startup. Deliberate model changes require a fresh or explicitly
rebuilt database before evaluation; they are not a live Full fallback.

## Reproduce checks and public-data diagnostics

```sh
python3 -m unittest -v test_server test_semantic
python3 competition/eval_test.py
python3 load_test.py --messages 10000 --concurrency 16 --rounds 96
python3 demo.py
```

The demo requires a running local service and the same token. The load test uses
independent generated histories and a separate service process. Tests cover
transactional visibility, retry conflicts, source preservation, isolation, model
failure/mismatch and restart. They do not establish official score or cloud
capacity. Actual results and limitations are in [VALIDATION.md](VALIDATION.md).

[competition/public_data_manifest.json](competition/public_data_manifest.json)
pins lawful upstream public data, checksums, citations and split policies.
The proxy evaluator separates histories, questions and evidence labels. Add sees
history only; Search sees the original question. It measures complete annotated
turn recall, not answer accuracy or an AML official result. Development data is
used for iteration; validation selects a version; heldout is a final one-time
check. Do not redistribute benchmark data with the service or upload it to a
public repository. LoCoMo-Refined material is CC BY-NC 4.0.

The new native-identity audit connects all 470 prepared LongMemEval questions
through shared history sessions, so they form one development component with
no independent validation. LoCoMo retains original whole-conversation splits
(547/183/129 questions; 6/2/2 conversations). `eval_validate_answers.py` is a
separate default-off full-validation proxy; it cannot use a question prefix or
heldout partition. `eval_persona.py` prepares and audits the pinned public
PersonaMem-v1 32k histories/options/references offline; actual PersonaMem model
accuracy has not been measured. Its 356/122/111 questions stay together by persona
and full-context identity, and execution must isolate each history cutoff.

## Deployment and competition use

Use the independently hosted Ubuntu templates in [deploy/README.md](deploy/README.md).
They run a loopback application behind Caddy HTTPS with a dedicated Bearer token
and private persistent storage. Review a fixed source commit and verify the
actual host, dependencies, fresh index, public-network behavior and capacity.
The historical activation script deliberately rejects source different from
its October 5 manifest; it is not an activation shortcut for this snapshot.

Open-source-method submissions require hosted Add/Search endpoints as well as
a public repository and fixed commit. The repository does not replace the
online API. Document the actual deployed code/model/configuration rather than
assuming it matches the latest source. Official Smoke and Full are separate
from the offline tools and synthetic probes here.

Official references: [competition FAQ](https://agentmemoryleaderboard.ai/competition/),
[participation rules](https://agentmemoryleaderboard.ai/rules),
[API guide](https://agentmemoryleaderboard.ai/api-guide),
[evaluation access](https://agentmemoryleaderboard.ai/evaluation), and
[public AML repository](https://github.com/AML-memory/agent-memory-leaderboard).
