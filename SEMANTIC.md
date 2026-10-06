# Optional semantic retrieval

`semantic.py` performs embedding and candidate ranking only. It does not generate
answers, open a database, download a model, or contact a service by default. It
contains no official evaluation data. `test_semantic.py` uses generated vectors,
an injected fake tokenizer/model, and a loopback HTTP fake; these tests verify
behavior rather than semantic quality.

## Server integration contract

```python
from semantic import EmbeddingConfig, build_backend, encode_vector, rank_vectors

backend = build_backend(EmbeddingConfig.from_env())  # None means lexical-only.
prepared = backend.embed_documents(source_texts)
# prepared[i] is a nonempty list of normalized float tuples, one per segment.
# Compute prepared before taking the SQLite write lock. Then, within ONE
# transaction, store each original message and all of its segment vectors.
# Store user_id, message_id, segment_index, backend.fingerprint,
# backend.dimension, and encode_vector(vector). Commit before acknowledging Add.
query_vector = backend.embed_query(question)
# SELECT message_id, vector FROM vectors
# WHERE user_id = exact_complete_user_id AND fingerprint = backend.fingerprint
candidates = rank_vectors(query_vector, scoped_rows, limit=100, chunk_size=256)
# candidates are unique (message_id, cosine_score) pairs, max-pooled over the
# source's segments. Re-fetch every original message with the SAME user_id;
# fuse with lexical candidates, and return source evidence, not generated answers.
```

`EmbeddingError` means the entire operation failed. The integration must not
acknowledge a partial embedding write, queue invisible background indexing, or
silently change to a different embedding model. Its `EmbeddingCapacityError`
subclass identifies a permanent declared-input-capacity rejection, such as an
exceeded document, query, or total segment limit. Add and Search return HTTP 422
for this case; retrying the same input cannot make it fit, and Add commits no
messages or partial vectors. Queue waits, model failures, and operation timeouts
remain retryable HTTP 503 failures. Any lexical fallback must be explicitly
designed, disclosed, and frozen before Full.

Persist the fingerprint with vectors. It binds dimensions, provider/model,
prefixes, normalization and segmentation; the local provider additionally hashes
model weights, tokenizer and configuration files. Refuse to reopen an indexed
database using a different fingerprint, or when some source messages lack their
vectors. Re-index into a new database before deliberately changing the model.
There is no global embedding/query cache or cross-user memory cache in this
module. Complete user_id isolation is enforced by the server's SQL, not inferred
from a shortened ID or dataset name.

## Offline CPU model

The selected first candidate is `intfloat/multilingual-e5-small`. Its
[official model card](https://huggingface.co/intfloat/multilingual-e5-small/raw/main/README.md)
states MIT, 12 layers, 384 dimensions, required `query: ` / `passage: ` prefixes,
and truncation beyond 512 tokens. Preserve the upstream license/attribution in
the release. This choice is a CPU feasibility baseline, not a claim of the best
competition score.

Download the model separately, on an authorized personal cloud machine, to its
persistent model directory. The module never automatically downloads it. Install
CPU PyTorch, Sentence Transformers with support for `local_files_only=True`, and
NumPy in an isolated environment; record and freeze the actual package versions
after the real-model smoke test. Loading is CPU-only and disables remote code.

The local macOS CPU smoke test used Python 3.12.14, Sentence Transformers 6.1.0,
PyTorch 2.14.1, Transformers 5.18.0, and NumPy 2.5.3, with model revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`. Its three independently written English /
Chinese paraphrase checks found the intended source, a 1713-token original was
encoded in six source windows including its tail, and sixteen queued short
queries completed. Those are functionality checks on this Mac, not competition
accuracy or a capacity guarantee for the cloud machine. For Linux, use the
official PyTorch CPU wheel index and record that environment separately instead
of installing unnecessary CUDA dependencies.

Example environment (the model directory must already exist):

```sh
export MEMORY_EMBEDDING_PROVIDER=local
export MEMORY_EMBEDDING_MODEL_DIR=/srv/aml/models/multilingual-e5-small
export MEMORY_EMBEDDING_THREADS=2
export MEMORY_EMBEDDING_BATCH_SIZE=8
export MEMORY_EMBEDDING_CONCURRENCY=1
```

Documents use overlapping source-offset token windows: 384 tokens with 64-token
overlap by default, then each prefixed window is retokenized to verify that it
fits the actual model capacity. There is no truncation. Each window gets its own
vector and maps back to the complete original message. Long queries average all
their normalized segment vectors and normalize again; the default query limit
is 8 segments. Over-capacity text fails explicitly instead of dropping its tail.
Defaults limit each source to 512 segments and one operation to 4096 segments.

One local inference operation runs at a time, with two PyTorch intraop threads
and one interop thread. Batches contain at most eight windows. Queue wait is
bounded to 120 seconds; an operation is checked against a 600-second deadline
between batches. An already-running model batch cannot be forcibly interrupted;
measure the largest batch on the actual machine before formal deployment.

## External embedding adapter

This is optional and disabled until `MEMORY_EMBEDDING_PROVIDER=http` AND
`MEMORY_EMBEDDING_ALLOW_HTTP=1` are set. Supply an individually authorized endpoint,
model, key and expected dimensions using `MEMORY_EMBEDDING_URL`,
`MEMORY_EMBEDDING_MODEL`, `MEMORY_EMBEDDING_API_KEY`, and
`MEMORY_EMBEDDING_DIMENSION`. The URL is the complete compatible embeddings path.
Use HTTPS; plain HTTP is accepted only for loopback tests. No company endpoint,
database or model service is configured here. There is no default paid API call.

The adapter requires explicit indexes, exact response count, finite values,
correct dimensions and nonzero vectors. It caps response sizes, batches, queue
wait, request timeout, and retries. Retries default to zero; opt-in retries are
bounded at two. API credentials are excluded from fingerprints and error text.
Provider-specific prefixes can be configured. All source characters are retained
in UTF-8 byte windows (1536 bytes by default); because the remote tokenizer is
unknown, verify its token limit and rejection/truncation behavior independently
before selecting this backend for Full.

### Academic text-embedding-v4 preparation

The [competition FAQ, question 05](https://agentmemoryleaderboard.ai/competition/)
requires this model for Academic embeddings. Its [official synchronous API](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api)
supports 64, 128, 256, 512, 768, 1024, 1536 or 2048 dimensions, at most ten
inputs per request, and 8192 tokens per input. This adapter sends the configured
`dimensions` explicitly for this model and rejects an invalid dimension or batch
before any call. Other HTTP models retain their existing request format.

Prepare `MEMORY_EMBEDDING_MODEL=text-embedding-v4`, dimension 2048 (an initial
candidate, not a measured quality choice), batch size at most ten, and empty E5
prefixes. Obtain the exact HTTPS compatible `/embeddings` endpoint and matching
regional key from the participant's own Model Studio account. Current vendor
documentation uses a workspace-specific host; do not invent a workspace ID or
assume a generic endpoint. Keep external calls disabled until the credential,
endpoint, allowed data scope and measured request/cost limits are supplied.

Local fake-service regressions verify 2048-dimensional requests, response
index ordering and ten-plus-one batching. They are not live-provider validation.
The byte-window adapter still needs real token-limit, non-truncation, quality,
rate-limit, cost and capacity checks. Use a fresh database/fingerprint; E5 vectors
cannot be reused for this model. No text-embedding-v4 index is deployed yet.

## Ranking bounds and verification

`rank_vectors` streams at most 256 vectors per chunk and keeps a bounded heap of
candidate source IDs. Multiple source segments use their maximum cosine score;
they do not consume duplicate candidate slots. NumPy scoring uses elementwise
multiply/reduce rather than multi-threaded BLAS. At most two ranking calls run at
once. Without NumPy, a functional but slower Python implementation is used.

Run `python3 -m unittest test_semantic -v` for offline correctness. Then evaluate
the actual model on independently generated paraphrases and lawful public
development histories, at matched evidence budgets. Also test full user_id
isolation, restart, model mismatch and failed Add rollback in the server suite.
Report measured latency, memory, recall-all and answer proxy scores separately;
neither fake-vector tests nor upstream model-card scores are official AML results.
