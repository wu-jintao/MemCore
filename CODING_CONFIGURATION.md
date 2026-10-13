# MemCore Coding v1.1 configuration

The formal Coding candidate uses a fresh database, an isolated loopback service,
and private environment files. Credentials and deployment addresses are never
included in the public source. The dense post-reranker defaults to disabled in
the generic server; the reviewed Coding v1.1 deployment enables it explicitly.

| Variable / flag | Fixed value |
| --- | --- |
| `MEMORY_PROFILE` | `coding` |
| `--context-radius` | `0` |
| `--semantic-weight` | `3.0` |
| `MEMORY_CODING_WEIGHT` | `2.0` |
| `MEMORY_CODING_DENSE_RERANK_WEIGHT` | `0.5` |
| `MEMORY_CODING_RESULT_LIMIT` | `12` |
| `MEMORY_CODING_CHARACTER_LIMIT` | `180000` |
| `--max-body-bytes` | `134217728` (128 MiB; Coding candidate only) |
| `MEMORY_EMBEDDING_PROVIDER` | `http` |
| `MEMORY_EMBEDDING_MODEL` | `text-embedding-v4` |
| `MEMORY_EMBEDDING_DIMENSION` | `2048` |
| `MEMORY_EMBEDDING_BATCH_SIZE` | `10` |
| `MEMORY_EMBEDDING_CONCURRENCY` | `1` |
| `MEMORY_EMBEDDING_HTTP_RETRIES` | `2` |
| `MEMORY_EMBEDDING_HTTP_SEGMENT_BYTES` | `8192` |
| `MEMORY_EMBEDDING_MAX_DOCUMENT_SEGMENTS` | `4` |
| `MEMORY_EMBEDDING_MAX_QUERY_SEGMENTS` | `4` |
| `MEMORY_EMBEDDING_MAX_TOTAL_SEGMENTS` | `4096` |
| `MEMORY_EMBEDDING_HTTP_TIMEOUT` | `30` seconds |
| `MEMORY_EMBEDDING_OPERATION_TIMEOUT` | `600` seconds |
| `MEMORY_EMBEDDING_QUEUE_TIMEOUT` | `900` seconds |
| process recovery | systemd `Restart=on-failure`, 5-second delay, bounded start rate |

Add and Search call no generative model. `text-embedding-v4` indexes and ranks
source-derived representations; final answer generation is owned by the fixed
AML evaluation workflow.

One bounded query embedding operation returns both the normalized segment mean
used by the original dense channel and the normalized source segments used by
the v1.1 post-reranker. The post-reranker makes no additional model request. It
reads only the stored vectors for evidence already selected by v1 and reorders
that exact set with a 0.5-weight reciprocal-rank term based on the maximum cosine
across every query/document segment pair.

The 8192-byte windows preserve every UTF-8 source character. Query and document
segment bounds remain unchanged from Coding v1. The reranker changes neither the
12-item result ceiling nor the 180,000-character whole-evidence budget.
