# MemCore Coding v1 configuration

The formal Coding candidate uses a fresh database, an isolated loopback service,
and private environment files. Credentials and deployment addresses are never
included in the public source.

| Variable / flag | Fixed value |
| --- | --- |
| `MEMORY_PROFILE` | `coding` |
| `--context-radius` | `0` |
| `--semantic-weight` | `3.0` |
| `MEMORY_CODING_WEIGHT` | `2.0` |
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

No generative language model is called by Add or Search. `text-embedding-v4` is
used only to index and rank source-derived representations; final answer generation
is owned by the fixed AML evaluation workflow.

The 8192-byte windows preserve every UTF-8 source character. The deterministic
Coding semantic representation is at most about 16 KB in the public proxy, so it
uses at most two complete segments there. Formal capacity tests must still verify
the actual platform payload distribution before Full.
