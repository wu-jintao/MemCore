# Configuration

Keep dedicated service authentication and embedding-provider credentials outside
the repository. Load them from a private local environment or service manager;
do not put them in command arguments, reports or published configuration.
The required secret names are `MEMORY_API_TOKEN` and, for HTTP embeddings,
`MEMORY_EMBEDDING_API_KEY`. Their values are deliberately absent here.

## v4 profile

Set the following nonsecret parameters explicitly:

| Variable | Value |
| --- | --- |
| MEMORY_EMBEDDING_PROVIDER | http |
| MEMORY_EMBEDDING_ALLOW_HTTP | 1 (explicit external-call opt-in) |
| MEMORY_EMBEDDING_MODEL | text-embedding-v4 |
| MEMORY_EMBEDDING_DIMENSION | 2048 |
| MEMORY_EMBEDDING_BATCH_SIZE | 10 |
| MEMORY_EMBEDDING_CONCURRENCY | 1 |
| MEMORY_EMBEDDING_HTTP_RETRIES | 0 |
| MEMORY_EMBEDDING_HTTP_SEGMENT_BYTES | 1536 |
| MEMORY_EMBEDDING_QUERY_PREFIX | empty |
| MEMORY_EMBEDDING_DOCUMENT_PREFIX | empty |
| MEMORY_EMBEDDING_HTTP_TIMEOUT | 30 seconds |
| MEMORY_EMBEDDING_OPERATION_TIMEOUT | 600 seconds |
| MEMORY_EMBEDDING_QUEUE_TIMEOUT | 900 seconds |

`MEMORY_EMBEDDING_URL` is the operator's credential-free HTTPS compatible
embeddings URL for the selected workspace. No deployment or account-specific
endpoint is supplied by this snapshot. The source allows HTTP only for loopback
mocks and rejects all embedding redirects. The core adapter inherits ordinary
urllib proxy behavior; the explicit budgeted v4 research client disables proxies
and redirects. Do not describe the core as having the research client's proxy
policy.

Use an empty database for a changed model, endpoint, dimension, prefix or
segmentation profile. The fingerprint prevents silently mixing incompatible
indexes; publication does not migrate an existing deployment or re-embed data.
Default runtime options are context radius 1, `reserved` grouping and semantic
weight 1. Freeze Python, NumPy presence/version and all parameters alongside the
core source manifest for a reproducible comparison.

The deployment profile explicitly keeps operation timeout at 600 seconds.
Queue acquisition can wait up to 900 seconds before the operation's separate
600-second clock begins. This selected profile therefore allows at most
1,500 seconds for queue plus embedding operation, leaving 300 seconds within a
30-minute request/header budget. Serialization, storage and transport need their
own measured margin; this arithmetic is not a capacity or Full-readiness test.
The HTTP queue validator accepts at most 900 seconds; local/disabled backends
retain the previous 300-second bound. Generic operation validation is unchanged
(the module/helper still permit up to 1,500 seconds); the 1,500-second combined
figure applies only when operators preserve this documented 600-second profile.

## Dependencies

Disabled embeddings and HTTP embedding transport need no third-party package.
NumPy optionally accelerates vector ranking. The unchanged local E5 backend
requires separately downloaded weights and the packages in
`requirements-semantic.txt`. The historical Linux/Python 3.10 CPU environment
is recorded in `requirements-semantic-linux-python310.lock.txt`; it is not a
universal lock or a requirement for hosted v4 transport. Research token accounting
also needs `requirements-research.txt`. No dependency install or model download
occurs on module import.
