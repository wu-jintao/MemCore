# Query-capacity and transport recovery amendment

This amendment is based on MemCore v1.1 public commit
`515be6c5c3a3806b1070f2bcc687d177c83d7a34` and is intended to recover an
interrupted evaluation from its original checkpoint. Organizer review is
pending. Publication of this amendment does not establish organizer approval,
an approved scored version, a new Full evaluation or an official result.

## Scope

The original deployment used `MEMORY_EMBEDDING_MAX_QUERY_SEGMENTS=8`. This
amendment raises the accepted configuration bound from 1..64 to 1..128 and
selects `MEMORY_EMBEDDING_MAX_QUERY_SEGMENTS=128` for recovery. The source
default remains 8. The capacity core change is one configuration-bound line
in `semantic.py`; `server.py` is unchanged.

The transport amendment imports `http.client.IncompleteRead` and handles it
in the existing retryable network-error path. The recovery profile selects
`MEMORY_EMBEDDING_HTTP_RETRIES=2`, within the existing valid range; the source
default remains zero. A partial HTTP read is discarded and the identical
complete input batch can be retried at most twice. Exhausting those retries
fails the whole embedding operation without acknowledging partial vectors.
The existing 30-second HTTP timeout, 600-second operation deadline, 900-second
queue wait and bounded backoff logic are preserved. Retries remain within
the operation deadline and may increase model requests, usage and latency.

The embedding model remains `text-embedding-v4`, with 2048 dimensions,
1536-byte UTF-8 segments, empty prefixes, batch size 10 and concurrency 1.
Segment encoding, normalization, normalized segment-mean query aggregation,
retrieval logic, isolation and request-id idempotence are unchanged. No input
is truncated. Model, endpoint, dimension, prefixes and segmentation determine
the existing vector fingerprint; query capacity does not, so the same profile
remains compatible with its stored vectors. This compatibility does not imply
that the declared configuration is unchanged or that organizer approval is
unnecessary. Existing memory, namespaces, indexes and idempotency records must
be retained when recovering a deployment.

## Validation and limits

Focused offline tests use synthetic text and stub vectors only. They check
configuration acceptance, complete source preservation across many batches,
unchanged short-query output and fingerprint, the inclusive 128-segment ASCII
boundary, and explicit overflow rejection. Transport mocks verify one partial
response then success, identical complete-batch retries, whole-operation
failure after exhausting retries, and deadline-limited read/backoff waits.
Existing core tests use generated
vectors and local HTTP mocks. No model service or evaluation input is used.

With empty prefixes, 128 segments permit at most 196608 ASCII bytes. Multi-byte
UTF-8 boundaries can reduce effective byte capacity slightly. Larger unknown
inputs can still be rejected; this amendment is not a Full-completion guarantee.
Longer accepted queries require more model batches, up to 13 at batch size 10,
and therefore can increase queue load, latency and model usage. Existing
timeouts and concurrency are preserved and require live operational monitoring.
These narrow changes do not promise recovery from every upstream response
error, interruption or oversized query.

The original manifests and October 8 validation history are retained in Git
at the base commit. The manifests in this tree identify the amended source;
prior quality or official evaluation results must not be relabeled as results
of this amendment.
