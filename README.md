# MemCore v1.1

MemCore is a source-preserving Add/Search memory service. It stores original
messages and synchronous SQLite indexes, then returns attributable evidence to
an external answer system. The service does not generate final answers.

MemCore **v1.1**, released as source dated October 8, 2026, updates the embedding transport
boundary and adds isolated v4 research/operations tools. The HTTP queue
validator now permits a 900-second wait; the documented deployment profile
keeps the operation timeout at 600 seconds (queue900/op600). The source checks
were completed before publication. Source publication does not establish final method selection, completed QA for the current v4
configuration, or official Smoke/Full readiness.

## Run locally

Python 3.10 or newer is recommended. Lexical retrieval and the HTTP embedding
adapter use the Python standard library. NumPy is optional for vector scoring;
freeze its presence and version when comparing exact ranking behavior.

```sh
export MEMORY_API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export MEMORY_EMBEDDING_PROVIDER=disabled
python3 server.py --host 127.0.0.1 --port 8080 --db .local/memory.sqlite3
```

In another terminal using the same dedicated local token:

```sh
python3 demo.py --base-url http://127.0.0.1:8080
```

The demo uses synthetic messages only. `GET /health` is unauthenticated;
`POST /add` and `POST /search` require `Authorization: Bearer <token>`.
The token authorizes the service caller, which can access any complete user ID;
it is not an end-user authorization mechanism.

## Interface

Add accepts `request_id`, `user_id`, `session_id` and a list of `messages`.
Each message has `role` (`user` or `assistant`), nonempty `content` and an
optional signed 64-bit Unix-millisecond `timestamp`. All identifiers preserve
case, Unicode and surrounding whitespace. An identical canonical Add body at
the same `(complete user_id, request_id)` is an idempotent retry; a different
body at that key returns 409.

Search accepts `query`, `user_id`, `top_k` (1 through 100) and optional `options`.
It returns `{"data":[{"id":"source-id","content":"evidence","score":0.0}]}`.
Only the complete requested user scope is searched. Missing dates remain missing.

The default method combines scoped BM25 and optional cosine ranking using
reciprocal-rank fusion, then packages same-session neighbors with independent
source identities. The runtime core is exactly `server.py` and `semantic.py`.
[CORE_MANIFEST.json](CORE_MANIFEST.json) pins those two files.

## Optional embeddings and tools

[CONFIGURATION.md](CONFIGURATION.md) documents the fixed v4 profile and a fresh
index requirement. The local E5 backend remains a separately obtained research
baseline; optional dependency files are retained for its reproducibility.
[competition/README.md](competition/README.md) describes public-data research
and offline analysis. [deploy/README.md](deploy/README.md) describes isolated
candidate preparation and small synthetic probes. These tools are not invoked
by the Add/Search server.

Run the dependency-free service checks:

```sh
python3 -m unittest test_server test_semantic
```

The 19 historical public reports under [competition/results](competition/results/RESULTS.md)
are preserved byte-for-byte from source commit
`544b72d51a49139512bdbeec2df1f3198780143f`. Each describes its own dated
configuration and proxy limits; none substitutes for current v4 QA, Smoke or
Full. Raw run outputs, datasets, model weights, credentials, deployment
addresses and Git metadata are excluded from the source package.
[VALIDATION.md](VALIDATION.md) records the local source checks completed before
this release. See [METHOD.md](METHOD.md),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the MIT [LICENSE](LICENSE).
