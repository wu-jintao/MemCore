# Source and deployment status

Snapshot date: October 6, 2026. This repository starts with a reviewed source-only
snapshot and does not import development-machine Git history or private runtime
artifacts. Use the published commit SHA to identify this snapshot. Code-file
checksums are in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json).

## Published research source

The current `server.py` includes source/date helpers and optional `reserved` /
`emitted` evidence grouping. `reserved` remains the default; the `emitted`
experiment regressed on the matched local answer proxy. `semantic.py` includes
an explicit opt-in HTTP adapter for text-embedding-v4 dimensions and batching.
Those changes have offline behavior tests; their publication is not a claim of
cloud activation or live-provider validation.

## Earlier deployed E5 version

The historical October 5 activation manifest pins different bytes:

| File | Deployed historical SHA-256 |
| --- | --- |
| server.py | 96124a789829afdfead08f74fb9c89ef7b41bbaf2523c90eb1746846e0d76ab9 |
| semantic.py | 8a0473f3ad91714ac5f88b4db177ef9f34e683dcfbcea1acbc3745c462b93145 |

These hashes document the difference; this snapshot does not include a copy of
that historical release. `deploy/activate_semantic.py` keeps its original guard
and will reject the newer source until a separately reviewed activation manifest
and capacity checks are prepared. No source/model deployment is performed by
publishing this repository.

Publication preparation changes documentation and one PersonaMem test fixture
to remove a dependency on a pre-existing local data directory. Service,
embedding and evaluation implementation files retain their reviewed bytes;
the optional real-data test skips only when its upstream files are absent.

## Academic preparation and formal evaluation

The deployed E5 integration baseline has not been replaced with text-embedding-v4.
Live API access, fresh-index validation, model quality, rate limits, cost and
end-to-end capacity remain gates for an Academic embedding candidate. The online
memory service has no generative LLM component; eligibility for that configuration
should be confirmed against the organizer's current submission checklist.

Official Smoke and Full have not been run. Public transport stability is still
unresolved even at low concurrency. Do not label this commit a validated official
submission or claim the source matches the submitted hosted API without an
explicit source/configuration audit and a new deployment record.
