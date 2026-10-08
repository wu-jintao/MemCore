# MemCore v1.1 source status

MemCore v1.1 is the October 8, 2026 source release with the explicit HTTP
deployment profile queue900/op600. `SOURCE_MANIFEST.json` fixes the complete
release tree; `CORE_MANIFEST.json` fixes the minimal two-file inference package.
The recorded local source checks were completed before publication. Source publication does not activate a hosted API or establish a
matching final submission.

`server.py` is byte-identical to the October 6 public source snapshot at commit
`544b72d51a49139512bdbeec2df1f3198780143f`. `semantic.py` adds rejection of
HTTP redirects for embedding requests and raises only the HTTP queue-wait
validation ceiling from 300 to 900 seconds. Local/disabled queue bounds,
operation clocks and slot-release behavior are unchanged. Ranking,
segmentation, storage and evidence packaging are unchanged. New isolated tools cover v4 budgeted
public-dev retrieval, native query-mode research, paired PersonaMem replay and
isolated HTTP candidate preparation/probing. Detailed differences are recorded
in `SNAPSHOT_DIFF.json`.

A small probe fix skips socket timeout updates after a complete HTTP/1.0 body
closes the socket. Deadline checks and the timer remain active. Three mocks
verify complete-body success, incomplete-body failure and preservation of true
read timeout/network errors. The source checks also use a self-contained
PersonaMem fixture with an optional absent-data skip and a reserved RFC 5737
address in a negative endpoint fixture. No retrieval algorithm is added.

All 19 earlier public files under `competition/results/` remain at their
original paths and are byte-identical to the earlier commit. They are historical
public research/transport reports, with the configurations and limitations
recorded in each report. Old proxy scores, source-inclusive measurement notes
and transport diagnostics must not be presented as a current v4 answer-quality
result or replace this version's QA, Smoke or Full. Their original contents are
retained rather than rewritten to describe the new configuration.

The old E5-specific activation script, generic historical bootstrap/host
inspection tools, old deployment template files and superseded narrative docs
are absent from this tree. Runtime core, dependencies, dataset attribution,
existing research modules and the MIT license remain. The two original dataset
manifests contain references and checksums, not dataset content.

This release does not claim completed QA for the current v4 configuration,
final method selection, official Smoke/Full, official score, public-network
readiness or a deployment-source match. Those need separate completed records
and an explicit source/configuration audit. Raw histories, questions/reference
corpora, credentials, logs, retrieval outputs, private run artifacts and private
operation scripts are excluded.

## Queue-wait source update and cache provenance

The queuefix profile is queue 900 seconds plus operation 600 seconds. The
existing operation timer begins after acquisition of one embedding slot and
covers all batches before releasing it. No scheduling or ranking algorithm is
changed. The default queue field remains 120 seconds; deployments must set the
documented 900-second profile explicitly.

Prior retrieval/QA caches were produced with semantic source SHA-256
`59163d78e889f8ae10ce4b63bba8b54fcfc4f5f6e5d00150160098bc5d1c0e50`.
This release has different semantic file bytes solely because of the HTTP
configuration bound. The complete `_Backend`, HTTP embedding, segmentation,
normalization/fingerprint and vector-ranking implementations remain byte-for-
byte unchanged; `server.py` is also unchanged. The model/configuration corpus
fingerprint excludes queue timeouts and therefore stays the same. Cached QA can
inform unchanged retrieval quality, but must not be labelled an execution of
the new source or proof of its deployment/capacity behavior.
