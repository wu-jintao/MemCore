# MemCore v1.1 source validation

These local synthetic/unit/mock checks were completed on October 8, 2026,
before publication of the v1.1 source release. They are not answer-quality
validation, deployment activation or official AML evaluation. No real credential
file, model API, SSH or non-loopback network was used.

| Check | Environment | Result |
| --- | --- | --- |
| Core service and embedding adapter | Python 3.12.14 | 55 tests passed |
| Operations tools | Python 3.12.14 | 45 tests passed |
| External network attempts | Isolated credential-free environment and non-loopback guard | 0 |

Six new instant mocks validate HTTP queue=900 and reject >900, non-finite or
nonpositive budgets; local/disabled queue limits remain 300. They verify that
one slot is held across all batches, the unchanged operation deadline starts
after acquisition with a 600-second budget, a failed acquisition does not
release an unheld slot, and timeout/model failures release the slot. A second
operation can succeed after a timed-out operation releases it. No test waits
900 seconds or invokes a model.

The queuefix changes only the HTTP configuration validation ceiling and its
matching preparation bound. The module's default queue remains 120; operation
validation/default and actual timer/slot semantics are unchanged. The selected
deployment profile is explicitly queue 900 / operation 600. Its 1,500-second
embedding budget leaves 300 seconds within a 30-minute transport budget, but
storage, serialization and transport margin still need actual verification.

`server.py` and ten complete embedding/ranking implementation blocks are byte-
identical to the earlier frozen source. That frozen 79-file source package had
193 research-harness checks (192 passed, one optional absent-data skip); those
unchanged research suites were not repeated for this queue-bound-only update.
The existing probe closed-socket regression mocks continue to pass in this run.
Prior QA/retrieval caches keep semantic SHA-256
`59163d78e889f8ae10ce4b63bba8b54fcfc4f5f6e5d00150160098bc5d1c0e50`;
they are not an execution of the new queuefix semantic source or evidence of
new-source deployment capacity.

The 19 historical public report-directory files remain exact bytes from commit
`544b72d51a49139512bdbeec2df1f3198780143f`. Each applies to its own configuration;
none substitutes for current v4 QA, Smoke or Full. The MIT license is retained.

`CORE_MANIFEST.json` pins the changed semantic core and unchanged server.
`SOURCE_MANIFEST.json` pins the complete source tree except itself.
`PRIVACY_SCAN.json` records the fixed whitelist, historical-file byte checks,
syntax/JSON/link checks and privacy review. This validation record covers the
work completed before source publication. It contains no new current-v4 QA
result, official Smoke/Full outcome, deployment activation or readiness claim.
The released profile is queue900/op600; source publication does not validate
its live deployment or capacity.
