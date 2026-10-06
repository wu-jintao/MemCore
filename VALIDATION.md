# Validation evidence and limits

Date: October 6, 2026. These are local synthetic checks and public-data research
results, not official AML Smoke, Full or leaderboard scores. Source status is
recorded separately in [SOURCE_STATUS.md](SOURCE_STATUS.md).

## Reproducible behavior checks

The standard-library lexical service and offline fake-model tests can run with
Python 3.9+. Semantic model and research environments have separate dependencies.

```sh
python3 -m unittest -v test_server test_semantic
python3 -m unittest discover -s competition -p '*_test.py'
python3 -m unittest discover -s deploy -p 'test_*.py'
python3 load_test.py --messages 10000 --concurrency 16 --rounds 96
```

Behavior tests cover authenticated schemas, exact complete-user isolation,
durable immediate visibility, canonical idempotency/conflicts, original text and
timestamps, rollback, restart, source attribution, model fingerprints, lossless
segmentation and fake HTTP response validation. They do not test semantic quality
or a live paid embedding service. Synthetic load must be measured on the actual
host before using it as a capacity declaration.

The clean publication copy was checked on Python 3.12.14: 49 service/semantic
tests, 140 research tests and 16 public-probe tests ran. 204 passed; one optional
real PersonaMem download/checksum test was skipped because upstream data is not
bundled. Manifest validation and synthetic tampering tests still run; when the
upstream files are present their size/checksum verification remains mandatory.
Research fixture output uses temporary private storage and requires no existing
development-machine `.local` directory. No live model API was called by these
checks.

## Public-data research

The reports retain separate retrieval, token-budget and Answer/Judge measures:

| Experiment | Baseline | BGE research candidate | Scope |
| --- | ---: | ---: | --- |
| Calibrated LongMemEval development | 41/60 | 43/60 | 5 gains, 3 losses, zero failures |
| Complete LoCoMo validation | 142/183 | 148/183 | 17 gains, 11 losses, zero failures |

See [LongMemEval calibration](competition/results/CALIBRATED_LME60_DEV.json),
[LoCoMo validation](competition/results/LOCOMO183_VALIDATION.json),
[capability/data boundaries](competition/results/CAPABILITY_COVERAGE.json), and
[local model details](competition/LOCAL_LLM.md). These matched runs use a local
Qwen3-32B Answer/Judge proxy. Public questions, the same model family for judging
and only two LoCoMo validation conversations limit inference about generalization.
All 470 prepared LongMemEval questions share a connected history component and
are development-only. The 129-question LoCoMo holdout has not been used for
retrieval/Answer. PersonaMem preparation is audited but model accuracy is pending.

Historical date-aware comparisons and the rejected evidence-grouping trial are
retained as separate reports. They must not be merged with the calibrated scores.
A CPU reranker probe took 290.70 seconds for one query's 100 candidates/157 pairs;
BGE remains offline research, with no online promotion or cloud capacity claim.

## Deployment and remaining gates

Historical cloud synthetic checks passed on an earlier E5 release. This snapshot
has additional local changes and requires its own deployment validation. The
public-network diagnostics contain unresolved failures and incomplete requests;
see [deploy/VALIDATION.md](deploy/VALIDATION.md). No stable concurrency or Full
capacity is established by the small checks.

Remaining gates include a live Academic model/index, actual-host quality and
capacity checks, source/configuration matching, stable hosted Add/Search,
organizer eligibility review, official access, official Smoke and Full.
