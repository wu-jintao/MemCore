# Third-party notices

The repository-level MIT license covers the participant-authored MemCore source
and accompanying original documentation. It does not relicense externally
obtained models, datasets, libraries, upstream evaluator code or their underlying
sources. No model weights, dependency wheels, dataset histories or vendored
upstream implementation are included in this snapshot.

## Retrieval methods and external models

[METHOD.md](METHOD.md) attributes BM25 to Robertson and Zaragoza, reciprocal-rank
fusion to Cormack, Clarke and Büttcher, and multilingual E5 to Wang and colleagues,
with primary references and the pinned E5 revision. E5 is downloaded separately;
retain its original license and model card when using or distributing it.

Offline research uses separately obtained Qwen3-32B and BGE reranker models.
Their revisions and provenance are recorded in
[LOCAL_LLM.md](competition/LOCAL_LLM.md) and the aggregate reports. Check and
retain the upstream model licenses, cards and any notices at the exact revision;
this project's MIT license makes no license grant for those assets. The same
principle applies to optional Sentence Transformers, PyTorch, Transformers,
NumPy, vLLM and tokenizers. Hosted text-embedding-v4 or other model services have
their own account and service terms.

## Public research datasets

Data is obtained separately by the participant. Pinned download URLs, file
checksums, citations, authors and license descriptions are recorded in
[public_data_manifest.json](competition/public_data_manifest.json) and
[personamem_data_manifest.json](competition/personamem_data_manifest.json).

- LongMemEval's pinned repository and cleaned dataset declare MIT. Underlying
  filler sources retain their separate terms; do not assume that a dataset-card
  label overrides them.
- LoCoMo-Refined and adapted original LoCoMo material use CC BY-NC 4.0. The local
  preparation retains upstream LICENSE and NOTICE. That data is not distributed
  by this repository and does not become MIT-licensed.
- The pinned PersonaMem-v1 data card declares MIT. Preserve its upstream authors,
  card and original license when obtaining or redistributing the dataset.

Published result files contain aggregate metrics and provenance, not benchmark
histories or question/reference corpora. Local raw-data and run artifacts are
excluded. Source hashes do not provide access to the corresponding private files.

## AML reference material

The AML repository and official website are references for public interface and
measurement contracts. MemCore's service and local evaluation harness were
independently implemented; no upstream AML evaluator or verbatim prompt package
is bundled. The pinned upstream reference commit is recorded in the data
manifest. This repository's MIT license does not assert or change the upstream
AML repository's license. Local proxy results are not official AML results.
