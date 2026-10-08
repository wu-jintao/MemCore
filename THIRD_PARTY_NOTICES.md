# Third-party notices

The repository MIT license covers participant-authored MemCore source and
original documentation. It does not relicense external models, datasets,
libraries, upstream evaluators or their underlying sources. No external weights,
datasets, dependency wheels or vendored upstream evaluator are included.

BM25, RRF and E5 are attributed in `METHOD.md`. The optional E5, BGE and Qwen
research assets must be obtained separately at recorded revisions with their
upstream licenses and model cards. Hosted embedding services have their own
account/service terms. Python packages retain their original licenses.

`competition/public_data_manifest.json` and
`competition/personamem_data_manifest.json` record public dataset references,
versions, file checksums and attribution. These manifests contain no histories,
questions or labels. LongMemEval's pinned repository/data declare MIT;
underlying filler sources retain their separate terms. LoCoMo-Refined and
adapted original LoCoMo use CC BY-NC 4.0 and must retain their upstream notices.
The pinned PersonaMem-v1 card declares MIT. None becomes MIT-licensed merely by
being used with MemCore.

AML public interface/evaluation documentation is referenced by the research
harness. MemCore independently implements its service and local proxy harness;
no upstream AML evaluator or verbatim prompt package is bundled. Local proxy
metrics are separate from official AML results.

Historical public reports under `competition/results/` are retained unchanged
from commit `544b72d51a49139512bdbeec2df1f3198780143f`. They contain aggregate
metrics and provenance, not benchmark histories or question/reference corpora.
Each records its own model/configuration and limitations; preservation does not
apply old results to the current v4 configuration.
