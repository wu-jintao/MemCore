# MemCore — method and attribution

This is a participant-authored memory-service implementation for the AML textual
track. Its purpose is to return attributable historical evidence to the platform
answer model. It is currently a development candidate; there is no official score,
final release, or claim of winning the competition.

The E5 method described here is a research baseline. The [October 6 competition
FAQ, question 05](https://agentmemoryleaderboard.ai/competition/) requires
`text-embedding-v4` for Academic embeddings; a compliant final method must be
measured and documented separately. The optional HTTP adapter supports its
explicit output dimensions and ten-input batch limit; no paid call or production
switch has been made for that model.

## What the service contributes

Add retains the original role, text and optional millisecond timestamp. A single
SQLite transaction makes the request acknowledgement, original messages, lexical
statistics and every configured embedding segment durable together. Idempotency
uses the complete user ID and request ID with canonical-body equality. Search
scopes every statistics, postings, vectors and source-text lookup to that same
complete user ID. Different evaluation scopes never share a retrieval corpus.

The lexical implementation uses BM25 with k1 = 1.2 and b = 0.75 and a positive
smoothed inverse-document-frequency term. Unicode NFKC/case-folded words and CJK
characters/bigrams are indexing representations; original text stays unchanged.
Original question terms have weight 1. Choice-only terms have weight 0.35. The
implementation bounds the selected informative terms and candidate lists rather
than scanning source text in Python. Matching postings still determine SQL work.

The optional semantic adapter loads the unchanged, pinned multilingual E5 small
model from a local directory. Original messages use source-offset token windows
of 384 tokens with 64-token overlap; prefix and retokenization checks ensure that
no embedding window exceeds the model limit. `query: ` and `passage: ` prefixes
follow the model instructions. Segment scores max-pool to their original message.
Vector scoring streams from the scoped database with bounded working memory.
It still scans the user's stored vectors and is not a sublinear vector index.

Hybrid ranking uses weighted reciprocal-rank fusion with constant 60. The current
equal-weight candidate has lexical weight 1 and semantic weight 1, fixed globally.
Exact fused-score ties prefer the lexical candidate rank, then stable source ID;
this preserves a unique lexical source when an unrelated dense-first source has
the same RRF contribution. The fusion scores themselves are unchanged.
Same-session neighboring messages may accompany an anchor, each with its own
source identity. Supplied dates control fully dated neighborhoods. If the anchor
or a received neighbor within the configured radius lacks a date, adjacency and
presentation use the scoped received sequence for that neighborhood; missing
event dates are not inferred. Evidence is returned at whole-message boundaries
in rank order, with top_k capped at 100. The development implementation currently
uses a 500,000-character output boundary and 16,000-character context windows;
these are character budgets, not the platform's token allowance. A single
oversized first original is preserved whole, so the character boundary is not
a hard response-size cap. Output-budget
changes are separate, measured candidates until a final version is selected.

The deployed Add/Search service has no generative extraction, training,
fine-tuning, query rewrite, reranker or final-answer generation. Separate offline
research tools test BGE and Answer/Judge models. Benchmark question/reference datasets and evidence
annotations are not embedded in service code. Local adapters do not append
question, option, reference or label fields to Add; native historical questions
remain unchanged, including those present in PersonaMem prefixes. Evaluation
labels are confined to the offline grader. The generation-model field
is N/A; any selected embedding model is disclosed separately.

## Prior work and implementation changes

The BM25 and RRF ideas are established retrieval methods, not algorithms invented
by this project. The service and local public-data evaluation harness were newly
implemented; neither copies the AML evaluator or an upstream memory-system
implementation. The contributions described above are the scoped transactional
storage, lossless segmentation, evidence/source packaging and engineering choices.
The E5 weights and model behavior are reused unchanged, not retrained on the
development data or evaluation data.

- Stephen Robertson and Hugo Zaragoza (2009),
  [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019).
  This implementation adds complete-user statistics, its disclosed tokenization,
  bounded query/candidate selection and optional-choice weighting around BM25.
- Gordon V. Cormack, Charles L. A. Clarke and Stefan Büttcher (2009),
  [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf).
  This implementation combines its two disclosed retrieval lists and exposes one
  global semantic weight; it does not learn ranks from benchmark labels.
- Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder and Furu Wei
  (2024), [Multilingual E5 Text Embeddings: A Technical Report](https://arxiv.org/abs/2402.05672).
  [Upstream model](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3),
  revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, MIT, 384 dimensions.
  Our adapter adds lossless long-message windows, synchronous index completion,
  scoped persistence and model/configuration fingerprint checks.

## Evaluation and its limits

The public-data manifest records dataset versions, authors, licenses, checksums
and group splits. Retrieval reports measure complete annotated-turn recall and
actual request failures. Separate local Qwen Answer/Judge comparisons measure
proxy correctness, including the public-contract calibration profile; they do
not measure platform answer accuracy, official AML scores, Streaming performance,
or winning rank. Token analyses use explicitly named
tokenizers and instruction reserves. Public family Answer/scoring prompts are
available at the pinned AML repository commit; production model/tokenizer,
packing and orchestration remain unconfirmed. Development comparisons guide experiments; validation selects the
method and heldout remains a final check. Small conversation counts and any
cross-split filler reuse are disclosed.

The native-identity audit places all 470 public LongMemEval questions in one
development component because their history-session IDs overlap transitively.
The original LoCoMo whole-conversation validation and holdout are retained.
PersonaMem preparation keeps history cutoff, question/options and reference
separate and groups all contexts/prefixes from one persona together. Synthetic
behavior, source grounding and actual model semantics are reported separately;
complete retrieval does not ensure correct role attribution in the answer.

See [the initial comparisons](competition/results/RESULTS.md),
[token diagnostics](competition/results/TOKEN_BUDGETS.md),
[expanded development comparisons](competition/results/EXPANDED_DEV.md),
[the separate output-cap experiment](competition/results/CAP500K_DEV.md),
[calibrated local correctness](competition/results/CALIBRATED_LME60_DEV.json),
[capability/data boundaries](competition/results/CAPABILITY_COVERAGE.json), and
[dataset attribution](competition/public_data_manifest.json). Raw benchmark
histories, source evidence, private retrieval outputs and credentials are excluded
from publication and from the deployed inference package.
