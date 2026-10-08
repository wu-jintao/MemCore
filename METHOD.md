# Method and attribution

Add retains original role, text and optional timestamp. One SQLite transaction
commits the request acknowledgement, original messages, scoped lexical
statistics and all configured embedding segments. Retry identity uses complete
user ID plus request ID and canonical-body equality. Search applies the same
complete user scope to statistics, postings, vectors and source lookup.

The lexical implementation uses BM25 with k1=1.2 and b=0.75, a positive smoothed
inverse-document-frequency term, Unicode NFKC/case-folded words and CJK
characters/bigrams. Original query terms receive weight 1; choice-only terms
receive weight 0.35. Query terms and ranking candidates are bounded; originals
are not rewritten.

The v4 HTTP profile uses `text-embedding-v4`, 2048 dimensions, batch size 10,
empty document/query prefixes and lossless 1536-byte UTF-8 source windows.
Character boundaries are preserved. The adapter requests every source window;
provider-side tokenization/truncation is a separate verification concern.
Document segment scores max-pool to their original message. Long query segment
vectors are averaged and normalized. Vectors are stored as normalized float32
BLOBs with a model/configuration fingerprint. Ranking streams the complete
scoped vector collection with bounded working memory; it is not a sublinear
vector index.

Weighted reciprocal-rank fusion uses constant 60. Lexical and semantic weights
are 1 by default. Exact fusion-score ties prefer lexical rank, then stable source
ID. Same-session neighbors carry independent source identities. Fully dated
neighborhoods use supplied dates; a missing date within the neighborhood uses
received sequence rather than an inferred date. Default context radius is 1
and grouping is `reserved`. `emitted` is an optional experiment, not the selected
profile. Search caps `top_k` at 100, whole returned evidence at 500,000 characters
and each anchor context window at 16,000 characters. These are character limits,
not an answer model's token budget.

The inference service has no generative extraction, training, fine-tuning, query
rewrite, reranker or final-answer generation. Offline BGE/Answer/Judge tools are
included as optional research source, not enabled service components. Labels
are confined to local grading; preparation keeps original history, questions
and reference/evidence labels separate.

## Prior work

BM25 and RRF are established methods. MemCore independently implements scoped
transactional storage, segmentation, retrieval and evidence packaging; it does
not claim to invent these ranking algorithms.

- Stephen Robertson and Hugo Zaragoza (2009),
  [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019).
- Gordon V. Cormack, Charles L. A. Clarke and Stefan Büttcher (2009),
  [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf).
- Liang Wang and colleagues (2024),
  [Multilingual E5 Text Embeddings: A Technical Report](https://arxiv.org/abs/2402.05672).
  The optional unchanged multilingual-e5-small model uses revision
  `614241f622f53c4eeff9890bdc4f31cfecc418b3`, 384 dimensions, local source-offset
  token windows of 384 with overlap 64 and fixed query/passage prefixes.
- Hosted v4 integration follows the provider's
  [embedding API documentation](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api).
  Native query/document modes are isolated research controls; the core runtime
  continues to use its explicit compatible embeddings route.

Public-dev complete-evidence recall, token-prefix retention and local answer
proxies measure different properties. None is an official AML score. This
v1.1 source release does not claim completed QA for the current v4
configuration, official evaluation, online capacity or competition rank. No new retrieval algorithm is added by this
publication preparation.

The queuefix update permits HTTP queue waits up to 900 seconds with the selected
600-second operation profile. It changes configuration validation only, not
ranking, source packing, segmentation, embeddings, operation timing or slot
release. Existing QA/retrieval caches retain their earlier semantic-source hash;
source-version equivalence must not be inferred from ranking equality alone.
