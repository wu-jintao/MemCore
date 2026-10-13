# MemCore Coding v1.1

MemCore Coding is an isolated Coding Memory profile for the AML `coding-v1`
contract. It preserves synchronous Add/Search persistence, complete user
isolation, request-ID idempotency, SQLite/WAL storage, compact engineering BM25,
engineering feature retrieval, dense vector retrieval, and weighted reciprocal-
rank fusion. Coding v1.1 adds a safe dense segment post-reranker after evidence
selection and character packing.

The Coding profile is enabled only with `MEMORY_PROFILE=coding`. The default
`textual` profile retains the published MemCore v1.1 behavior. The Coding
candidate uses a fresh database and an independent deployment; it does not read,
modify, migrate, or route the Textual evaluation database.

## Retrieval pipeline

1. **Compact engineering BM25.** Raw source evidence remains stored unchanged.
   BM25 indexes a deterministic representation containing the task description,
   changed files and diff lines, tests, exceptions, task type, and bounded
   engineering features.
2. **Coding feature retrieval.** A compact SQLite index stores repository-relative
   paths, qualified symbols, exceptions, test identifiers, commands, task intent,
   root-cause markers, patch markers, and verification signals. Query confidence
   dampens this channel for broad environment dumps and indirect relations.
3. **Dense engineering retrieval.** `text-embedding-v4` embeds deterministic
   compact engineering representations. Complete UTF-8 inputs are segmented
   without truncation, while raw source text is returned to the Coding agent.
4. **Content-addressed document-vector reuse.** Identical compact representations
   reuse vectors by semantic fingerprint and SHA-256. Every user still receives
   independently scoped vector rows, and Search always filters by the complete
   requested user ID.
5. **Primary weighted RRF.** Lexical weight is 1.0, Coding feature weight is 2.0
   before confidence scaling, and dense weight is 3.0.
6. **Safe dense segment post-reranking.** V1 first selects and packs at most 12
   complete evidence items under the 180,000-character budget. V1.1 then reorders
   only that fixed set. Dense rank uses the maximum cosine across every query
   segment and every stored document segment; a 0.5-weight reciprocal-rank term
   is fused with the original selected order. The evidence set, source content,
   count, and character total cannot change at this stage.

The query mean and original query segments come from one embedding operation.
The reranker makes no second model request and stores no new vectors. Search does
not summarize or generate the final answer, patch, or code solution.

## Public evidence

The scored CAMBench Coding data is private and was not inspected. Method
selection used public SWEContextBench Lite tasks; the remaining public Full tasks
were held out for one-shot validation. On 258 independent validation tasks:

| Metric | Coding v1 | Coding v1.1 |
| --- | ---: | ---: |
| Recall@1 | 70.93% | **72.09%** |
| Recall@3 | 85.66% | **87.21%** |
| Recall@8 | 95.74% | **96.12%** |
| Recall@12 | 98.06% | **98.06%** |
| MRR | 80.09% | **80.96%** |
| Tasks with all related experience returned | 253/258 | **253/258** |
| Mean returned characters | 117,422.64 | **117,422.64** |

A formal implementation replay covered all 357 public Full tasks. For every task,
the service order matched an independent formula implementation, and the selected
ID set, raw content, returned count, and character total matched v1. Duplicate
query-text caches with small numerical variation produced identical order and
metrics. These are public proxy retrieval results, not an official CAMBench or
AML score.

## Capacity and integrity

The v1 database and vector schema are unchanged. The reranker reads at most the
stored segment vectors for the already packed evidence items, so it adds no index,
no migration, and no extra persistent storage. Existing 128 MiB authenticated
request handling, bounded embedding queues, process restart controls, idempotency,
and user isolation remain in force. A fresh loopback-only v1.1 candidate passed
30/30 bounded real-model requests, a two-segment query, and controlled restart
recovery without changing the public route.
