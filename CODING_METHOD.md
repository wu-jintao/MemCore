# MemCore Coding v1

MemCore Coding is an isolated Coding Memory profile for the AML `coding-v1`
contract. It preserves the proven synchronous Add/Search persistence, complete
user isolation, request-ID idempotency, SQLite/WAL storage, BM25, dense vector
retrieval, and reciprocal-rank fusion used by MemCore Textual, while adding
engineering-specific retrieval channels.

The Coding profile is enabled only with `MEMORY_PROFILE=coding`. The default
`textual` profile retains the published MemCore v1.1 behavior. The Coding
candidate uses a fresh database and an independent deployment; it does not read,
modify, migrate, or route the Textual evaluation database.

## Retrieval channels

1. **Compact engineering BM25.** Raw source evidence remains stored unchanged.
   BM25 indexes a deterministic source-derived representation containing the task
   description, changed files and diff lines, tests, exceptions, task type, and
   bounded engineering features. This avoids indexing repeated unchanged patch
   context while retaining identifiers needed for repository work.
2. **Coding feature retrieval.** A compact SQLite index stores repository-relative
   paths, qualified symbols, exceptions, test identifiers, commands, task intent,
   root-cause markers, patch markers, and verification signals. Query confidence
   dampens this channel for broad environment dumps and indirect task relations.
3. **Dense engineering retrieval.** `text-embedding-v4` embeds deterministic
   compact engineering representations. Raw source text is still returned to the
   AML Coding agent. Complete UTF-8 inputs are segmented without truncation.
4. **Content-addressed embedding reuse.** Coding document vectors are cached by
   the fixed semantic fingerprint and exact compact-representation SHA-256. The
   cache stores no source text and never participates directly in Search. Every
   user still receives independent scoped vector rows; identical histories across
   relevant/noisy attempts avoid repeated model calls without weakening isolation.
5. **Weighted RRF.** Lexical weight is 1.0, the Coding feature weight is 2.0
   before query-confidence scaling, and the dense weight is 3.0. The public proxy
   selected these weights by high-noise Recall@1/MRR before average metrics.

Search returns at most twelve complete source evidence items and stops only at a
whole-item boundary before 180,000 characters. It does not summarize or generate
the final answer, patch, or code solution.

## Public proxy evidence

The scored CAMBench Coding data is private and was not inspected. Development
used the public SWEContextBench Lite experience/related/relationship files as a
proxy for retrieval of reusable engineering experience under relevant/noisy
conditions. At 96 adversarial distractors per target, the real MemoryStore
Add/Search path achieved:

- Recall@1: 76.77%
- Recall@3: 87.88%
- Recall@8: 96.97%
- Recall@12: 97.98%
- MRR: 83.07%
- Mean returned evidence: about 103K characters

These are public proxy retrieval metrics, not an official CAMBench score and not
a claim about Coding-agent task resolution. A cached-vector output-budget comparison
used the same 99 high-noise public tasks: moving from 8 / 120,000 characters to
12 / 180,000 recovered all public relevant experiences for two additional tasks
(95/99 versus 93/99) while keeping mean returned content near 103K characters.
Larger limits added substantially more distractors, so the fixed candidate stops at 12.

## Capacity and integrity

A 9,604-message, 96-distractor public proxy database used about 349 MB after
compact index changes, versus 1.28 GB for the first prototype. The real HTTP
runtime passed 16 concurrent Add, 16 concurrent Search, 32 concurrent identical
Add retries, complete-user isolation, deterministic results, SIGKILL restart,
and SQLite `quick_check` recovery using a deterministic synthetic 2048-dimensional
backend. No model or non-loopback call was made by that runtime verification.
