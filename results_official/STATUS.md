# Official-run status

Completed: 2026-09-05 (Asia/Shanghai).

- Environment, model and upstream-version audits: complete.
- GraphRAG-Bench Medical: pinned and frozen into aligned P0/P1 splits.
- Generation/extraction: Qwen2.5-VL-7B-Instruct, text-only, temperature 0.
- Embedding: BAAI/bge-m3 for all three official backends.
- P0: Vector RAG, LightRAG and PathRAG each completed 10/10 questions.
- P1: all three backends each completed 120/120 questions with no runtime errors.
- Official evaluation: 390/390 P0+P1 backend-question rows have valid metrics and
  structurally valid factuality classifications.
- P1 analysis: silver labels, TF-IDF router, fixed/adaptive/oracle comparisons,
  confusion matrix and representative cases are complete.
- Tests: 18/18 pass.
- Active experiment/evaluation processes: none.

Headline P1 test-set Answer Correctness (24 questions): Vector 0.7135,
LightRAG 0.3524, PathRAG 0.4908, Adaptive 0.7135 and Oracle 0.7266. The
router sends all test questions to Vector and does not outperform the majority
route; this limitation is reported rather than hidden.

The 20-question manual-review template is generated, but independent human
verdicts are still pending. See `report/OFFICIAL_PHASE1_RESULTS.md` for the
complete results and artifact paths.

No proxy row has been copied into this directory as an official backend result.
