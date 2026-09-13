# Current-API-batch rerun report

## Scope

- Baseline questions: 120 (`train=72`, `validation=24`, `test=24`).
- Rerun with the current API/model batch: Vector and LightRAG, 120 rows each.
- Reused from the same-day batch previously identified as current: PathRAG
  baseline (120 rows) and the three 180-row expansion outputs. These were not
  called again.
- The configured API model identifier is `deepseek-chat`. No API key is stored in
  this report, configuration, output, or log. Because neither the credential nor
  the provider-side model variant is stored, the PathRAG artifact cannot
  independently prove whether the provider mapped that alias to Flash or another
  internal variant.
- All three methods use the same shared LightRAG graph. The graph SHA-256 is
  `4c2f2c35c00b94aa8eb4a728572ffbebc66ec31c2ad5caa6c58546613fdc1a21`.

## Completion and integrity

| Item | Result |
|---|---:|
| Vector generation | 120/120 |
| LightRAG generation | 120/120 |
| Vector official evaluation | 120/120 valid |
| LightRAG official evaluation | 120/120 valid |
| Frozen test questions | 24 |
| Router development questions | 276 |
| New Vector answer calls | 120 |
| New LightRAG calls | 240 (120 retrieval-keyword + 120 answer) |

The official evaluation output contains four successful judge operations per
answer row. Vector evaluation emitted all 120 valid rows before a CUDA cache
cleanup error; that post-write cleanup error did not invalidate or omit any row.

Two explicitly enumerated evaluation protocol hashes occur in the merged input:

- `209096a999b2a761cfd08eb9ddadd88a804f51ecd79acf8a61a8b592ed9fd851`
- `d1f6aaf676bddfd08f81e0e7ccafc78fa3cbe92e8d50bbd47687a775cf2d6abd`

They use the same official metric implementation, BGE-M3 embeddings, API model
identifier, and temperature. The hash difference is caused by the local embedding
device/runtime fields in the new configuration. The merge accepts only these two
explicit hashes rather than disabling protocol validation.

## Frozen 24-question test results

| Method | Answer Correctness | Evidence Recall | Average context words | Average input tokens | Average output tokens |
|---|---:|---:|---:|---:|---:|
| Vector | 0.5349 | 0.6826 | 4,747.9 | 4,985.4 | 51.5 |
| LightRAG | 0.5164 | 0.5479 | 4,779.5 | 5,678.7 | 83.2 |
| PathRAG | 0.5815 | 0.5215 | 1,909.3 | 5,481.2 | 114.0 |
| Adaptive (multi-output Ridge) | 0.5727 | 0.6424 | 3,601.6 | 5,421.1 | 83.9 |

Adaptive routes the 24 frozen questions as `Vector/LightRAG/PathRAG = 6/8/10`.
The current multi-objective router improves evidence recall relative to LightRAG
and PathRAG, but it does not beat the best fixed-method answer correctness
(PathRAG, 0.5815) or the fixed Vector evidence recall (0.6826). This is therefore
not evidence of a universal improvement.

## End-to-end latency on the frozen test set

All values below are milliseconds per question. Retrieval and answer generation
were recorded during the actual run; total includes their small orchestration
overhead. PathRAG timings are reused from its current-model run.

| Method | Retrieval mean / P50 / P95 | Generation mean / P50 / P95 | Total mean / P50 / P95 |
|---|---:|---:|---:|
| Vector | 30.8 / 36.7 / 39.6 | 1,145.2 / 1,139.8 / 1,501.5 | 1,176.0 / 1,167.5 / 1,535.2 |
| LightRAG | 1,151.8 / 1,084.1 / 1,671.9 | 1,104.7 / 1,004.3 / 1,668.2 | 2,256.6 / 2,206.2 / 2,883.8 |
| PathRAG | 3,830.2 / 4,126.8 / 5,835.1 | 1,045.1 / 996.8 / 1,629.2 | 4,878.0 / 5,112.8 / 6,800.5 |
| Adaptive | 1,757.9 / 1,161.1 / 4,547.1 | 1,053.0 / 1,001.0 / 1,596.8 | 2,812.1 / 2,443.8 / 5,986.9 |

## Router inference benchmark

Environment: one RTX 4090 (GPU 5), local BGE-M3 (1,024 dimensions) plus the
multi-output Ridge head. This measures routing only and excludes retrieval and
answer generation. Multiple GPUs are useful for parallel offline embedding, but
single-query latency is intentionally measured on one GPU.

| Scenario | Mean | P50 | P95 |
|---|---:|---:|---:|
| One question, BGE-M3 + Ridge | 14.24 ms | 11.64 ms | 13.07 ms |
| 24-question batch | 78.94 ms | 74.12 ms | 100.05 ms |
| 276-question batch | 880.00 ms | 884.10 ms | 905.07 ms |
| Ridge head only, cached embeddings, 24 rows | 0.117 ms | 0.103 ms | 0.147 ms |

The mean for one-question routing is above P95 because a small number of runtime
outliers affect the arithmetic mean. Batched throughput is about 3.29 ms/question
for 24 questions and 3.19 ms/question for 276 questions. Cold loading BGE-M3 plus
the Ridge model took 7.64 s and should be amortized by keeping the service warm.

## Router training result after alignment

- Training/development set: 276 question-only BGE-M3 embeddings.
- Model: six-output Ridge regression, selected `alpha=0.001`.
- Development-selected evidence utility weight: `0.0`.
- Development OOF versus the 300-question weighted hard-label reference:
  Answer Correctness `0.6204 -> 0.6296`; Evidence Recall `0.5499 -> 0.5875`.
- Frozen-test Adaptive result: Answer Correctness `0.5727`, Evidence Recall
  `0.6424`.
- Model SHA-256:
  `8801ee7f3a06ca575e1225a2d24836f481eb1139c366fcb0cc675309240c3a80`.

The previously reported `0.6762 / 0.6424` Adaptive result mixed answer-generation
model batches and is not used as the current-batch result. The current result is
the table above, subject to the PathRAG provider-variant provenance caveat.
