# Current-API-batch router strategy comparison

## Protocol

- Total evaluated questions: 300.
- Router development set: 276 questions, evaluated with fixed stratified 5-fold OOF.
- Frozen test set: 24 questions.
- Router inputs: question text only (TF-IDF or cached 1,024-dimensional BGE-M3
  question embeddings).
- Method outcomes: current-credential Vector and LightRAG plus the same-day reused
  PathRAG results, all on the same shared graph/corpus setup. All artifacts record
  `deepseek-chat`; the stored PathRAG artifact cannot independently prove the
  provider-side model variant because credentials are intentionally not recorded.
- No retrieval, answer-generation, judge, or other API call was made for this
  comparison.

`H` below is the joint harmonic mean of aggregate Answer Correctness (`AC`) and
Evidence Recall (`ER`): `H = 2 * AC * ER / (AC + ER)`. It is used only as a
balanced comparison statistic.

## 1. Hard labels: 12 settings

| Threshold | Margin | Feature | 5-fold Accuracy | 5-fold Macro-F1 | OOF AC | OOF ER | OOF H | Test Accuracy | Test Macro-F1 | Test AC | Test ER | Test H | Test routes V/L/P |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.05 | TF-IDF | 0.4022 | 0.3252 | 0.6022 | 0.5683 | 0.5848 | 0.4583 | 0.3032 | 0.5292 | 0.5965 | 0.5609 | 14/3/7 |
| 0.50 | 0.05 | BGE-M3 | 0.5399 | 0.3725 | 0.6245 | 0.6163 | 0.6204 | 0.6667 | 0.4358 | 0.5737 | 0.6257 | 0.5986 | 17/0/7 |
| 0.50 | 0.10 | TF-IDF | 0.4601 | 0.3641 | 0.6104 | 0.5925 | 0.6013 | 0.5000 | 0.3111 | 0.5303 | 0.6312 | 0.5764 | 16/2/6 |
| **0.50** | **0.10** | **BGE-M3** | **0.5688** | **0.3743** | **0.6174** | **0.6479** | **0.6323** | **0.7500** | **0.4931** | **0.5749** | **0.6465** | **0.6086** | **18/0/6** |
| 0.50 | 0.15 | TF-IDF | 0.4638 | 0.3526 | 0.6050 | 0.5846 | 0.5946 | 0.5833 | 0.3862 | 0.5464 | 0.6174 | 0.5797 | 14/1/9 |
| 0.50 | 0.15 | BGE-M3 | 0.4891 | 0.3906 | 0.6125 | 0.5943 | 0.6033 | 0.5833 | 0.4183 | 0.5733 | 0.6257 | 0.5983 | 13/3/8 |
| 0.60 | 0.05 | TF-IDF | 0.3804 | 0.3110 | 0.5991 | 0.5602 | 0.5790 | 0.4167 | 0.2836 | 0.5153 | 0.5382 | 0.5265 | 12/3/9 |
| 0.60 | 0.05 | BGE-M3 | 0.4022 | 0.3598 | 0.6164 | 0.6027 | 0.6095 | 0.5000 | 0.4520 | 0.5868 | 0.5632 | 0.5748 | 8/7/9 |
| 0.60 | 0.10 | TF-IDF | 0.4312 | 0.3465 | 0.6053 | 0.5766 | 0.5906 | 0.4583 | 0.3086 | 0.5421 | 0.5965 | 0.5680 | 13/2/9 |
| 0.60 | 0.10 | BGE-M3 | 0.4348 | 0.3753 | 0.6114 | 0.6177 | 0.6145 | 0.5000 | 0.3827 | 0.5710 | 0.6257 | 0.5971 | 11/5/8 |
| 0.60 | 0.15 | TF-IDF | 0.4710 | 0.3747 | 0.6027 | 0.5826 | 0.5925 | 0.5417 | 0.3791 | 0.5486 | 0.6174 | 0.5809 | 13/3/8 |
| 0.60 | 0.15 | BGE-M3 | 0.4384 | 0.3730 | 0.6104 | 0.6084 | 0.6094 | 0.5417 | 0.4050 | 0.5745 | 0.5840 | 0.5792 | 10/4/10 |

## 2. Same hard labels with sample down-weighting: 12 settings

Weights are `stable=1.0`, `ambiguous=0.5`, and `all_failed=0.3`.

| Threshold | Margin | Feature | 5-fold Accuracy | 5-fold Macro-F1 | OOF AC | OOF ER | OOF H | Test Accuracy | Test Macro-F1 | Test AC | Test ER | Test H | Test routes V/L/P |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.05 | TF-IDF | 0.3768 | 0.3072 | 0.6014 | 0.5616 | 0.5808 | 0.4583 | 0.3111 | 0.5293 | 0.5799 | 0.5534 | 13/4/7 |
| 0.50 | 0.05 | BGE-M3 | 0.3986 | 0.3373 | 0.6117 | 0.6009 | 0.6062 | 0.3333 | 0.2523 | 0.5183 | 0.5590 | 0.5379 | 9/6/9 |
| 0.50 | 0.10 | TF-IDF | 0.4928 | 0.3431 | 0.6161 | 0.5727 | 0.5936 | 0.5000 | 0.2984 | 0.5303 | 0.6312 | 0.5764 | 17/0/7 |
| 0.50 | 0.10 | BGE-M3 | 0.5072 | 0.3558 | 0.6170 | 0.5704 | 0.5928 | 0.6250 | 0.4212 | 0.5827 | 0.6049 | 0.5936 | 12/0/12 |
| 0.50 | 0.15 | TF-IDF | 0.5109 | 0.3397 | 0.6064 | 0.5898 | 0.5980 | 0.5417 | 0.3399 | 0.5432 | 0.6174 | 0.5779 | 16/0/8 |
| 0.50 | 0.15 | BGE-M3 | 0.5507 | 0.3698 | 0.6197 | 0.6254 | **0.6225** | 0.6667 | 0.4405 | 0.5737 | 0.6257 | 0.5986 | 16/0/8 |
| 0.60 | 0.05 | TF-IDF | 0.4094 | 0.2950 | 0.6076 | 0.5179 | 0.5591 | 0.5000 | 0.3460 | 0.5454 | 0.5215 | 0.5332 | 10/0/14 |
| 0.60 | 0.05 | BGE-M3 | 0.3841 | 0.3320 | 0.6101 | 0.5923 | 0.6011 | 0.5000 | 0.3876 | 0.5920 | 0.6049 | 0.5983 | 8/5/11 |
| 0.60 | 0.10 | TF-IDF | 0.4493 | 0.3195 | 0.6038 | 0.5281 | 0.5634 | 0.4583 | 0.3065 | 0.5209 | 0.5340 | 0.5274 | 12/0/12 |
| 0.60 | 0.10 | BGE-M3 | 0.4819 | 0.3429 | 0.6204 | 0.5499 | 0.5830 | 0.6667 | 0.4521 | **0.6009** | **0.6049** | **0.6029** | 11/0/13 |
| 0.60 | 0.15 | TF-IDF | 0.4348 | 0.3401 | 0.6019 | 0.5768 | 0.5891 | 0.4583 | 0.3348 | 0.5453 | 0.5965 | 0.5697 | 11/4/9 |
| 0.60 | 0.15 | BGE-M3 | 0.5290 | 0.3742 | 0.6236 | 0.5660 | 0.5934 | 0.6667 | 0.4521 | **0.6009** | **0.6049** | **0.6029** | 11/0/13 |

## 3. Strategy-level comparison

The table retains the development-selected representative and also marks the
operating point selected for the course demo after the full comparison.

| Strategy | Development-selected setting | OOF AC | OOF ER | OOF H | Test AC | Test ER | Test H | Test routes V/L/P |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Hard labels | threshold 0.50, margin 0.10, BGE-M3 | 0.6174 | 0.6479 | **0.6323** | 0.5749 | **0.6465** | **0.6086** | 18/0/6 |
| Sample down-weighting (development-selected) | threshold 0.50, margin 0.15, BGE-M3 | 0.6197 | 0.6254 | 0.6225 | 0.5737 | 0.6257 | 0.5986 | 16/0/8 |
| **Sample down-weighting (project-selected)** | **threshold 0.60, margin 0.10, BGE-M3** | 0.6204 | 0.5499 | 0.5830 | **0.6009** | 0.6049 | 0.6029 | 11/0/13 |
| Multi-output metric regression | Ridge alpha 0.001, evidence weight 0.0 | **0.6296** | 0.5875 | 0.6078 | 0.5727 | 0.6424 | 0.6055 | 6/8/10 |
| Exploratory balanced-utility labels | min(AC, ER), gap-weighted logistic regression | 0.6071 | **0.7138** | **0.6562** | 0.5228 | 0.6826 | 0.5921 | 23/0/1 |

The exploratory balanced-utility candidate was selected from 540 candidates using
development OOF only. It has the strongest development `H`, but on the frozen test
set it collapses to almost always choosing Vector. Its test `H=0.5921` is below the
three established representatives, so it is not recommended.

## Conclusion

The operating point selected for the course demo is:

> **Sample down-weighting, correctness threshold 0.60, best-method margin 0.10,
> BGE-M3.**

It has the highest observed frozen-test Answer Correctness in the 24-row grid
(`0.6009`), with Evidence Recall `0.6049`, joint `H=0.6029`, and routes
`Vector/LightRAG/PathRAG = 11/0/13`. Margin 0.15 produces the same frozen-test
routes and downstream scores; margin 0.10 is retained as the requested operating
point.

The development-only statistical selection remains hard labels at
`threshold=0.50`, `margin=0.10`, BGE-M3 (`OOF H=0.6323`). Therefore the selected
sample-weighted setting must be described as a post-comparison, correctness-first
operating choice rather than an untouched-test model selection.

Because all 24 configurations have now been repeatedly inspected on the 24 test
questions, future method development should treat those questions as an analysis
set and reserve a new untouched test set for the final reported comparison.

## Retrieval effect and routing latency of the recommended router

This section uses the selected sample-weighted `threshold=0.60`, `margin=0.10`,
BGE-M3 router. Evidence coverage is the official test Evidence Recall. The
serialized result's `contexts` field contains only source chunks, while graph
entities, relations, and paths are stored separately. Consequently, source-only
word counts must not be presented as total PathRAG context size.

| Strategy | Evidence Recall | Source-chunk words | Approx. full retrieved payload words | Recorded API input tokens | End-to-end latency |
|---|---:|---:|---:|---:|---:|
| Vector | 0.6826 | 4,747.9 | 4,749.1 | 4,985.4 | 1.176 s |
| LightRAG | 0.5479 | 4,779.5 | 6,945.6 | 5,678.7 | 2.257 s |
| PathRAG | 0.5215 | 1,909.3 | 5,484.8 | 5,481.2 | 4.878 s |
| Adaptive | 0.6049 | 3,215.8 | 5,179.5 | 5,248.8 | 3.233 s |

The approximate full payload is a whitespace count over serialized `contexts`,
`entities`, `relations`, and `paths`; it is useful diagnostically but is not an
exact reconstruction of the generation prompt. Recorded API input tokens are
also not a pure context-size metric: graph backends make a retrieval/keyword LLM
call before answer generation, and the stored total covers both calls. The answer
generation adapter caps its supplied context at 5,000 tokens. A strictly fair
context comparison requires separately logging generation-call prompt tokens in
a future rerun.

Adaptive selected `Vector/LightRAG/PathRAG = 11/0/13`. Its measured selected-RAG
latency is 3.222 s/question; the table adds the 11.60 ms mean online routing time.

Routing was benchmarked on one RTX 4090 (GPU 5) with local BGE-M3 embeddings
(1,024 dimensions), scikit-learn 1.7.2, and a 25,535-byte logistic-regression
head. Times include BGE-M3 encoding and classifier prediction.

| Routing scenario | Mean | P50 | P95 |
|---|---:|---:|---:|
| One question | 11.597 ms | 11.513 ms | 12.761 ms |
| 24-question batch | 85.360 ms | 83.535 ms | 88.709 ms |
| 62-question batch | 224.460 ms | 220.623 ms | 238.608 ms |
| 276-question batch | 952.314 ms | 947.540 ms | 975.418 ms |

The mean batch cost is 3.557 ms/question for 24 questions, 3.620 ms/question
for 62 questions, and 3.450 ms/question for 276 questions. One cold load of
BGE-M3 plus the classifier took 8.652 s;
production inference should keep the model resident. Routing time excludes RAG
retrieval and answer generation, while the first table's end-to-end latency
includes them.
