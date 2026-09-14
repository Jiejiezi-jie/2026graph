# Server integration notes

## Phase-one prototype

The archive was inspected before extraction. It contains 33 entries (4,054,892
uncompressed bytes), with no absolute paths, `..` components, or symbolic-link
entries. It was extracted into this new directory; the original ZIP and all
`results/` proxy artifacts remain untouched.

The prototype loads `medical.json` and `medical_questions.json` from the
GraphRAG-Bench checkout, keeps Fact Retrieval and Complex Reasoning questions,
and splits the single Medical corpus into 180-word windows with a 40-word
overlap. The archived full run used all 1,607 eligible questions and seed 42.

`OfflineGraphIndex` is explicitly a protocol proxy:

- `vector` ranks text windows with word/bi-gram TF-IDF and returns the top 3.
- `light_proxy` builds a nearest-neighbour chunk graph, runs personalized graph
  diffusion from TF-IDF seed chunks, and returns the top 5 combined scores.
- `path_proxy` decomposes a question with lexical delimiters, locates seed
  chunks, adds shortest-path flow over the same chunk graph, and returns the
  top 7 combined scores.

All three produce an extractive `answer_proxy`. Evidence quality is a lexical
coverage score over reference-evidence terms. A route is eligible when its
coverage is within 0.02 of the best route, and the first eligible route in the
cost order `vector -> light_proxy -> path_proxy` becomes the silver label. A
word/character TF-IDF Logistic Regression classifier is tuned on the dev set by
Macro-F1, then fitted on train+dev. Dataset question types are not router input;
the type-to-route mapping is only an archived ablation.

## Replacement boundary

The following parts must be replaced for official results:

- TF-IDF vector retrieval must use a dedicated embedding model and an answer
  generation LLM.
- Chunk-graph diffusion must be replaced by HKUDS LightRAG graph extraction and
  `hybrid` retrieval.
- Lexical clause/path flow must be replaced by BUPT-GAMMA PathRAG's real
  entity-path enumeration and pruning.
- `answer_proxy`, lexical evidence coverage, and proxy silver labels must not be
  presented as official generation metrics or reused as official labels.

The reusable parts are Medical data loading, deterministic sampling/splitting,
basic text normalization, ROUGE-L support, TF-IDF + Logistic Regression router
construction, policy aggregation/plots, and the existing unit-test style. New
official code and outputs live in `src/backend/` and
`result/official/`, respectively.

## Pinned upstream sources

| Component | Repository | Pin | Local path |
|---|---|---|---|
| GraphRAG-Bench | `https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git` | `fdbab5959b18c96532580877ffe27d112bccc0ec` | `data/vendor/GraphRAG-Benchmark` |
| LightRAG | `https://github.com/HKUDS/LightRAG.git` | tag `v1.5.7`, `28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c` | `deps/LightRAG` |
| PathRAG | `https://github.com/BUPT-GAMMA/PathRAG.git` | `32567bfc93605b8393996d5fa9ccdc0edbb865b2` | `deps/PathRAG` |

LightRAG v1.5.7 exposes structured `aquery_data`, so no core patch is required
to save entities, relationships, and chunks. The current PathRAG checkout has
no packaging metadata and omits the documented `RAGRunner`; its `llm.py` also
eagerly imports every optional provider. The project-side adapter therefore
injects only the two unused default-provider symbols before importing PathRAG,
then supplies the real local model and embedding callbacks. This is an import
compatibility shim; PathRAG's extraction, path enumeration, alpha=0.8
propagation, threshold=0.3 pruning, and top-path selection remain upstream code.

