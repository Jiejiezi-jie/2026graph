# Model and server audit

Audit time: 2026-09-04 (Asia/Shanghai). Free-memory values are a point-in-time
snapshot and may change on this shared machine.

## Compute and software

| Item | Observed state |
|---|---|
| GPU | 6 x NVIDIA GeForce RTX 4090, 23.52 GiB CUDA-visible memory each |
| GPU memory free (0..5) | 3.25, 22.62, 5.11, 1.49, 10.84, 10.49 GiB |
| GPU diagnostic caveat | host `nvidia-smi` fails: kernel driver 580.95.05 vs NVML 580.126; PyTorch 2.6.0+cu124 in `qwen_saliency` nevertheless detects and uses all six GPUs |
| CPU | 2 x Intel Xeon Gold 6330; 56 physical cores / 112 threads |
| RAM | 251 GiB total, 205 GiB available; 8 GiB swap, 3.3 GiB free |
| Disk | 29 TiB filesystem, 6.4 TiB free (78% used) |
| Base Python | 3.13.12; no project ML/scientific packages installed |
| CUDA toolkit | nvcc 12.0 |
| Useful existing env | `qwen_saliency`: Python 3.11.15, PyTorch 2.6.0+cu124, Transformers 5.13.0.dev0 |
| Other useful env | `obliviate`: Python 3.10.20, sklearn 1.7.2, pandas 2.3.3, Transformers 4.57.0; its cu128 PyTorch cannot initialize with this driver |
| Local serving software | no Ollama, vLLM, or SGLang executable in PATH; no matching inference process found |
| Listening endpoint | `127.0.0.1:12345` returns HTTP 400 to `/`, `/health`, `/api/tags`, and `/v1/models`; it is not a usable OpenAI-compatible endpoint |
| Project environment | `.venv_official`, based on the CUDA-working `qwen_saliency` interpreter with system-site packages and isolated LightRAG/evaluation dependencies |

No credential values were read or printed.

## Local model inventory and suitability

| Purpose | Candidate | Path/interface | Approx. weight/disk size | Available | Notes |
|---|---|---|---:|---|---|
| Entity/relation extraction | Qwen2.5-VL-3B-Instruct | `/home/user/wangyuhan/models/Qwen2.5-VL-3B-Instruct` | 7.1 GiB | Rejected | 5/5 QA were non-empty, but strict structured output was truncated/unstable; retained only as an audit artifact. |
| Entity/relation extraction | Qwen3-VL-4B-Instruct | `/home/user/wangyuhan/models/Qwen3-VL-4B-Instruct` | 8.6 GiB | Weights complete; fallback | Larger/newer VL fallback; also needs text/JSON probe. |
| Final answer generation | Qwen2.5-VL-3B-Instruct | same path | 7.1 GiB | Rejected | Too weak for the required structured extraction reliability. |
| Final answer generation | Qwen2.5-VL-7B-Instruct | `/home/user/wangyuhan/models/Qwen2.5-VL-7B-Instruct` | 16 GiB | **Probe passed** | 5/5 concise medical QA responses and 5/5 strict JSON extraction outputs; selected for P0 at temperature 0 and 256 final-answer tokens. |
| Final answer generation | Vicuna 7B v1.1 | `/home/user/wangyuhan/models/vicuna-7b-v1.1` | 13 GiB | Weights complete | Text-only, but substantially older and less reliable for structured extraction than Qwen candidates. |
| Embedding | BAAI/bge-m3 | `/home/user/wangyuhan/models/bge-m3` | 2.27 GB main weights | **Downloaded and validated** | Selected for all three backends. Native context limit covers the shared 1,200-token chunks. A real 4-chunk probe returned normalized 1024-d vectors and peaked at 2.36 GiB on GPU 4. |
| Embedding fallback | BAAI/bge-large-en-v1.5 | `/home/user/wangyuhan/models/bge-large-en-v1.5` | 1.34 GB main weights | Downloaded and validated | Valid 1024-d normalized embeddings, but its 512-token limit would force LightRAG to split the shared 1,200-token chunks; retained only for a possible short-chunk ablation. |
| Router baseline | TF-IDF + Logistic Regression | project code / sklearn 1.7.2 | negligible | Yes | Required reproducible baseline; selected on validation Macro-F1 only. |
| Neural router | T5-Small/Base | no local checkpoint found | n/a | No | Optional only after a separate model download; not needed to unblock the required lightweight router. |
| Independent judge | Qwen2.5-VL-7B-Instruct | local 7B path above | 16 GiB | Technically present, not yet validated | Same model family and VL architecture, so independence is weak. Official Answer Correctness judging should be reported with this bias or replaced by a stronger separate text model. |

The Hugging Face cache contains a partial Qwen2-VL-72B snapshot and
`bert-base-uncased`, but neither is a usable dedicated sentence embedding model;
the 72B cache is only 2.6 GiB and does not contain complete weights. The local
`instructblip-vicuna-7b` directory is also incomplete (1.6 GiB).

## Current selection and blockers

- Generation/extraction model: Qwen2.5-VL-7B-Instruct in text-only mode. It
  passed 5/5 answer and 5/5 strict-JSON checks (1,732 input and 764 output
  tokens over ten calls). The 3B checkpoint is not used for official indexing
  or answers. Raw probes are saved under `results_official/model_probe_*.json`.
- Embedding model: BAAI/bge-m3, shared by Vector RAG, LightRAG and PathRAG.
  It was fetched from the BAAI ModelScope mirror at commit
  `a46a13810c9d7f876fccd8d7017512ebc265d2c1`; the main weight SHA-256 is
  `b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`.
  No proxy or VLM hidden-state substitute is used.
- Router: TF-IDF + Logistic Regression first; no test labels used for tuning.
- GPU assignment: Qwen2.5-VL-7B on GPU 1 with an 18 GiB free-memory preflight
  floor; BGE-M3 on GPU 4 with a 6 GiB floor. The user authorized the sustained
  P0/P1 workload after the concrete memory and runtime estimate was reported.
