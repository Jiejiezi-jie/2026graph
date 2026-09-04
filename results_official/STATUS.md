# Official-run status

- Environment and archive audit: complete.
- GraphRAG-Bench Medical pin: verified.
- LightRAG and PathRAG official source pins: verified; third-party trees unmodified.
- Qwen2.5-VL-7B text QA probe: 5/5 non-empty.
- Qwen2.5-VL-7B strict entity/relation JSON probe: 5/5 valid.
- Core/project tests: 11/11 passed.
- Dedicated embedding model: BAAI/bge-m3 downloaded; weight hash and a real
  4 x 1,200-token chunk embedding probe passed (1024 dimensions, unit norms,
  2.36 GiB peak allocation on GPU 4).
- Vector P0: 10/10 answers and contexts, zero runtime errors; 199-chunk index
  built in 18.85 seconds, with 10 online LLM calls.
- LightRAG P0: official index construction is in progress; no LightRAG result
  is counted until all ten aligned queries complete.
- PathRAG P0: pending LightRAG completion.
- P1 official 120-question run: not started; it cannot precede P0 acceptance.

No proxy row has been copied into this directory as an official backend result.
