"""Official RAG backends, shared evaluation utilities and the Medical workbench app.

Layout:
    app/      FastAPI + retrieval workbench (Medical bundle driven)
    scripts/  workbench entry points (serve_web.py, prepare_medical.py, ...)
    common/   shared data loading, model clients, official scoring and runners
    vector/   Vector baseline adapter and its runner
    lightRAG/ LightRAG adapter, runner and graph export probes
    pathRAG/  PathRAG adapter, runner, tuning and graph export probes
"""
