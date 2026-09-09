# LightRAG Phase 1 Risk Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested Python CLI that selects the shortest GraphRAG-Bench novel, builds or reuses one LightRAG workspace, calls `aquery_llm()` once, and prints the answer and structured retrieval data.

**Architecture:** All new code stays in the application repository. The existing GraphRAG-Benchmark checkout is a read-only corpus source; the existing LightRAG checkout and Conda environment are a read-only runtime dependency. Corpus loading, workspace lifecycle, LightRAG construction, and result adaptation are separate modules that the later FastAPI layer can reuse.

**Tech Stack:** Python 3.10.21; LightRAG 1.5.7 at `c1248646e4eda4d89054926af2e094730daf23fe`; Pydantic 2; pydantic-settings; Transformers; BAAI/bge-small-en-v1.5; pytest; pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-09-03-lightrag-graphrag-bench-platform-design.md`

## Global Constraints

- Write only in `D:\HuaweiMoveData\Users\huawei\Desktop\AI知识库\知识图谱`.
- Never modify the sibling LightRAG or GraphRAG-Benchmark repositories.
- Run Python with `D:\Anaconda\Scripts\conda.exe run -n lightrag`.
- Use only the editable local LightRAG checkout at commit `c1248646e4eda4d89054926af2e094730daf23fe`; do not install LightRAG from PyPI.
- Default corpus: `D:\HuaweiMoveData\Users\huawei\Desktop\kg\GraphRAG-Benchmark\Datasets\Corpus\novel.json`.
- Maintain one active document and one `data/workspace`.
- Fix graph storage to `NetworkXStorage`.
- Public modes are `local`, `global`, `hybrid`, and `mix` only.
- Query through `aquery_llm()` with `top_k=chunk_top_k` and `stream=False`.
- Handle both raised exceptions and `status != "success"`.
- Never print, save, log, or commit the DeepSeek key.
- Do not create FastAPI or React code in this phase.

---

### Task 1: Package, Configuration, and Domain Types

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/app/__init__.py`
- Create: `backend/app/config.py`
- Create: `backend/app/domain/__init__.py`
- Create: `backend/app/domain/models.py`
- Create: `backend/app/domain/errors.py`
- Create: `backend/tests/test_config.py`
- Create: `.env.example`

**Interfaces:**
- Produces `Settings`, `CorpusDocument`, `QueryMode`, `QueryResult`, `QueryError`, and `AppError`.

- [ ] **Step 1: Write the failing settings test**

    def test_settings_fix_workspace_and_graph_storage(tmp_path: Path) -> None:
        settings = Settings(app_root=tmp_path, deepseek_api_key="test-secret")
        assert settings.active_dir == tmp_path / "data" / "active"
        assert settings.workspace_dir == tmp_path / "data" / "workspace"
        assert settings.graph_storage == "NetworkXStorage"
        assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
        assert settings.embedding_dim == 384

- [ ] **Step 2: Run it and confirm the missing module failure**

    & 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python -m pytest backend/tests/test_config.py -v

- [ ] **Step 3: Add and install package metadata**

`backend/pyproject.toml` declares Python `>=3.10,<3.11`, dependencies `datasets`, `pydantic`, `pydantic-settings`, and `transformers`, plus test dependencies `pytest` and `pytest-asyncio`. It deliberately does not declare `lightrag-hku`: LightRAG is the separately pinned local runtime dependency. Configure pytest `pythonpath=["."]` and `asyncio_mode="auto"`.

First bind the Conda environment to the local checkout, then install the application package:

    & 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python -m pip install -e 'D:\HuaweiMoveData\Users\huawei\Desktop\kg\LightRAG'

    & 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python -m pip install -e '.\backend[test]'

Verify both the import path and source commit:

    & 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python -c "import lightrag; print(lightrag.__file__); print(lightrag.__version__)"
    git -C 'D:\HuaweiMoveData\Users\huawei\Desktop\kg\LightRAG' rev-parse HEAD

Expected: the import path is inside the local checkout, version is `1.5.7`, and Git prints `c1248646e4eda4d89054926af2e094730daf23fe`.

- [ ] **Step 4: Implement exact settings and types**

`Settings` fields: `app_root`; sibling benchmark path; secret loaded with `AliasChoices("DEEPSEEK_API_KEY", "LLM_API_KEY")`; DeepSeek base URL/model; BGE model/dimension/max tokens; chunk size 500; overlap 50; graph storage. Derived paths are `app_root/data/active` and `app_root/data/workspace`.

`QueryMode` is a four-value string enum. `CorpusDocument` holds corpus name, context, subset, character count, and word count. `QueryResult` holds answer, latency, display context, four structured lists, warnings, and optional error. `AppError` stores only a stable code and safe message.

`.env.example` contains an empty key plus non-secret model defaults.

- [ ] **Step 5: Verify and commit**

    & 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python -m pytest backend/tests/test_config.py -v
    git add backend .env.example
    git commit -m 'chore: bootstrap phase1 Python package'

---

### Task 2: GraphRAG-Bench Loader and Active Document

**Files:**
- Create: `backend/app/adapters/__init__.py`
- Create: `backend/app/adapters/graphrag_bench.py`
- Create: `backend/app/services/__init__.py`
- Create: `backend/app/services/dataset_service.py`
- Create: `backend/tests/adapters/test_graphrag_bench.py`
- Create: `backend/tests/services/test_dataset_service.py`

**Interfaces:**
- `GraphRAGBenchLoader.load_documents(path: Path, subset: str) -> list[CorpusDocument]`
- `DatasetService.select_shortest(path: Path, subset: str) -> CorpusDocument`
- `DatasetService.load_active() -> CorpusDocument | None`
- `corpus_fingerprint(document: CorpusDocument) -> str`

- [ ] **Step 1: Write failing loader and selection tests**

    def test_selects_shortest_document(tmp_path: Path) -> None:
        path = tmp_path / "novel.json"
        path.write_text(json.dumps([
            {"corpus_name": "long", "context": "one two three"},
            {"corpus_name": "short", "context": "one two"},
        ]), encoding="utf-8")
        service = DatasetService(GraphRAGBenchLoader(), tmp_path / "active")
        selected = service.select_shortest(path, "novel")
        assert selected.corpus_name == "short"
        assert service.load_active() == selected

    def test_rejects_empty_context(tmp_path: Path) -> None:
        path = tmp_path / "novel.json"
        path.write_text(json.dumps([
            {"corpus_name": "empty", "context": ""}
        ]), encoding="utf-8")
        with pytest.raises(AppError) as error:
            GraphRAGBenchLoader().load_documents(path, "novel")
        assert error.value.code == "INVALID_CORPUS"

- [ ] **Step 2: Run tests and confirm failure**

    & 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python -m pytest backend/tests/adapters/test_graphrag_bench.py backend/tests/services/test_dataset_service.py -v

- [ ] **Step 3: Implement loading**

JSON must be an array of rows with non-empty string `corpus_name` and `context`. Parquet uses `load_dataset("parquet", data_files=str(path), split="train")` and the same validator. Counts are `len(context)` and `len(context.split())`. Unsupported extensions raise `UNSUPPORTED_CORPUS`.

- [ ] **Step 4: Implement deterministic selection and persistence**

    selected = min(
        documents,
        key=lambda item: (item.character_count, item.corpus_name),
    )

The SHA-256 fingerprint covers subset, a null byte, corpus name, a null byte, and context. Atomically replace `data/active/corpus.json` and `manifest.json` through sibling temporary files. Manifest fields: schema version, source path, counts, and fingerprint.

- [ ] **Step 5: Verify and commit**

    & 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python -m pytest backend/tests/adapters/test_graphrag_bench.py backend/tests/services/test_dataset_service.py -v
    git add backend/app/adapters backend/app/services backend/tests
    git commit -m 'feat: select one GraphRAG-Bench document'

---

### Task 3: LightRAG Factory and Workspace Reuse

**Files:**
- Create: `backend/app/services/lightrag_factory.py`
- Create: `backend/app/services/workspace_service.py`
- Create: `backend/tests/services/test_workspace_service.py`

**Interfaces:**
- `LightRAGFactory.create() -> LightRAG` with initialized storages.
- `LightRAGFactory.runtime_fingerprint() -> str`.
- `WorkspaceService.ensure_ready(document: CorpusDocument) -> IndexOutcome`.
- `IndexOutcome.action` is `built` or `reused`.

- [ ] **Step 1: Write failing fake-based tests**

    class FakeRAG:
        def __init__(self, factory) -> None:
            self.factory = factory

        async def ainsert(self, text: str) -> None:
            self.factory.inserted.append(text)

        async def finalize_storages(self) -> None:
            self.factory.finalized += 1
            self.factory.workspace_dir.mkdir(parents=True, exist_ok=True)
            (self.factory.workspace_dir / "graph_chunk_entity_relation.graphml").write_text(
                "<graphml />",
                encoding="utf-8",
            )

    class FakeFactory:
        def __init__(self, workspace_dir: Path) -> None:
            self.workspace_dir = workspace_dir
            self.create_count = 0
            self.inserted: list[str] = []
            self.finalized = 0

        async def create(self) -> FakeRAG:
            self.create_count += 1
            return FakeRAG(self)

    def make_document(text: str) -> CorpusDocument:
        return CorpusDocument(
            corpus_name="Novel-test",
            context=text,
            subset="novel",
            character_count=len(text),
            word_count=len(text.split()),
        )

    @pytest.mark.asyncio
    async def test_second_call_reuses_workspace(tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        factory = FakeFactory(workspace)
        service = WorkspaceService(workspace, factory, "runtime-1")
        first = await service.ensure_ready(make_document("one two"))
        second = await service.ensure_ready(make_document("one two"))
        assert first.action == "built"
        assert second.action == "reused"
        assert factory.create_count == 1
        assert factory.inserted == ["one two"]

    @pytest.mark.asyncio
    async def test_changed_document_is_rejected(tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        service = WorkspaceService(
            workspace,
            FakeFactory(workspace),
            "runtime-1",
        )
        await service.ensure_ready(make_document("first text"))
        with pytest.raises(AppError) as error:
            await service.ensure_ready(make_document("changed text"))
        assert error.value.code == "INDEX_CONFIG_MISMATCH"

- [ ] **Step 2: Confirm tests fail**

    & 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python -m pytest backend/tests/services/test_workspace_service.py -v

- [ ] **Step 3: Implement LightRAGFactory**

Before implementation, probe the pinned runtime contract:

    & 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python -c "import inspect; from lightrag import LightRAG; source=inspect.getsource(LightRAG.initialize_storages); print('AUTO_PIPELINE_STATUS=' + str('initialize_pipeline_status' in source))"

Expected at pinned commit: `AUTO_PIPELINE_STATUS=True`. The pinned implementation therefore calls only `await rag.initialize_storages()`; it must not call `initialize_pipeline_status()` a second time. If this probe is false, stop execution because the imported runtime is not the reviewed commit.

Build `EmbeddingFunc` from `hf_embed.func`, `AutoTokenizer`, and `AutoModel`. Build DeepSeek completion with:

    llm_func = partial(
        openai_complete_if_cache,
        settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.deepseek_api_key.get_secret_value(),
    )

Construct `LightRAG` with workspace, model, embedding, chunk settings, and `graph_storage="NetworkXStorage"`. Reject a missing key before construction and initialize storages before return. Cache the embedding model in the factory instance.

- [ ] **Step 4: Implement manifest lifecycle**

`ensure_ready()` uses an `asyncio.Lock`. Matching ready fingerprints return `reused` without creating LightRAG. A differing ready manifest raises `INDEX_CONFIG_MISMATCH`.

A new build follows this exact order:

    write status="building"
    create LightRAG and initialize storages
    await rag.ainsert(document.context)
    await rag.finalize_storages()
    verify graph_chunk_entity_relation.graphml exists and has non-zero length
    write status="ready"

If creation, insertion, finalization, or integrity checking fails, write `status="failed"` with only the exception class and re-raise. Never write `ready` before finalization. Add a third fake test where `finalize_storages()` raises; assert that the manifest ends in `failed`, not `ready`.

- [ ] **Step 5: Verify and commit**

    & 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python -m pytest backend/tests/services/test_workspace_service.py -v
    git add backend/app/services backend/tests/services/test_workspace_service.py
    git commit -m 'feat: build and reuse one LightRAG workspace'

---

### Task 4: aquery_llm Adapter

**Files:**
- Create: `backend/app/adapters/lightrag.py`
- Create: `backend/tests/adapters/test_lightrag_adapter.py`

**Interfaces:**
- `LightRAGAdapter.query(rag, query: str, mode: QueryMode, top_k: int) -> QueryResult`
- `serialize_context(data: dict) -> str | None`
- `map_lightrag_exception(exc: Exception) -> AppError`

- [ ] **Step 1: Write failing contract tests**

    class SuccessfulFakeRAG:
        def __init__(self) -> None:
            self.param = None

        async def aquery_llm(self, query, param):
            self.param = param
            return {
                "status": "success",
                "data": {
                    "entities": [{"entity_name": "ALICE"}],
                    "relationships": [],
                    "chunks": [{"content": "Alice appears."}],
                    "references": [],
                },
                "llm_response": {"content": "Alice is the protagonist."},
            }

    class RaisingFakeRAG:
        async def aquery_llm(self, query, param):
            raise TimeoutError("provider timed out")

    class FailureFakeRAG:
        async def aquery_llm(self, query, param):
            return {
                "status": "failure",
                "message": "no results",
                "data": {},
                "llm_response": {"content": None},
            }

    @pytest.mark.asyncio
    async def test_success_unifies_top_k() -> None:
        rag = SuccessfulFakeRAG()
        result = await LightRAGAdapter().query(
            rag, "Who is Alice?", QueryMode.MIX, 5
        )
        assert result.answer == "Alice is the protagonist."
        assert result.entities == [{"entity_name": "ALICE"}]
        assert rag.param.mode == "mix"
        assert rag.param.top_k == 5
        assert rag.param.chunk_top_k == 5
        assert rag.param.stream is False

    @pytest.mark.asyncio
    async def test_raised_timeout_is_mapped() -> None:
        with pytest.raises(AppError) as error:
            await LightRAGAdapter().query(
                RaisingFakeRAG(), "Who is Alice?", QueryMode.MIX, 5
            )
        assert error.value.code == "LLM_TIMEOUT"

    @pytest.mark.asyncio
    async def test_failure_dictionary_is_rejected() -> None:
        with pytest.raises(LightRAGQueryError, match="no results"):
            await LightRAGAdapter().query(
                FailureFakeRAG(), "Who is Alice?", QueryMode.MIX, 5
            )

- [ ] **Step 2: Confirm tests fail**

    & 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python -m pytest backend/tests/adapters/test_lightrag_adapter.py -v

- [ ] **Step 3: Implement the exact call**

    try:
        result = await rag.aquery_llm(
            query,
            param=QueryParam(
                mode=mode.value,
                top_k=top_k,
                chunk_top_k=top_k,
                stream=False,
            ),
        )
    except Exception as exc:
        raise map_lightrag_exception(exc) from exc

    if result.get("status") != "success":
        raise LightRAGQueryError(
            result.get("message", "LightRAG query failed")
        )

Validate non-empty query and `1 <= top_k <= 50`. Map timeout, authentication, rate limit, and generic errors to safe codes. Read only `llm_response.content` and `data.entities/relationships/chunks/references`. Missing lists become empty plus warnings. Serialize non-empty sections into `context_text` and label it as display data, not the exact internal prompt. Measure latency with `time.perf_counter()`.

- [ ] **Step 4: Run all offline tests**

    & 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python -m pytest backend/tests -v

Expected: all tests pass.

- [ ] **Step 5: Commit**

    git add backend/app/adapters/lightrag.py backend/tests/adapters/test_lightrag_adapter.py
    git commit -m 'feat: add structured aquery_llm adapter'

---

### Task 5: Smoke CLI and Real Gate

**Files:**
- Create: `scripts/smoke_lightrag.py`
- Create: `backend/tests/test_smoke_output.py`
- Create: `README.md`
- Modify: `.gitignore`

**Interfaces:**
- CLI: `--corpus`, `--subset`, `--query`, `--mode`, `--top-k`.
- Output: JSON with status, corpus, index action, answer, latency, four lists, counts, and warnings.

- [ ] **Step 1: Write the failing summary test**

    def test_summary_contains_action_and_counts() -> None:
        result = QueryResult(
            query="Who is Alice?",
            mode=QueryMode.MIX,
            answer="Alice is the protagonist.",
            latency_ms=12.5,
            context_text="context",
            entities=[{"entity_name": "ALICE"}],
            relationships=[],
            chunks=[{"content": "Alice appears."}],
            references=[],
        )
        summary = build_summary("Novel-test", "built", result)
        assert summary["index_action"] == "built"
        assert summary["counts"] == {
            "entities": 1,
            "relationships": 0,
            "chunks": 1,
            "references": 0,
        }

- [ ] **Step 2: Implement CLI orchestration**

Select the shortest document, ensure workspace, create a query RAG instance, call the adapter, finalize in `finally`, and print one success JSON object. Catch `AppError` and print only safe code/message JSON to stderr with exit 1. Unexpected errors print only the exception class.

- [ ] **Step 3: Pass tests and verify local LightRAG import**

    & 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python -m pytest backend/tests -v
    & 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python -c "import lightrag; print(lightrag.__file__); print(lightrag.__version__)"

Expected: tests pass; import path points into the local LightRAG checkout; version is 1.5.7.

- [ ] **Step 4: Commit before paid execution**

    git add scripts backend/tests/test_smoke_output.py
    git commit -m 'feat: add LightRAG phase1 smoke command'

- [ ] **Step 5: Run first real Mix query**

With `DEEPSEEK_API_KEY` already set in the current PowerShell process, run this as one command:

    & 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python scripts\smoke_lightrag.py --corpus 'D:\HuaweiMoveData\Users\huawei\Desktop\kg\GraphRAG-Benchmark\Datasets\Corpus\novel.json' --subset novel --mode mix --top-k 5 --query 'Who is the central character, and which relationships are most important to the story?'

Required evidence: exit 0; success status; `index_action="built"`; non-empty answer; four counts; non-empty `data/workspace/graph_chunk_entity_relation.graphml`.

- [ ] **Step 6: Repeat the identical command**

Required evidence: exit 0; success; `index_action="reused"`; non-empty answer; no new indexing phase.

- [ ] **Step 7: Add runbook and ignores**

README documents installation, environment variables, the real command, first-run cost/model download, workspace reuse, display-context semantics, and safe error codes. Ignore:

    .env
    data/active/
    data/workspace/
    data/workspace_backups/
    artifacts/
    backend/*.egg-info/
    backend/.pytest_cache/
    __pycache__/

- [ ] **Step 8: Final verification and documentation commit**

    & 'D:\Anaconda\Scripts\conda.exe' run -n lightrag python -m pytest backend/tests -v
    git status --short
    git add README.md .gitignore
    git commit -m 'docs: add phase1 LightRAG runbook'

Phase 1 is complete only after both first-run and reuse-run evidence are observed. Stop here; FastAPI and React require a separate plan.
