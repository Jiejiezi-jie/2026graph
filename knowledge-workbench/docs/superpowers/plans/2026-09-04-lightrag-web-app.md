# LightRAG Web Application Implementation Plan

> 2026-09-05 执行补充：用户已确认改为可插拔 retrieval。
> 当前网页采用 `aquery_data() → RetrievalResult → 统一生成`；
> 下文原 `aquery_llm()` 约束只保留给 CLI，不再约束 Web。
> 已落实：状态/选数/显式建索引 API、查询与本地记录、GraphML 只读局部图、
> React/Cytoscape 网页、插件注册入口。没有实现额外检索算法。
> 以 `docs/RETRIEVAL_PLUGINS.md` 和根 README 为当前实现约定。
> 验证采用 pytest、TypeScript/build 和 Playwright 离线浏览器检查；
> 当前 Key 401，未将模拟界面的成功当作真实模型端到端成功。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the validated LightRAG pipeline through FastAPI and build a single-page React/Cytoscape retrieval workbench.

**Architecture:** Reuse the existing dataset, workspace, factory, and `aquery_llm()` adapter modules. Add thin application services and FastAPI routes, then consume one stable JSON contract from a Vite React frontend; GraphML remains the only graph source and direct retrieval hits are visually distinct from one-hop context.

**Tech Stack:** Python 3.10, FastAPI, Pydantic 2, SQLite standard library, pytest/httpx; React, TypeScript, Vite, Cytoscape, Vitest, Testing Library.

**Spec:** `docs/superpowers/specs/2026-09-03-lightrag-graphrag-bench-platform-design.md`

## Global Constraints

- Work only in the existing `codex/phase1-backend` linked worktree.
- Keep the sibling LightRAG and GraphRAG-Benchmark checkouts as runtime/data dependencies; do not copy their source into the app.
- Keep one active dataset and one `data/workspace`; never rebuild during a query.
- Public query modes remain `local`, `global`, `hybrid`, and `mix`.
- Use `aquery_llm()` with `top_k=chunk_top_k`, `stream=False`.
- V1 graph storage remains `NetworkXStorage` and the graph source is `graph_chunk_entity_relation.graphml`.
- Never accept, return, log, save, or commit an API key.
- Do not add benchmark evaluation, reranking, custom retrieval, authentication, or multi-user state.

---

### Task 1: FastAPI Application and Dataset/Index Status

**Files:**
- Modify: `backend/pyproject.toml`
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/schemas.py`
- Create: `backend/app/api/dependencies.py`
- Create: `backend/app/main.py`
- Modify: `backend/app/services/dataset_service.py`
- Modify: `backend/app/services/workspace_service.py`
- Create: `backend/tests/api/test_status_routes.py`

**Interfaces:**
- Produces `create_app(context: AppContext | None = None) -> FastAPI`.
- Produces `GET /api/health`, `GET /api/dataset/status`, and `GET /api/index/status`.
- Produces `DatasetService.get_status()` and `WorkspaceService.get_status(document)` without creating LightRAG.

- [ ] **Step 1: Write failing route tests** using a temporary real `DatasetService` and workspace manifest. Assert health never contains a secret, missing dataset returns `selected=false`, and a matching ready manifest returns `status="ready"` plus `reusable=true`.
- [ ] **Step 2: Run** `python -m pytest backend/tests/api/test_status_routes.py -v`; expect import failure for `app.main`.
- [ ] **Step 3: Add FastAPI/httpx dependencies** and implement response schemas, application context construction, CORS for local Vite origins, and the three read-only routes. `get_status()` must not call the LightRAG factory.
- [ ] **Step 4: Run the route tests and all backend tests**; expect all pass.
- [ ] **Step 5: Commit** with `feat: expose health and workspace status APIs`.

---

### Task 2: Dataset Selection, Query Service, and Run Records

**Files:**
- Modify: `backend/app/services/dataset_service.py`
- Create: `backend/app/persistence/__init__.py`
- Create: `backend/app/persistence/query_runs.py`
- Create: `backend/app/services/query_service.py`
- Modify: `backend/app/api/schemas.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/services/test_query_service.py`
- Create: `backend/tests/api/test_query_routes.py`

**Interfaces:**
- Produces `POST /api/dataset/load-options`, `POST /api/dataset/select`, `POST /api/index/build`.
- Produces `POST /api/query` and `GET /api/query/{run_id}`.
- `QueryService.query(text: str, mode: QueryMode, top_k: int) -> QueryResult` checks active data and a reusable ready index before creating a query RAG.
- `QueryRunRepository` stores run metadata in `app.db`; answer/context storage remains disabled.

- [ ] **Step 1: Write failing service tests** proving a query is rejected when no dataset/index is ready, a ready query finalizes storage, and success/failure metadata is recorded without answer/context.
- [ ] **Step 2: Run the service tests** and confirm missing modules/functions fail.
- [ ] **Step 3: Implement selection by exact `corpus_name`, explicit index build, query orchestration, and SQLite metadata records.** Convert `AppError` into `{error:{code,message}}` with appropriate 4xx/5xx status; never include traceback.
- [ ] **Step 4: Write route tests** for valid mode/top-k, invalid mode, empty query, index-not-ready, successful response, and run lookup.
- [ ] **Step 5: Run targeted and full backend tests**, then commit with `feat: add LightRAG query API`.

---

### Task 3: Query-Related GraphML Service

**Files:**
- Modify: `backend/app/domain/models.py`
- Create: `backend/app/services/graph_service.py`
- Modify: `backend/app/services/query_service.py`
- Create: `backend/tests/services/test_graph_service.py`

**Interfaces:**
- Produces `GraphNode` and `GraphEdge` response models with `retrieved: bool`.
- Produces `GraphService.build_subgraph(result: QueryResult, include_neighbors: bool = True, node_limit: int = 80) -> tuple[list[GraphNode], list[GraphEdge], list[str]]`.
- Exact normalized names are the only GraphML matching rule; direct returned entities/relationships are highlighted, one-hop additions are not.

- [ ] **Step 1: Write a tiny GraphML fixture and failing behavior tests** for exact entity matching, one-hop expansion, direct-hit highlighting, node limit, and empty-result warning.
- [ ] **Step 2: Run the graph tests** and confirm the missing service failure.
- [ ] **Step 3: Implement GraphML loading with NetworkX**, defensive attribute conversion, deterministic node/edge ordering, and no fuzzy matching.
- [ ] **Step 4: Attach graph nodes/edges to query responses**, run targeted/full backend tests, and commit with `feat: return highlighted query subgraphs`.

---

### Task 4: React Retrieval Workbench

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/styles.css`
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/api/types.ts`
- Create: `frontend/src/components/StatusBar.tsx`
- Create: `frontend/src/components/QueryComposer.tsx`
- Create: `frontend/src/components/ResultPanel.tsx`
- Create: `frontend/src/App.test.tsx`

**Interfaces:**
- Consumes the FastAPI contracts from Tasks 1–3.
- Produces one responsive page with dataset/index status, query/mode/top-k controls, answer, latency, warnings, and entity/relation/chunk/reference tabs.

- [ ] **Step 1: Read the applicable UI skills**, then install the minimal React/Vite/testing dependencies and lock them in `package-lock.json`.
- [ ] **Step 2: Write failing UI tests** proving query is disabled until index is ready, selected mode/top-k reach the request, successful data renders, and structured API errors show an actionable message.
- [ ] **Step 3: Run `npm test -- --run`** and confirm missing UI behavior fails.
- [ ] **Step 4: Implement the accessible single-page workbench** with a calm medical/research visual language, explicit loading/empty/error states, and no API-key field.
- [ ] **Step 5: Run tests and `npm run build`**, then commit with `feat: build retrieval workbench UI`.

---

### Task 5: Cytoscape Graph and End-to-End Runbook

**Files:**
- Create: `frontend/src/components/QueryGraph.tsx`
- Create: `frontend/src/components/QueryGraph.test.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Modify: `README.md`
- Modify: `.gitignore`

**Interfaces:**
- Renders backend graph nodes/edges without querying the full graph separately.
- Supports fit/zoom/pan, exact node search, type filtering, “hits only”, and one-hop visibility.

- [ ] **Step 1: Write failing component tests** for empty state, retrieved classes, hits-only filtering, and node detail selection.
- [ ] **Step 2: Implement a lifecycle-safe Cytoscape component** that destroys its instance on cleanup; retrieved nodes/edges use a high-contrast class and neighbor context is muted.
- [ ] **Step 3: Run frontend tests and production build.**
- [ ] **Step 4: Start FastAPI and Vite, run browser smoke checks** for status, one real query, result tabs, graph highlight, and error rendering. Do not rebuild the index.
- [ ] **Step 5: Update README** with exact Windows commands, environment variables, ports, workspace reuse, and demo questions.
- [ ] **Step 6: Run all backend/frontend verification and commit** with `feat: complete LightRAG web demo`.
