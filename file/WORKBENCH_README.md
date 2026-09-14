# 知图 · Medical 检索工作台

FastAPI + React/Cytoscape 知识检索工作台,面向本项目 **GraphRAG-Bench Medical** 评测:
在 LightRAG / PathRAG 共享的 Medical 知识图谱上做检索问答演示,并展示 vector / LightRAG / PathRAG / **Adaptive(router 自动路由)** 四种检索方法的差异。

> 详细运行手册见 [MEDICAL_WORKBENCH.md](MEDICAL_WORKBENCH.md)——导入索引、preflight、四方法验证、常见问题都以它为准。

## 启动网页

前置产物(不在 Git 内,首次需在任意克隆上生成):

```bash
# 从仓库自身导入 Medical 索引 + router + vendor 化 official_backends 到运行时 bundle
python src/backend/scripts/prepare_medical.py --repo <本仓库路径> --ref <本仓库 medical 集成 commit> \
  --output src/backend/data/medical
```

终端 1(后端):

```bash
export DEEPSEEK_API_KEY='<your-key>'
python src/backend/scripts/serve_web.py --port 8000 \
  --medical-bundle src/backend/data/medical \
  --medical-embedding-path <BGE-M3 权重目录> \
  --medical-pathrag-root deps/PathRAG     # 仓库自带精简上游;也可指向独立克隆的固定提交检出
```

终端 2(前端):

```bash
cd src/frontend
npm ci
npm run dev
```

打开 [检索工作台](http://127.0.0.1:5173/);[API 文档](http://127.0.0.1:8000/docs)。
预检:`serve_web.py --check`;完整启动与验证流程见 MEDICAL_WORKBENCH.md。

## 功能

- **数据**:GraphRAG-Bench · Medical(官方 vendored 语料位于仓库 `data/vendor/GraphRAG-Benchmark`);本模式使用**只读导入索引**(LightRAG 构建的 Medical 图谱 3,396 节点 / 5,693 关系 / 199 chunk,PathRAG 与之共享同一张图),不现场建图、不产生额外抽取费用。
- **检索方法**:vector(纯向量)、lightrag、pathrag、adaptive(路由到三者之一)。检索与答案生成分离,统一走 DeepSeek 生成。
- **图谱**:Cytoscape 渲染,可缩放、定位实体、查看节点/关系描述;命中高亮与补充邻居严格区分;支持展开全屏。
- **查询流**:分阶段 NDJSON 流式(`routing → retrieval → result`),前端的 Adaptive 卡片可视化 router 决策过程。
- **历史记录**:本地保存查询、答案、检索结果、耗时与安全错误码。
- **API 设置**:侧栏填写 DeepSeek Key/Base URL,仅存后端内存,不落盘。

## 代码结构

- `src/backend/app/`:FastAPI 应用。`main.py` 入口;`medical/`(engine / retrievers / preflight / profile / pathrag_protocol / shared_index)是 Medical 模式的全部接线;`retrieval/` 检索插件契约;`api/`、`services/`、`domain/` 通用服务。
- `src/frontend/`:React 19 + Vite + Cytoscape,`src/AdaptiveRouting.tsx` 为路由可视化卡片。
- `src/backend/scripts/serve_web.py`:启动器(支持 `--check` / `--preview-only`);`src/backend/scripts/prepare_medical.py`:从仓库生成只读导入 bundle(sha256 校验)。
- `test/workbench/`:pytest;`src/frontend/qa/`:Playwright 离线冒烟(拦截查询请求,不调付费 API)。

## 验证

```bash
python -m pytest test/workbench -q
cd src/frontend && npm run build
```

离线浏览器检查在前后端启动后运行 `src/frontend/qa/*.js`(脚本拦截查询,不产生 API 调用)。

## Runtime 契约

- Medical profile 依赖见 `requirements-medical.txt`(LightRAG v1.5.7 commit `28ff1b0`、scikit-learn 1.7.2、BGE-M3)。
- Embedding:BGE-M3(1024 维),本地权重目录由 `--medical-embedding-path` 指定;PathRAG upstream 由 `--medical-pathrag-root` 指定。
- LLM:DeepSeek OpenAI 兼容 API;查询仅 Key 决定,不在源码落盘。
- 仓库仍保留一套 legacy 本地建图运行时代码(`services/lightrag_factory` 等,面向 `data/vendor` 语料的通用建图 profile),非默认路径;Medical 展示请始终走 `--medical-bundle` 导入模式。
