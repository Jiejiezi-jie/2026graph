# 知图工作台交接

本分支在原有项目旁新增 `knowledge-workbench/`，保留根目录原有 GraphRAG 项目。

- `knowledge-workbench/frontend/`：React / Cytoscape 网页。
- `knowledge-workbench/backend/`：FastAPI 检索与答案生成服务。
- `knowledge-workbench/exports/Novel-4128-chunks/`：51 个小说分块、原始分块 JSON 和完整 GraphML（729 节点、1,369 条关系）。

数据读取见 [分块与图谱说明](knowledge-workbench/exports/Novel-4128-chunks/README.md)。
工作台启动与运行环境见 [工作台 README](knowledge-workbench/README.md)。启动命令应在 `knowledge-workbench/` 中运行，并将示例中的本机路径替换为你的路径。

交接数据可以直接读取；完整问答仍需配置匹配的 LightRAG、模型、数据路径以及完整索引。API Key、运行时向量库、查询历史和依赖目录未随此分支上传。

代码来源：本地工作台提交 `51229e0`。
