# 知图工作台交接

本分支在原有项目旁新增 `knowledge-workbench/`，保留根目录原有 GraphRAG 项目。

- `knowledge-workbench/frontend/`：React / Cytoscape 网页。
- `knowledge-workbench/backend/`：FastAPI 检索与答案生成服务。
- `knowledge-workbench/exports/Novel-4128-chunks/`：51 个小说分块、原始分块 JSON 和完整 GraphML（729 节点、1,369 条关系）。

数据读取见 [分块与图谱说明](knowledge-workbench/exports/Novel-4128-chunks/README.md)。
工作台启动与运行环境见 [工作台 README](knowledge-workbench/README.md)。启动命令应在 `knowledge-workbench/` 中运行，并将示例中的本机路径替换为你的路径。

完整查询索引已补齐：[恢复说明](knowledge-workbench/exports/Novel-4128-runtime/README.md)。其中包括原文、51 条文本向量、729 条实体向量、1,369 条关系向量和全部配套存储。队友配置匹配版本与自己的 Key 后，可恢复索引直接查询，无需重新建图。API Key、模型权重、查询历史和依赖目录不随分支上传。

代码来源：本地工作台提交 `51229e0`。
