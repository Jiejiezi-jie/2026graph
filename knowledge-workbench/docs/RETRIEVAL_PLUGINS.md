# 检索插件开发约定

当前在自己的应用仓库开发，不修改外部 LightRAG / GraphRAG-Benchmark / 2026graph。

## Web 查询链路

```text
React → POST /api/query → RetrievalRegistry
                            ↓ retrieve(request)
                        RetrievalResult
                            ↓
                  统一 DeepSeek 答案生成
                            ↓
                 答案 / 证据 / 局部图 / 耗时
```

后端接口：`backend/app/retrieval/contracts.py`。
注册入口：`backend/app/retrieval/plugins.py`。
Web 不再使用“先生成答案再提取 context”的方式。LightRAG 插件调用
`aquery_data()`，统一答案服务使用检索结果生成回答。
原有 `LightRAGAdapter.aquery_llm()` CLI 链路保留，评测脚本不变。
**Web 的生成提示词已经不同，因此不能视作旧评测的同一个实验配置。**

## 添加方法

例如（开发示例，默认不注册、不作为真实结果展示）：

```python
from app.retrieval.contracts import MethodDescriptor, RetrievalResult

class MyRetriever:
    descriptor = MethodDescriptor(
        id="my-retriever", name="我的检索方法", description="读取自己的索引",
    )

    async def retrieve(self, request):
        hits = await my_index.search(request.query, top_k=request.top_k)
        chunks = [{"chunk_id": h.id, "content": h.text} for h in hits]
        return RetrievalResult(
            chunks=chunks,
            context_text="\n\n".join(h.text for h in hits),
        )

# 在 plugins.py 的 build_registry 中添加：
registry.register(MyRetriever())
```

方法不必依赖 LightRAG，也不必提供图谱。自定义方法可自行连接向量库或读取
三元组；实体、关系、chunks、references 无则为空，不虚构 score/rank。
图谱可用 `GraphData / GraphNode / GraphEdge` 返回，ID 必须唯一，边端点必须存在。
前端图形 ID 会加前缀，避免实体 ID 与边 ID 冲突。

`MethodDescriptor.options` 提供简单的下拉参数（key、label、choices、default）。
前端动态展示这些参数，不按算法写分支。更复杂的参数表单以后再扩展协议。
插件应校验自己的 options；返回的文本必须是实际证据，不是另一个生成答案。

## 内置 LightRAG 方法

- local / global / hybrid / mix / naive；网页将 naive 显示为 `vector（纯向量）`。`top_k = chunk_top_k`，范围 1–50；naive 返回文本检索证据，不扩展图谱邻居。
- 明确关闭未配置的 rerank。
- 同时捕获 Python exception 和 `status != success`。
- 查询不触发 `ainsert()`；workspace 必须就绪且 fingerprint 匹配。
- `context_text` 是固定模板序列化，并非 LightRAG 内部最终 prompt。
- GraphML 仅是这个插件的图谱来源，不是所有插件都必须使用的存储。
- GraphML 精确 ID 匹配，不模糊猜名字；保留原图有向/无向属性。
- 命中实体/关系突出显示；一跳补充邻居淡化显示。不是 LLM 推理轨迹。
- 图谱展示最多 80 个节点、200 条边。初始预览最多 60 个高连接度节点。

## 错误、记录及运行限制

- 检索失败：返回 failure 和安全错误码。
- 生成失败：仍返回已有 retrieval 数据，便于定位问题。
- query API 的业务失败以 HTTP 200 + status=failure 返回；验证错误 422，
  忙碌/索引未就绪为 409；前端同时处理 HTTP 与业务状态。
- SQLite 保存最近 100 次查询全文/证据及耗时，列表显示最近 20 条；
  位于忽略追踪的 `data/runs.sqlite3`。不保存 Key。
- 本地单用户演示，仅绑定 127.0.0.1；无登录鉴权，不应直接公开到互联网。
- **只运行一个后端进程/worker**。同一 workspace 不得同时运行 CLI 建图、
  CLI 评测或其他服务。进程内锁不是跨进程文件锁。
- 关闭窗口/浏览器不是取消后端请求；索引由后端后台任务运行。
  后端正常关闭会等待建图；强行退出后显示 interrupted，需要确认备份再重建。
- 重建只移动 `data/workspace` 到 `data/workspace_backups/<timestamp-id>`；
  旧图谱和向量文件保留，不删除。网页不提供自动删除备份。
- 新一轮建图在互斥锁下清理 pinned LightRAG 的进程级共享 namespace 缓存，
  避免备份文件后仍读取旧 corpus；不清理磁盘上的模型下载缓存。
- 标记 ready 前检查持久化 doc_status=processed、非空 chunks 和三个 VDB 文件，
  防止 `ainsert()` 内部失败但未抛异常时误报成功。
- 生成证据超过 100,000 字符时按前缀截断并提示，完整检索数据仍保存。
