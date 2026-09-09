# Novel-4128 分块与图谱交接

来源：当前知识检索工作台已建好的 GraphRAG-Bench / novel / Novel-4128 索引。
原文规模：92,796 字符，18,244 词。共 51 个 chunks，按 chunk_order_index 从 0 开始排序。

文件：
- chunks.jsonl：推荐用于队友接入，每行一个 chunk，包含 chunk_id、corpus_name、chunk_order_index、tokens、content、full_doc_id、file_path。UTF-8，无 BOM。
- kv_store_text_chunks.json：LightRAG 原始分块存储，键为 chunk ID，原样复制供核对。
- graph_chunk_entity_relation.graphml：同一篇文档已建好的完整知识图谱，包含 729 个节点、1,369 条关系；不是网页只加载的 60 节点预览。

分块相关配置的代码默认值为 500 tokens、重叠 50 tokens。当前分块记录含实际 tokens；旧索引 manifest 仅存配置指纹，未保存可直接读取的重叠参数，因此本包保留现成分块，不重新切分。
file_path 原值为 unknown_source；不要将其当作原文页码或实际文件路径。corpus_name 明确标识本文档。
原始存储中的 llm_cache_list 是缓存引用，不是分块正文；简化 JSONL 未带出这些内部字段。

Python 读取：
```python
import json
with open('chunks.jsonl', encoding='utf-8') as f:
    chunks = [json.loads(line) for line in f if line.strip()]
assert len(chunks) == 51
texts = [chunk['content'] for chunk in chunks]
```

读取图谱（安装 networkx）：
```python
import networkx as nx
graph = nx.read_graphml('graph_chunk_entity_relation.graphml')
print(graph.number_of_nodes(), graph.number_of_edges())  # 729 1369
```

这是单篇文档的分块与图谱包，不是完整小说数据集，也不包含向量库。代码在仓库的 backend/、frontend/ 和 scripts/ 中。
队友可以直接对这些 chunks 做自己的检索实验，或读取 GraphML 分析节点关系。
仅把这几个文件复制到工作目录，不能恢复完整 LightRAG 问答：仍需要相匹配的向量存储、完整文档存储、索引 manifest 等运行数据，以及匹配的本地运行环境。
相邻分块可能有重叠，不要直接拼接作为去重原文。

## 需要直接恢复查询时

本目录仍为分块与图谱包。完整运行索引已另外提供在 [Novel-4128-runtime](../Novel-4128-runtime/README.md)，请恢复该完整包，不要只复制本目录的几个文件。
