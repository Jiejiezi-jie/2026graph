# Novel-4128 完整运行索引交接

这个包用于恢复已经构建好的小说索引，让队友使用同一份分块、图谱、实体向量、关系向量和文本块向量查询，避免重新执行实体抽取和建图。它补全旧 `Novel-4128-chunks` 包缺失的运行索引。

## 包含内容

- `data/active/`：Novel-4128 原文及数据清单。
- `data/workspace/`：完整 LightRAG 存储，包含 GraphML、三个 `vdb_*.json` 向量文件、完整文档/分块/实体/关系记录、处理状态、关联映射、模型响应缓存及索引清单。
- `runtime.json`：索引构建配置、版本及数量。
- `runtime.env.example`：无密钥配置模板。
- `python-packages-observed.json`：导出时本机实际包版本快照，供排查版本差异；不是跨平台安装锁文件。
- `checksums.json`：SHA-256 校验和。
- `restore.py`：标准库恢复工具，校验文件且拒绝覆盖已有运行目录。

数据规模：51 个文本块、729 个图谱节点、1,369 条关系；向量维度 384。仅此一篇小说，不是 GraphRAG-Benchmark 全量小说集合。

包内不含 API Key、个人 `.env`、网页查询历史数据库、Python/Node 环境或 embedding 模型权重。已有文本的 embedding 向量已包含；新查询仍需使用相同 embedding 模型生成查询向量。首次安装模型需要联网，离线机器需另行准备该模型缓存。

## 1. 安装匹配环境

先获取知识检索工作台代码；在远程交接分支中，项目根目录是 `knowledge-workbench/`，不是上一级队友的 GraphRAG 项目。以下命令中的 `python` 应来自一个新建的 Python 3.10 环境。例如：

```powershell
conda create -n knowledge-workbench python=3.10 -y
conda activate knowledge-workbench
git clone https://github.com/HKUDS/LightRAG.git ../LightRAG-runtime
git -C ../LightRAG-runtime checkout c1248646e4eda4d89054926af2e094730daf23fe
python -m pip install -e ../LightRAG-runtime
python -m pip install -e ./backend
```

LightRAG 必须使用此固定提交；不要直接安装另一版本的 PyPI LightRAG 替代。前端需要 Node.js 22.12+（本项目使用 Vite 7），安装时使用仓库中的 `package-lock.json` 执行 `npm ci`。

embedding 模型为 `BAAI/bge-small-en-v1.5`，本机缓存 revision 为 `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`。可提前获取并核对该快照：

```powershell
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('BAAI/bge-small-en-v1.5', revision='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'))"
```

工作台当前按模型 ID 加载；如果远端 `main` revision 日后变化，需要保证本地加载的模型仍与记录快照一致。索引文件和恢复校验可精确复现；外部 API 的未来模型行为、逐字答案和运行耗时不保证完全一致。

## 2. 恢复现成索引

在项目根目录执行。如果收到 ZIP，先完整解压到 `exports/Novel-4128-runtime/`，使其中能看到 `restore.py` 和 `data/`。

```powershell
python exports/Novel-4128-runtime/restore.py --verify-only
python exports/Novel-4128-runtime/restore.py --target .
```

恢复后应有 `data/active/corpus.json` 和 `data/workspace/vdb_chunks.json` 等文件。恢复操作不调用模型、不重新分块、不重建图谱。

若提示目标已存在，先停止后端，备份原有 `data/active` 与 `data/workspace`，再恢复到空目录。脚本不会覆盖原数据。

不需要另行下载 GraphRAG-Benchmark 原始语料来查询这篇小说：原文已放在 `data/active/corpus.json`。若使用网页“重新选择数据”功能，才需要配置你本机的 benchmark 原始数据路径。

## 3. 保持配置一致并启动

`runtime.env.example` 中除 Key 外的配置必须保持一致，否则索引指纹不匹配。尤其不要随意改变 embedding 模型/维度、分块参数、LLM 模型名或 URL。当前指纹也包含 LLM 模型名和 URL。

在新环境、没有个人 `.env` 时：

```powershell
Copy-Item exports/Novel-4128-runtime/runtime.env.example .env
```

如果已有 `.env`，请对照模板修改，不要覆盖自己的设置。将 `DEEPSEEK_API_KEY` 填成你自己的有效 Key；也可留空，在启动终端设置环境变量，或后端启动后到网页 API 设置中填写。

终端一（项目根目录）：

```powershell
$env:DEEPSEEK_API_KEY = '你自己的有效 Key'
python scripts/serve_web.py
```

终端二：

```powershell
cd frontend
npm ci
npm run dev
```

打开 `http://127.0.0.1:5173/`。数据应显示 Novel-4128，索引应已就绪。直接在检索工作台查询，无需点击“重新建索引”。local/global/hybrid/mix/vector 均使用这套索引；vector 底层名称为 naive。

有完整索引仍需要有效 Key 来生成回答，图谱检索模式还可能调用模型提取查询关键词。这些是查询调用，不是重新建图费用。只共享这个索引包不能代替后端代码、模型安装及 Key 配置。

## 验证范围

本包已校验文件完整性，并在全新临时数据目录恢复后，用交付后端检查 corpus/runtime 指纹、文档 processed 状态、图谱及三个向量存储。固定版本 LightRAG 也已成功初始化所有存储，三套向量矩阵维度均为 384，详见 `verification.json`。验证不发送新的付费查询；队友机器上的依赖安装和实际 API 连通性需要在当地确认。

## 上传注意

日常运行的 `data/active/`、`data/workspace/` 仍保持 Git 忽略；应上传整个 `exports/Novel-4128-runtime/` 快照目录及说明，而不是只上传 GraphML。ZIP 便于直接发送，但本仓库默认忽略 ZIP；Git 交付使用展开的快照目录。
