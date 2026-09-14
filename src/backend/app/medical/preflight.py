"""Read-only runtime diagnostics; never downloads or loads model weights."""
import importlib.metadata
import json
import subprocess


PATHRAG_COMMIT = "32567bfc93605b8393996d5fa9ccdc0edbb865b2"
LIGHTRAG_COMMIT = "28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c"


def pathrag_revision(upstream):
    """Revision of a PathRAG source tree.

    A real checkout reports its own Git HEAD. The vendored `deps/PathRAG` copy
    ships without `.git`, where `git rev-parse` would silently resolve to the
    enclosing repository, so it carries an `UPSTREAM_COMMIT` marker instead.
    """
    if (upstream / ".git").exists():
        try:
            result = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                                    check=True, capture_output=True, text=True, timeout=10)
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    marker = upstream / "UPSTREAM_COMMIT"
    if not marker.is_file():
        return None
    try:
        for line in marker.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    except OSError:
        return None
    return None


def runtime_issues(settings):
    issues = []
    for name, expected in [("scikit-learn", "1.7.2"), ("lightrag-hku", "1.5.7"),
                           ("joblib", None), ("torch", None), ("transformers", None),
                           ("nano-vectordb", None), ("tiktoken", None)]:
        try:
            actual = importlib.metadata.version(name)
            if expected and actual != expected:
                issues.append(f"{name} 需要 {expected}，当前为 {actual}；请使用 Medical 独立环境。")
        except importlib.metadata.PackageNotFoundError:
            issues.append(f"Medical 独立环境缺少依赖：{name}。")
    path = settings.medical_embedding_path
    if not path or not (path / "config.json").is_file():
        issues.append("未配置本地 BGE-M3 模型：下载后设置 --medical-embedding-path。")
    else:
        try:
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            if config.get("hidden_size") != 1024 or config.get("model_type") != "xlm-roberta":
                issues.append("模型配置不是预期的 BGE-M3（XLM-RoBERTa / 1024 维）。")
            if not any((path / name).is_file() for name in ("pytorch_model.bin", "model.safetensors",
                                                           "pytorch_model.bin.index.json", "model.safetensors.index.json")):
                issues.append("BGE-M3 目录缺少权重文件。")
            if not (path / "tokenizer_config.json").is_file():
                issues.append("BGE-M3 目录缺少 tokenizer_config.json。")
        except (ValueError, OSError):
            issues.append("BGE-M3 config.json 无法读取。")
    upstream = settings.medical_pathrag_root
    if not upstream or not (upstream / "PathRAG/__init__.py").is_file():
        issues.append("未配置 PathRAG 源码：设置 --medical-pathrag-root，固定提交 " + PATHRAG_COMMIT + "。")
    else:
        actual = pathrag_revision(upstream)
        if actual is None:
            issues.append("无法确认 PathRAG 固定 Git 提交：既不是独立 Git 检出，也没有 UPSTREAM_COMMIT 标记。")
        elif actual != PATHRAG_COMMIT:
            issues.append("PathRAG 源码版本与 Medical 索引要求不符。")
    return issues
