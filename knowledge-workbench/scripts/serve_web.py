"""Start only the local web backend. Never builds indexes on startup."""
import argparse
import sys
import os
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--medical-bundle", type=Path, help="导入的 Medical bundle 目录")
    parser.add_argument("--medical-embedding-path", type=Path, help="本地 BGE-M3 模型目录")
    parser.add_argument("--medical-pathrag-root", type=Path, help="固定版本 PathRAG 仓库目录")
    parser.add_argument("--medical-device", default="cpu")
    parser.add_argument("--check", action="store_true", help="校验索引与运行环境，不启动服务或调用 API")
    parser.add_argument("--preview-only", action="store_true", help="依赖不齐时仍允许启动以查看已有图谱")
    args = parser.parse_args()
    if args.medical_bundle:
        os.environ["MEDICAL_BUNDLE"] = str(args.medical_bundle.resolve())
        os.environ["MEDICAL_DEVICE"] = args.medical_device
        for key in ("medical_embedding_path", "medical_pathrag_root"):
            value = getattr(args, key)
            if value:
                os.environ[key.upper()] = str(value.resolve())
        from app.config import Settings
        from app.medical.bundle import MedicalBundle
        from app.medical.preflight import runtime_issues
        from app.domain.errors import AppError
        settings = Settings(app_root=ROOT / "backend", _env_file=ROOT / "backend/.env")
        try:
            bundle = MedicalBundle(settings.medical_bundle)
        except AppError as exc:
            parser.exit(2, exc.safe_message + "\n")
        print(json.dumps(bundle.summary(), ensure_ascii=False, indent=2))
        issues = runtime_issues(settings)
        for issue in issues:
            print("- " + issue)
        if args.check:
            print("Index audit passed; runtime incomplete." if issues else "Index and runtime preflight passed.")
            raise SystemExit(2 if issues else 0)
        if issues and not args.preview_only:
            parser.exit(2, "Medical runtime is incomplete. Install dependencies/model or use --preview-only.\n")
    elif args.check or args.preview_only or args.medical_embedding_path or args.medical_pathrag_root:
        parser.error("Medical 参数需与 --medical-bundle 一起使用。")
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
