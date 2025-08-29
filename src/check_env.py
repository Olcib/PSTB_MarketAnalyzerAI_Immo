import json, os, sys, platform
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

PKGS = [
    "pandas","numpy","scikit-learn","xgboost","lightgbm","tensorflow","torch",
    "transformers","langchain","chromadb","faiss-cpu","openai","streamlit","plotly","altair"
]

def pkg_ver(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT INSTALLED"

report = {
    "python": sys.version,
    "platform": platform.platform(),
    "venv_active": os.environ.get("VIRTUAL_ENV") is not None,
    "packages": {p: pkg_ver(p) for p in PKGS},
    "env_vars": {
        "OPENAI_API_KEY": "SET" if os.getenv("OPENAI_API_KEY") else "MISSING",
        "HUGGINGFACEHUB_API_TOKEN": "SET" if os.getenv("HUGGINGFACEHUB_API_TOKEN") else "MISSING"
    }
}

# On force l’écriture à la racine du projet
project_root = Path(__file__).resolve().parent.parent
output_file = project_root / "env_report.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(f"✅ Environment report written to {output_file}")
print(json.dumps(report, indent=2))

print("Script executed from:", os.getcwd())

