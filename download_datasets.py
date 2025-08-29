import os
import subprocess
import pandas as pd
from pathlib import Path
import requests

# ---------- Config ----------
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "seattle": {
        "source": "Kaggle - Airbnb Seattle",
        "kaggle": "airbnb/seattle",
        "files": ["calendar.csv", "listings.csv", "reviews.csv"]
    },
    "paris": {
        "source": "InsideAirbnb - Paris",
        "urls": {
            "listings.csv.gz": "http://data.insideairbnb.com/france/ile-de-france/paris/2025-06-06/data/listings.csv.gz",
            "calendar.csv.gz": "http://data.insideairbnb.com/france/ile-de-france/paris/2025-06-06/data/calendar.csv.gz",
            "reviews.csv.gz": "http://data.insideairbnb.com/france/ile-de-france/paris/2025-06-06/data/reviews.csv.gz"
        }
    }
}
# ---------- Utilitaires ----------
def download_kaggle(dataset: str, target_dir: Path):
    """Télécharge un dataset Kaggle dans target_dir"""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement Kaggle : {dataset} -> {target_dir}")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(target_dir), "--unzip"],
            check=True
        )
    except Exception as e:
        print(f"Erreur Kaggle ({dataset}): {e}")

def download_url(url: str, target_path: Path):
    """Télécharge un fichier depuis une URL"""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement URL : {url} -> {target_path}")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(target_path, "wb") as f:
            f.write(r.content)
    else:
        print(f"Erreur {r.status_code} pour {url}")

def check_schema(path: Path, required_cols: list):
    """Vérifie que les colonnes essentielles existent"""
    try:
        df = pd.read_csv(path, nrows=5)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return f"{path.name}: colonnes manquantes {missing}"
        return f"{path.name}: OK"
    except Exception as e:
        return f"{path.name}: Erreur lecture ({e})"

# ---------- Pipeline ----------
report_lines = ["# Data Sources Report\n"]

# Seattle
seattle_dir = RAW_DIR / "seattle"
download_kaggle(DATASETS["seattle"]["kaggle"], seattle_dir)
report_lines.append("## Seattle (Kaggle)")
report_lines.append(f"- Source: {DATASETS['seattle']['source']}")
report_lines.append("- URL: https://www.kaggle.com/datasets/airbnb/seattle\n")

for fname in DATASETS["seattle"]["files"]:
    fpath = seattle_dir / fname
    if fpath.exists():
        check = check_schema(fpath, ["id", "price", "latitude", "longitude"])
        report_lines.append(f"- {fname}: {check}")
    else:
        report_lines.append(f"- {fname}: non trouvé")

# Paris
paris_dir = RAW_DIR / "paris"
report_lines.append("\n## Paris (InsideAirbnb)")
report_lines.append(f"- Source: {DATASETS['paris']['source']}")
report_lines.append("- URL: http://insideairbnb.com/get-the-data.html\n")

for fname, url in DATASETS["paris"]["urls"].items():
    target = paris_dir / fname
    if not target.exists():
        download_url(url, target)
    if target.exists():
        check = check_schema(target, ["id", "price", "latitude", "longitude"])
        report_lines.append(f"- {fname}: {check}")
    else:
        report_lines.append(f"- {fname}: non trouvé")

# ---------- Sauvegarde rapport ----------
with open("DATA_SOURCES.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("Rapport généré: DATA_SOURCES.md")