#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preview_features.py
-------------------------------------
Affiche un aperçu rapide des features générées (Paris & Seattle).
Affiche en console :
- dimensions (lignes × colonnes)
- noms des colonnes
- 20 premières lignes

Exporte :
- Résumé global (CSV) dans outputs/reports/features_preview_summary.csv
- Excel multi-onglets (Résumé + Paris + Seattle) dans outputs/reports/features_preview_summary.xlsx
"""

import pandas as pd
from pathlib import Path

def preview_file(filepath: Path, city: str, n=20):
    if not filepath.exists():
        print(f"❌ Fichier introuvable : {filepath}")
        return None, None
    try:
        df = pd.read_csv(filepath, engine="python")
        print(f"\n=== {filepath} ===")
        print(f"Shape: {df.shape[0]} lignes × {df.shape[1]} colonnes")
        print(f"Colonnes ({len(df.columns)}): {list(df.columns)}\n")
        print(df.head(n).to_string(index=False))
        summary = {
            "city": city,
            "filepath": str(filepath),
            "rows": df.shape[0],
            "cols": df.shape[1],
            "columns": "; ".join(df.columns)
        }
        return summary, df.head(n)
    except Exception as e:
        print(f"⚠️ Erreur lors de la lecture de {filepath}: {e}")
        return {"city": city, "filepath": str(filepath), "error": str(e)}, None

def main():
    base = Path("data/features")
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "Paris": base / "paris_features.csv",
        "Seattle": base / "seattle_features.csv"
    }

    summary = []
    preview_dfs = {}

    for city, f in files.items():
        res, df_preview = preview_file(f, city, n=20)
        if res:
            summary.append(res)
        if df_preview is not None:
            preview_dfs[city] = df_preview

    # Export résumé CSV
    if summary:
        df_summary = pd.DataFrame(summary)
        csv_fp = reports_dir / "features_preview_summary.csv"
        df_summary.to_csv(csv_fp, index=False, encoding="utf-8")
        print(f"\n📊 Résumé exporté dans {csv_fp}")

        # Export Excel multi-onglets
        xlsx_fp = reports_dir / "features_preview_summary.xlsx"
        with pd.ExcelWriter(xlsx_fp, engine="openpyxl") as writer:
            df_summary.to_excel(writer, sheet_name="Résumé", index=False)
            for city, df_prev in preview_dfs.items():
                df_prev.to_excel(writer, sheet_name=city, index=False)
        print(f"📊 Excel exporté dans {xlsx_fp}")

if __name__ == "__main__":
    main()
