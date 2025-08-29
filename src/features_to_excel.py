#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
features_to_excel.py
-------------------------------------
1) Génère un Excel léger pour le jury : outputs/reports/features_export_resume.xlsx
   - Onglets : Paris, Seattle, Paris_MonthOcc, Seattle_MonthOcc (chacun limité à 50k lignes)
   - Onglet 'Résumé' avec statistiques
   - Moteur d'écriture: xlsxwriter (plus fiable et rapide)

2) (Optionnel) Exporte les jeux COMPLETS en Parquet individuel :
   outputs/parquet/<name>.parquet
   (si pyarrow/fastparquet indisponible -> skip proprement)
"""

from pathlib import Path
import pandas as pd

EXCEL_LIMIT = 50_000
FILES = {
    "Paris": Path("data/features/paris_features.csv"),
    "Seattle": Path("data/features/seattle_features.csv"),
    "Paris_MonthOcc": Path("data/features/paris_month_occ.csv"),
    "Seattle_MonthOcc": Path("data/features/seattle_month_occ.csv"),
}


def read_csv_robust(fp: Path) -> pd.DataFrame:
    """Lecture CSV tolérante."""
    return pd.read_csv(fp, engine="python", on_bad_lines="skip")


def main():
    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    excel_fp = out_dir / "features_export_resume.xlsx"

    # --- 1) Excel léger (50k max / onglet), moteur xlsxwriter ---
    summary_rows = []
    # Important: XlsxWriter pour éviter les blocages d'openpyxl sur gros tableaux
    with pd.ExcelWriter(
        excel_fp, engine="xlsxwriter",
        engine_kwargs={"options": {"strings_to_urls": False}}
    ) as writer:
        for name, fp in FILES.items():
            if not fp.exists():
                summary_rows.append({
                    "name": name, "filepath": str(fp),
                    "rows_total": 0, "cols": 0, "rows_exported": 0,
                    "note": "❌ Missing file"
                })
                continue

            try:
                df = read_csv_robust(fp)
                rows_total, cols = df.shape

                # échantillon pour Excel (<= 50k)
                sample = df.head(EXCEL_LIMIT)
                # écrire sans aucun style pour vitesse/fiabilité
                safe_sheet = name[:31]  # Excel limite le nom de feuille à 31 chars
                sample.to_excel(writer, sheet_name=safe_sheet, index=False)

                summary_rows.append({
                    "name": name, "filepath": str(fp),
                    "rows_total": rows_total, "cols": cols,
                    "rows_exported": len(sample), "note": ""
                })
            except Exception as e:
                summary_rows.append({
                    "name": name, "filepath": str(fp),
                    "rows_total": 0, "cols": 0, "rows_exported": 0,
                    "note": f"⚠️ Read/Write error: {e}"
                })

        # Onglet Résumé
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Résumé", index=False)

    print(f"✅ Excel léger généré → {excel_fp}")

    # --- 2) Parquet complet individuel (optionnel) ---
    parquet_dir = Path("outputs/parquet")
    parquet_dir.mkdir(parents=True, exist_ok=True)
    for name, fp in FILES.items():
        if not fp.exists():
            continue
        try:
            df_full = read_csv_robust(fp)
            # Tentative Parquet ; si pyarrow absent -> on passe
            try:
                df_full.to_parquet(parquet_dir / f"{name}.parquet", index=False)
                print(f"✅ Parquet complet → {parquet_dir / (name + '.parquet')}")
            except Exception as e_parq:
                # fallback : CSV gz pour garder une sortie complète compressée
                gz_fp = parquet_dir / f"{name}.csv.gz"
                df_full.to_csv(gz_fp, index=False, compression="infer")
                print(f"ℹ️ PyArrow/Fastparquet indisponible → fallback CSV.GZ → {gz_fp} ({e_parq})")
        except Exception as e:
            print(f"⚠️ Skip {name}: {e}")


if __name__ == "__main__":
    main()
