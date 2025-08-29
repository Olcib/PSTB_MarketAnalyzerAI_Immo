#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
features_to_excel_light.py
-------------------------------------
Produit un Excel léger lisible pour le jury (≤ 50 000 lignes par onglet) avec XlsxWriter.
- Onglets : Paris, Seattle, Paris_MonthOcc, Seattle_MonthOcc (échantillons 50k)
- Onglet 'Résumé' (nb total de lignes, nb exportées, nb colonnes, chemin)
- PAS de chunking géant, PAS d'openpyxl, PAS de styles lourds
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
    # Lecture tolérante aux lignes corrompues
    return pd.read_csv(fp, engine="python", on_bad_lines="skip")

def main():
    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    excel_fp = out_dir / "features_export_resume.xlsx"

    summary = []
    # IMPORTANT: forcer xlsxwriter (plus fiable & rapide que openpyxl à l'écriture)
    with pd.ExcelWriter(
        excel_fp,
        engine="xlsxwriter",
        engine_kwargs={"options": {"strings_to_urls": False}}  # pas d'auto-détection d'URL (ralentit)
    ) as writer:
        for name, fp in FILES.items():
            if not fp.exists():
                summary.append({"name": name, "filepath": str(fp),
                                "rows_total": 0, "cols": 0, "rows_exported": 0,
                                "note": "❌ Missing file"})
                continue

            try:
                df = read_csv_robust(fp)
                rows_total, cols = df.shape
                # échantillon max 50k pour Excel (fiable & rapide)
                sample = df.head(EXCEL_LIMIT)
                sheet = name[:31]  # Excel limite à 31 caractères
                # écriture brute, sans formats/styling
                sample.to_excel(writer, sheet_name=sheet, index=False)
                summary.append({"name": name, "filepath": str(fp),
                                "rows_total": rows_total, "cols": cols,
                                "rows_exported": len(sample), "note": ""})
            except Exception as e:
                summary.append({"name": name, "filepath": str(fp),
                                "rows_total": 0, "cols": 0, "rows_exported": 0,
                                "note": f"⚠️ Read/Write error: {e}"})

        # Onglet Résumé
        pd.DataFrame(summary).to_excel(writer, sheet_name="Résumé", index=False)

    print(f"✅ Excel léger généré → {excel_fp}")

if __name__ == "__main__":
    main()
