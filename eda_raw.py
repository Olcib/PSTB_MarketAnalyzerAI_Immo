#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eda_raw.py
----------
Exploration initiale des fichiers RAW (paris & seattle).
- head(), info(), describe(), shape, columns
- Exporte résultats dans un fichier Excel unique (1 onglet par dataset).
"""

import os
import pandas as pd

RAW_DIR = "data/raw"
CITIES = ["paris", "seattle"]
FILES = ["listings.csv", "calendar.csv", "reviews.csv"]

OUT_FP = "EDA_raw.xlsx"


def explore_file(city, fname):
    """Retourne un dictionnaire de DataFrames (meta, head, info, describe_num, describe_obj)."""
    path = os.path.join(RAW_DIR, city, fname)
    if not os.path.exists(path):
        print(f"❌ Fichier introuvable: {path}")
        return None

    print(f"\n📂 {city.upper()} — {fname}")
    df = pd.read_csv(path)

    # ---- Shape & colonnes ----
    meta = pd.DataFrame({
        "Shape": [df.shape],
        "Columns": [list(df.columns)]
    })

    # ---- Aperçu head ----
    head_df = df.head()

    # ---- Info simplifié ----
    info_df = pd.DataFrame({
        "Nb lignes": [df.shape[0]],
        "Nb colonnes": [df.shape[1]],
        "Types": [df.dtypes.astype(str).to_dict()]
    })

    # ---- Describe numérique ----
    desc_num = df.describe().transpose()

    # ---- Describe objets (limité à 20 colonnes pour éviter lenteur) ----
    obj_cols = df.select_dtypes(include=['object']).columns[:20]
    desc_obj = df[obj_cols].describe().transpose() if len(obj_cols) > 0 else pd.DataFrame()

    return {
        "meta": meta,
        "head": head_df,
        "info": info_df,
        "desc_num": desc_num,
        "desc_obj": desc_obj
    }


if __name__ == "__main__":
    # Supprimer ancien fichier
    if os.path.exists(OUT_FP):
        os.remove(OUT_FP)

    # Ouvrir UNE SEULE fois ExcelWriter
    with pd.ExcelWriter(OUT_FP, engine="openpyxl") as writer:
        for city in CITIES:
            for fname in FILES:
                results = explore_file(city, fname)
                if results:
                    sheet_name = f"{city}_{fname.replace('.csv','')}"[:31]  # max 31 chars

                    # Écrire les différents DataFrames en continu sur la même feuille
                    row = 0
                    for key, df in results.items():
                        if df is not None and not df.empty:
                            # Ajouter un titre (nom de la section)
                            title_df = pd.DataFrame({f"=== {key.upper()} ===": []})
                            title_df.to_excel(writer, sheet_name=sheet_name, startrow=row, index=False)
                            row += 1
                            # Ajouter le contenu
                            df.to_excel(writer, sheet_name=sheet_name, startrow=row, index=True)
                            row += len(df) + 3

    print(f"\n✅ Rapport Excel généré: {OUT_FP}")


