#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_datasets.py
-------------------------------------
Prépare les datasets réduits choisis (clean8000, clean20000, etc.)
et les renomme en fichiers standard pour le pipeline de features.
"""

import shutil
from pathlib import Path


def prepare_city(city, mapping, processed_dir="data/processed"):
    """
    Copie/renomme les bons fichiers de travail pour une ville donnée.
    mapping = dict avec clés 'listings', 'calendar', 'reviews'
    """
    city_path = Path(processed_dir) / city

    for target, source in mapping.items():
        src_fp = city_path / source
        dst_fp = city_path / f"{target}.csv"

        if not src_fp.exists():
            print(f"❌ Fichier introuvable : {src_fp}")
            continue

        # Copie avec overwrite
        shutil.copy(src_fp, dst_fp)
        print(f"✅ {city}: {source} → {dst_fp.name}")


def main():
    # Mappings spécifiques à tes choix
    mappings = {
        "paris": {
            "listings": "listings_clean8000.csv",
            "calendar": "calendar_clean.csv",
            "reviews": "reviews_clean8000.csv",
        },
        "seattle": {
            "listings": "listings_clean.csv",
            "calendar": "calendar_clean.csv",
            "reviews": "reviews_clean20000.csv",
        },
    }

    for city, mapping in mappings.items():
        print(f"\n🔧 Préparation des fichiers pour {city.capitalize()}...")
        prepare_city(city, mapping)


if __name__ == "__main__":
    main()