#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest_core.py
---------------
Load and preprocess raw Airbnb data (listings, calendar, reviews).

Steps:
1. Load raw CSVs from data/raw/<city>
2. Check if files exist, empty or readable
3. Save to data/processed/<city>
4. Update DATA_SOURCES.md
"""

import os
import pandas as pd

RAW_DIR = "data/raw"
PROC_DIR = "data/processed"

CITIES = ["paris", "seattle"]
FILES = ["listings.csv", "calendar.csv", "reviews.csv"]


def load_csv_safe(path: str, city: str, fname: str):
    """Try loading a CSV with explicit error messages."""
    if not os.path.exists(path):
        print(f"❌ Missing file for {city}: {fname}")
        return None

    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"⚠️ File found but empty for {city}: {fname}")
        else:
            print(f"✅ Loaded {fname} for {city} → shape = {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error reading {fname} for {city}: {e}")
        return None


def ingest_city(city: str):
    print(f"\n🔎 Processing {city}...")

    base_raw = os.path.join(RAW_DIR, city)
    base_proc = os.path.join(PROC_DIR, city)
    os.makedirs(base_proc, exist_ok=True)

    data = {}

    for fname in FILES:
        fpath = os.path.join(base_raw, fname)
        df = load_csv_safe(fpath, city, fname)
        if df is not None:
            # Save processed copy
            out_fp = os.path.join(base_proc, fname)
            df.to_csv(out_fp, index=False)
            data[fname] = out_fp

    # Update sources documentation
    update_sources(city, data)


def update_sources(city: str, data: dict):
    """Append info to DATA_SOURCES.md for traceability."""
    md_file = "DATA_SOURCES.md"
    with open(md_file, "a", encoding="utf-8") as f:
        f.write(f"\n### {city}\n")
        for fname, path in data.items():
            f.write(f"- {fname}: {path}\n")
    print(f"📖 DATA_SOURCES.md mis à jour pour {city}")


if __name__ == "__main__":
    for city in CITIES:
        ingest_city(city)

