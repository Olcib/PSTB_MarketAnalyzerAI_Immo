#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
features_engineering_paris.py
-------------------------------------
Construction des features spécifiques pour Paris.
Entrées : data/processed/paris/listings.csv, calendar.csv, reviews.csv
Sorties : data/features/paris_features.csv (+ paris_month_occ.csv)
"""

import pandas as pd
from pathlib import Path
from textblob import TextBlob


# =====================================================
# Helper
# =====================================================
def compute_sentiment(text: str) -> float:
    """Retourne un score de sentiment (-1 à 1) basé sur TextBlob."""
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception:
        return 0.0


def ensure_id(df: pd.DataFrame, city: str) -> str:
    """
    Vérifie si 'id' ou 'listing_id' existe, sinon crée une clé artificielle.
    Retourne le nom de la colonne clé.
    """
    if "id" in df.columns:
        return "id"
    elif "listing_id" in df.columns:
        return "listing_id"
    else:
        df.insert(0, "listing_id", range(1, len(df) + 1))
        print(f"⚠️ {city}: aucun identifiant trouvé → listing_id artificiel créé")
        return "listing_id"


# =====================================================
# Feature engineering Paris
# =====================================================
def build_features(processed_dir="data/processed/paris", features_dir="data/features"):
    proc_path = Path(processed_dir)
    feat_path = Path(features_dir)
    feat_path.mkdir(parents=True, exist_ok=True)

    # Charger les fichiers
    listings_fp = proc_path / "listings.csv"
    calendar_fp = proc_path / "calendar.csv"
    reviews_fp = proc_path / "reviews.csv"

    if not (listings_fp.exists() and calendar_fp.exists() and reviews_fp.exists()):
        raise FileNotFoundError(f"❌ Fichiers manquants dans {proc_path}")

    listings = pd.read_csv(listings_fp, on_bad_lines="skip", engine="python")
    calendar = pd.read_csv(calendar_fp, on_bad_lines="skip", engine="python")
    reviews = pd.read_csv(reviews_fp, on_bad_lines="skip", engine="python")

    # Vérifier clé primaire
    id_col = ensure_id(listings, "Paris")

    # ================== Features listings ==================
    if "amenities" in listings.columns:
        listings["amenities_count"] = listings["amenities"].apply(
            lambda x: len(str(x).split(",")) if pd.notnull(x) else 0
        )
    else:
        listings["amenities_count"] = 0

    if "host_is_superhost" in listings.columns:
        listings["is_superhost"] = listings["host_is_superhost"].map({"t": 1, "f": 0}).fillna(0)
    else:
        listings["is_superhost"] = 0

    # ================== Features calendar ==================
    occ_proxy = pd.Series(index=listings[id_col], dtype=float)

    if not calendar.empty and "available" in calendar.columns:
        calendar["available"] = calendar["available"].map({"t": 1, "f": 0}).fillna(calendar["available"])
        calendar["available"] = pd.to_numeric(calendar["available"], errors="coerce").fillna(0)

        occ = calendar.groupby("listing_id")["available"].apply(lambda x: 1 - x.mean())
        occ_proxy.loc[occ.index] = occ

    # ================== Features reviews ==================
    sentiment = pd.Series(dtype=float)
    if not reviews.empty and "comments" in reviews.columns:
        reviews["sentiment_score"] = reviews["comments"].apply(compute_sentiment)
        sentiment = reviews.groupby("listing_id")["sentiment_score"].mean()

    # ================== Merge ==================
    feats = listings.copy()

    if not occ_proxy.empty:
        feats = feats.merge(occ_proxy.rename("occupancy_rate"), left_on=id_col, right_index=True, how="left")

    if not sentiment.empty:
        feats = feats.merge(sentiment.rename("sentiment_score"), left_on=id_col, right_index=True, how="left")

    # ================== Monthly occupancy ==================
    if not calendar.empty and "date" in calendar.columns:
        calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
        calendar = calendar.dropna(subset=["date"])
        calendar["month"] = calendar["date"].dt.month
        month_occ = calendar.groupby(["listing_id", "month"])["available"].apply(lambda x: 1 - x.mean()).reset_index()
        month_occ.rename(columns={"available": "monthly_occ"}, inplace=True)
        month_occ.to_csv(feat_path / "paris_month_occ.csv", index=False, encoding="utf-8")

    # Sauvegarde
    out_path = feat_path / "paris_features.csv"
    feats.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Features Paris sauvegardées dans {out_path}")


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    try:
        build_features()
    except Exception as e:
        print(f"⚠️ Erreur Paris: {e}")
