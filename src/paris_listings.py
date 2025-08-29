#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paris_listings.py
-------------------
Nettoyage des colonnes de listings Paris :
- Supprime les colonnes inutiles (❌ À retirer)
- Supprime aussi toutes les colonnes optionnelles (⚖️)
- Conserve uniquement les colonnes essentielles pour l’EDA et la modélisation
"""

import os
import pandas as pd

RAW_FP = "data/raw/paris/listings.csv"
OUT_FP = "data/processed/paris/listings_clean.csv"

if not os.path.exists(RAW_FP):
    raise FileNotFoundError(f"❌ Fichier introuvable : {RAW_FP}")

df = pd.read_csv(RAW_FP)

# -----------------------------
# 1. Colonnes à supprimer (❌ + ⚖️)
# -----------------------------
cols_to_drop = [
    # Colonnes inutiles (identifiants, urls, métadonnées…)
    "id", "listing_url", "scrape_id", "experiences_offered", "notes",
    "thumbnail_url", "medium_url", "picture_url", "xl_picture_url",
    "host_id", "host_url", "host_name", "host_about", "host_thumbnail_url",
    "host_picture_url", "host_total_listings_count", "host_has_profile_pic",
    "street", "neighbourhood_group_cleansed", "country_code", "square_feet",
    "calendar_updated", "calendar_last_scraped", "license", "jurisdiction_names",
    "require_guest_profile_picture", "require_guest_phone_verification",

    # Colonnes optionnelles (⚖️)
    "last_scraped", "name", "summary", "space", "description", "transit",
    "host_since", "host_location", "host_response_time", "host_response_rate",
    "host_acceptance_rate", "host_is_superhost", "host_neighbourhood",
    "host_listings_count", "host_verifications", "host_identity_verified",
    "neighbourhood", "market", "smart_location", "is_location_exact",
    "bed_type", "security_deposit", "maximum_nights", "has_availability",
    "availability_60", "availability_90", "first_review", "last_review",
    "requires_license", "instant_bookable", "calculated_host_listings_count"
]

# Supprimer uniquement celles qui existent (évite les erreurs KeyError)
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# -----------------------------
# 2. Sauvegarde du fichier nettoyé
# -----------------------------
os.makedirs(os.path.dirname(OUT_FP), exist_ok=True)
df.to_csv(OUT_FP, index=False)

print(f"\n✅ Paris listings nettoyé → {OUT_FP}")
print(f"➡️ Colonnes finales ({len(df.columns)}) : {list(df.columns)}")
