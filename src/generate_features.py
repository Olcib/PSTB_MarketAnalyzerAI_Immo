import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ==========================================
# CHEMINS CORRIGÉS - ANNULE & REMPLACE
# ==========================================

# Chemins d'entrée (depuis data/features/)
INPUT_PARIS = "data/features/paris_features.csv"
INPUT_SEATTLE = "data/features/seattle_features.csv"

# Chemins de sortie (vers data/processed/)
OUTPUT_PARIS = "data/processed/enriched_paris.csv"
OUTPUT_SEATTLE = "data/processed/enriched_seattle.csv"

# Créer le dossier processed s'il n'existe pas
os.makedirs("data/processed", exist_ok=True)

def add_temporal_features(df):
    """
    Ajoute des features temporelles au DataFrame
    """
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['quarter'] = df['date'].dt.quarter
        
        # Saisons (pour hémisphère nord)
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
    
    return df

def add_statistical_features(df, target_cols=None):
    """
    Ajoute des features statistiques
    """
    if target_cols is None:
        # Sélectionner automatiquement les colonnes numériques
        target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclure les colonnes temporelles déjà créées
        exclude_cols = ['year', 'month', 'day', 'day_of_week', 'week_of_year', 'is_weekend', 'quarter']
        target_cols = [col for col in target_cols if col not in exclude_cols]
    
    for col in target_cols:
        if col in df.columns:
            # Features de rolling window (7 jours)
            df[f'{col}_rolling_7_mean'] = df[col].rolling(window=7, min_periods=1).mean()
            df[f'{col}_rolling_7_std'] = df[col].rolling(window=7, min_periods=1).std()
            
            # Features de lag
            df[f'{col}_lag_1'] = df[col].shift(1)
            df[f'{col}_lag_7'] = df[col].shift(7)
            
            # Différences
            df[f'{col}_diff_1'] = df[col].diff(1)
            df[f'{col}_diff_7'] = df[col].diff(7)
    
    return df

def add_city_specific_features(df, city="Unknown"):
    """
    Ajoute des features spécifiques à la ville
    """
    df['city'] = city
    
    # Encodage de la ville
    city_mapping = {'Paris': 1, 'Seattle': 2}
    df['city_encoded'] = df['city'].map(city_mapping).fillna(0)
    
    # Features spécifiques selon la ville
    if city.lower() == 'paris':
        # Paris : ajout de features liées au climat européen continental
        if 'month' in df.columns:
            df['is_high_tourism_season'] = df['month'].isin([6, 7, 8, 12]).astype(int)
            df['is_rainy_season'] = df['month'].isin([10, 11, 12, 1, 2]).astype(int)
    
    elif city.lower() == 'seattle':
        # Seattle : ajout de features liées au climat océanique
        if 'month' in df.columns:
            df['is_dry_season'] = df['month'].isin([7, 8, 9]).astype(int)
            df['is_very_rainy_season'] = df['month'].isin([11, 12, 1, 2]).astype(int)
    
    return df

def enrich_file(input_filepath, output_filepath, city="Unknown"):
    """
    Enrichit un fichier CSV avec de nouvelles features
    """
    print(f"▶ Enrichissement {city}...")
    
    # Lecture du fichier
    df = pd.read_csv(input_filepath)
    print(f"  📊 Données originales : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Sauvegarde du nombre de colonnes original
    original_cols = df.shape[1]
    
    # Ajout des features temporelles
    df = add_temporal_features(df)
    
    # Ajout des features statistiques
    df = add_statistical_features(df)
    
    # Ajout des features spécifiques à la ville
    df = add_city_specific_features(df, city=city)
    
    # Nettoyage des valeurs infinies et NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Affichage du résumé
    new_cols = df.shape[1]
    added_cols = new_cols - original_cols
    print(f"  ✅ Nouvelles features ajoutées : {added_cols}")
    print(f"  📊 Données enrichies : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Sauvegarde
    df.to_csv(output_filepath, index=False)
    print(f"  💾 Sauvegardé dans : {output_filepath}")
    
    return df

def main():
    """
    Fonction principale
    """
    print("🚀 Génération des features enrichies")
    print("="*50)
    
    try:
        # Enrichissement Paris
        if os.path.exists(INPUT_PARIS):
            enrich_file(INPUT_PARIS, OUTPUT_PARIS, city="Paris")
        else:
            print(f"⚠️ Fichier manquant : {INPUT_PARIS}")
        
        print()
        
        # Enrichissement Seattle  
        if os.path.exists(INPUT_SEATTLE):
            enrich_file(INPUT_SEATTLE, OUTPUT_SEATTLE, city="Seattle")
        else:
            print(f"⚠️ Fichier manquant : {INPUT_SEATTLE}")
            
        print("\n✅ Enrichissement terminé avec succès !")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'enrichissement : {str(e)}")
        raise

if __name__ == "__main__":
    main()