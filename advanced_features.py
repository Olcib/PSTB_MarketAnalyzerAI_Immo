# Script: day4_advanced_features.py
def load_enriched_features():
    # Charger TOUTES les colonnes des fichiers enrichis
    df_paris = pd.read_csv("data/processed/enriched_paris_fixed.csv")
    df_seattle = pd.read_csv("data/processed/enriched_seattle_fixed.csv")
    
    # Sélectionner features géographiques
    geo_features = ['latitude', 'longitude', 'neighbourhood_cleansed']
    
    # Features de qualité
    quality_features = ['review_scores_*', 'host_*', 'property_type']
    
    # Features temporelles enrichies
    temporal_features = ['*_rolling_*', '*_lag_*', 'season', 'quarter']
    
return enhanced_dataset_with_100_features