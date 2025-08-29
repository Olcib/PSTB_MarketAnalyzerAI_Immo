import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = "results/paris_consolidated/"
MODEL_DIR = "models/paris_final/"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots/", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_paris_complete():
    """
    Charge le dataset Paris complet avec toutes les colonnes
    """
    print("CONSOLIDATION DATASET PARIS - ACTION 1")
    print("="*50)
    
    df_paris = pd.read_csv("data/processed/enriched_paris_fixed.csv")
    
    print(f"Dataset Paris complet:")
    print(f"  Lignes: {df_paris.shape[0]:,}")
    print(f"  Colonnes: {df_paris.shape[1]:,}")
    print(f"  Taille: {df_paris.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return df_paris

def audit_data_quality(df):
    """
    Audit complet de la qualité des données Paris
    """
    print(f"\nAUDIT QUALITE DES DONNEES")
    print("="*35)
    
    # Analyse des valeurs manquantes
    missing_analysis = pd.DataFrame({
        'colonne': df.columns,
        'manquantes': df.isnull().sum(),
        'pourcentage': (df.isnull().sum() / len(df)) * 100,
        'type': df.dtypes
    }).sort_values('manquantes', ascending=False)
    
    print("Top 15 colonnes avec valeurs manquantes:")
    for _, row in missing_analysis.head(15).iterrows():
        if row['manquantes'] > 0:
            print(f"  {row['colonne']}: {row['manquantes']:,} ({row['pourcentage']:.1f}%)")
    
    # Colonnes à exclure (trop de valeurs manquantes)
    high_missing = missing_analysis[missing_analysis['pourcentage'] > 50]['colonne'].tolist()
    
    print(f"\nColonnes à exclure (>50% manquantes): {len(high_missing)}")
    for col in high_missing[:10]:
        print(f"  - {col}")
    
    # Analyse des doublons
    duplicates = df.duplicated().sum()
    print(f"\nDoublons: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")
    
    # Analyse des outliers sur le prix
    if 'price' in df.columns:
        price_stats = df['price'].describe()
        print(f"\nAnalyse prix (variable cible):")
        print(f"  Médiane: ${price_stats['50%']:.0f}")
        print(f"  Moyenne: ${price_stats['mean']:.0f}")
        print(f"  Q1-Q3: ${price_stats['25%']:.0f} - ${price_stats['75%']:.0f}")
        print(f"  Outliers potentiels (>Q3+3*IQR): {len(df[df['price'] > price_stats['75%'] + 3*(price_stats['75%'] - price_stats['25%'])])}")
    
    return missing_analysis, high_missing

def create_clean_features(df, high_missing_cols):
    """
    Création du dataset Paris nettoyé avec features optimales
    """
    print(f"\nCREATION FEATURES OPTIMALES")
    print("="*35)
    
    # Exclure colonnes problématiques
    exclude_patterns = ['Unnamed', 'index', 'id'] + high_missing_cols
    excluded_cols = []
    
    for col in df.columns:
        if any(pattern.lower() in col.lower() for pattern in exclude_patterns):
            excluded_cols.append(col)
    
    print(f"Colonnes exclues: {len(excluded_cols)}")
    
    # Dataset nettoyé
    clean_cols = [col for col in df.columns if col not in excluded_cols]
    df_clean = df[clean_cols].copy()
    
    print(f"Dataset nettoyé: {df_clean.shape[1]} colonnes")
    
    # Standardisation des noms
    if 'price' in df_clean.columns:
        df_clean['revenue'] = df_clean['price']
    if 'estimated_occupancy_l365d' in df_clean.columns:
        df_clean['occupancy_rate'] = df_clean['estimated_occupancy_l365d']
    if 'review_scores_rating' in df_clean.columns:
        df_clean['sentiment_score'] = df_clean['review_scores_rating']
    
    return df_clean

def optimize_feature_selection(df_clean):
    """
    Sélection optimale des features par importance
    """
    print(f"\nSELECTION FEATURES OPTIMALES")
    print("="*35)
    
    if 'revenue' not in df_clean.columns:
        print("Variable cible 'revenue' manquante")
        return None, None
    
    # Préparer les features
    target_col = 'revenue'
    exclude_targets = [target_col, 'price', 'sentiment_score', 'review_scores_rating']
    
    feature_cols = [col for col in df_clean.columns if col not in exclude_targets]
    
    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()
    
    print(f"Features candidates: {len(feature_cols)}")
    
    # Nettoyage des features
    categorical_cols = []
    for col in X.columns:
        if X[col].dtype == 'object':
            # Vérifier si c'est vraiment catégoriel
            unique_values = X[col].nunique()
            if unique_values < 50:  # Catégoriel
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str).fillna('Unknown'))
                categorical_cols.append(col)
            else:  # Trop de valeurs uniques, supprimer
                X = X.drop(col, axis=1)
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    print(f"Variables catégorielles encodées: {len(categorical_cols)}")
    print(f"Features numériques: {X.shape[1] - len(categorical_cols)}")
    
    # Imputation
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Éliminer features avec variance nulle
    zero_variance = X_imputed.columns[X_imputed.var() == 0].tolist()
    if zero_variance:
        X_imputed = X_imputed.drop(columns=zero_variance)
        print(f"Features variance nulle supprimées: {len(zero_variance)}")
    
    # Sélection des K meilleures features
    k_best = min(30, X_imputed.shape[1])  # Maximum 30 features
    
    # Nettoyer y
    y_clean = pd.to_numeric(y, errors='coerce').fillna(y.median())
    
    selector = SelectKBest(f_regression, k=k_best)
    X_selected = selector.fit_transform(X_imputed, y_clean)
    
    # Récupérer les noms des features sélectionnées
    selected_features = X_imputed.columns[selector.get_support()].tolist()
    
    X_final = pd.DataFrame(X_selected, columns=selected_features)
    
    print(f"Features finales sélectionnées: {len(selected_features)}")
    print("Top 15 features sélectionnées:")
    
    # Scores des features
    feature_scores = pd.DataFrame({
        'feature': X_imputed.columns,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    for i, (_, row) in enumerate(feature_scores.head(15).iterrows(), 1):
        selected = "✓" if row['feature'] in selected_features else " "
        print(f"  {i:2d}. {selected} {row['feature']}: {row['score']:.1f}")
    
    return X_final, y_clean, selected_features, imputer

def train_optimized_model(X, y, feature_names):
    """
    Entraînement du modèle Paris optimisé
    """
    print(f"\nENTRAINEMENT MODELE PARIS OPTIMISE")
    print("="*40)
    
    # Split avec stratification par quartiles de prix
    quartiles = pd.qcut(y, q=4, labels=False)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=quartiles
    )
    
    print(f"Split données:")
    print(f"  Train: {len(X_train):,} échantillons")
    print(f"  Test: {len(X_test):,} échantillons")
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=feature_names
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=feature_names
    )
    
    # Modèle LightGBM optimisé pour Paris
    model = lgb.LGBMRegressor(
        n_estimators=400,        # Plus d'arbres pour dataset large
        max_depth=10,            # Profondeur adaptée
        learning_rate=0.05,      # Learning rate conservateur
        num_leaves=80,           # Feuilles optimales
        min_child_samples=50,    # Régularisation
        subsample=0.8,           # Subsampling
        colsample_bytree=0.8,    # Feature sampling
        random_state=42,
        verbose=-1
    )
    
    print("Entraînement en cours...")
    model.fit(X_train_scaled, y_train)
    
    # Prédictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Métriques
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"\nPERFORMANCES FINALES:")
    print(f"  Train - RMSE: ${rmse_train:.2f}, R²: {r2_train:.3f}")
    print(f"  Test  - RMSE: ${rmse_test:.2f}, R²: {r2_test:.3f}")
    
    # Vérification overfitting
    overfitting_gap = r2_train - r2_test
    print(f"  Écart train/test: {overfitting_gap:.3f}")
    
    if overfitting_gap > 0.1:
        print("  ⚠️ Overfitting détecté")
    else:
        print("  ✅ Pas d'overfitting significatif")
    
    # Validation croisée
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"  CV R² (5-fold): {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
    
    # Feature importance
    print(f"\nTop 10 Features Importantes:")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']}: {row['importance']:.1f}")
    
    return model, scaler, r2_test, rmse_test, feature_importance

def save_final_model(model, scaler, imputer, feature_names, performance_metrics):
    """
    Sauvegarde du modèle final Paris
    """
    print(f"\nSAUVEGARDE MODELE FINAL")
    print("="*30)
    
    # Sauvegarder modèle et preprocessing
    joblib.dump(model, f"{MODEL_DIR}/paris_final_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/paris_scaler.pkl")
    joblib.dump(imputer, f"{MODEL_DIR}/paris_imputer.pkl")
    
    # Métadonnées
    metadata = {
        'model_type': 'LightGBM',
        'features': feature_names,
        'performance': {
            'r2_test': float(performance_metrics['r2']),
            'rmse_test': float(performance_metrics['rmse'])
        },
        'training_date': pd.Timestamp.now().isoformat(),
        'training_samples': 53455,
        'feature_count': len(feature_names)
    }
    
    import json
    with open(f"{MODEL_DIR}/paris_model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Modèle sauvegardé:")
    print(f"  Modèle: {MODEL_DIR}/paris_final_model.pkl")
    print(f"  Scaler: {MODEL_DIR}/paris_scaler.pkl")
    print(f"  Métadonnées: {MODEL_DIR}/paris_model_metadata.json")

def main():
    """
    Action 1 - Consolidation complète dataset Paris
    """
    print("🎯 ACTION 1 - CONSOLIDATION DATASET PARIS")
    print("="*60)
    
    try:
        # Étape 1: Charger données complètes
        df_paris = load_paris_complete()
        
        # Étape 2: Audit qualité
        missing_analysis, high_missing_cols = audit_data_quality(df_paris)
        
        # Étape 3: Nettoyage et features
        df_clean = create_clean_features(df_paris, high_missing_cols)
        
        # Étape 4: Sélection optimale
        X, y, feature_names, imputer = optimize_feature_selection(df_clean)
        
        if X is None:
            print("Échec de la préparation des données")
            return None
        
        # Étape 5: Entraînement optimisé
        model, scaler, r2, rmse, feature_importance = train_optimized_model(X, y, feature_names)
        
        # Étape 6: Sauvegarde
        save_final_model(model, scaler, imputer, feature_names, {'r2': r2, 'rmse': rmse})
        
        print(f"\n✅ ACTION 1 TERMINÉE AVEC SUCCÈS")
        print("="*45)
        print(f"Performance finale: R² = {r2:.3f}, RMSE = ${rmse:.2f}")
        print(f"Modèle prêt pour les actions suivantes")
        
        return {
            'model': model,
            'performance': {'r2': r2, 'rmse': rmse},
            'features': feature_names,
            'feature_importance': feature_importance
        }
        
    except Exception as e:
        print(f"\n❌ Erreur Action 1: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()