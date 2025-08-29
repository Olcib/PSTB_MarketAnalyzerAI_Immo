import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "results/day6_city_models/"
MODEL_DIR = "models/city_specific/"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots/", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_city_data():
    """
    Charge les données séparément pour chaque ville
    """
    print("CHARGEMENT DES DONNÉES PAR VILLE")
    print("="*45)
    
    df_paris = pd.read_csv("data/processed/enriched_paris_fixed.csv")
    df_seattle = pd.read_csv("data/processed/enriched_seattle_fixed.csv")
    
    # Préparation spécialisée pour Paris
    paris_clean = df_paris.copy()
    if 'estimated_occupancy_l365d' in paris_clean.columns:
        paris_clean['occupancy_rate'] = paris_clean['estimated_occupancy_l365d']
    if 'price' in paris_clean.columns:
        paris_clean['revenue'] = paris_clean['price']
    if 'review_scores_rating' in paris_clean.columns:
        paris_clean['sentiment_score'] = paris_clean['review_scores_rating']
    
    # Préparation spécialisée pour Seattle
    seattle_clean = df_seattle.copy()
    if 'price' in seattle_clean.columns:
        seattle_clean['revenue'] = seattle_clean['price']
    if 'review_scores_rating' in seattle_clean.columns:
        seattle_clean['sentiment_score'] = seattle_clean['review_scores_rating']
    
    print(f"Paris: {paris_clean.shape[0]:,} propriétés")
    print(f"Seattle: {seattle_clean.shape[0]:,} propriétés")
    
    return paris_clean, seattle_clean

def create_city_features(df_city, city_name):
    """
    Créer des features spécialisées par ville
    """
    print(f"\nFeatures spécialisées pour {city_name}...")
    
    # Features de base communes
    features = ['revenue', 'sentiment_score', 'occupancy_rate', 'nb_amenities', 
               'latitude', 'longitude', 'month', 'day_of_week']
    
    # Sélectionner features disponibles
    available_features = [f for f in features if f in df_city.columns]
    
    # Features spécialisées par ville
    if city_name == "Paris":
        # Features spécifiques Paris
        paris_patterns = ['review_scores_', 'host_', 'neighbourhood_']
        for col in df_city.columns:
            if any(pattern in col for pattern in paris_patterns) and col not in available_features:
                available_features.append(col)
                if len(available_features) >= 30:  # Limiter à 30 features
                    break
    
    elif city_name == "Seattle":
        # Features spécifiques Seattle
        seattle_patterns = ['review_scores_', 'property_', 'room_type']
        for col in df_city.columns:
            if any(pattern in col for pattern in seattle_patterns) and col not in available_features:
                available_features.append(col)
                if len(available_features) >= 25:  # Limiter à 25 features
                    break
    
    # Nettoyer et préparer les features
    df_features = df_city[available_features].copy()
    
    # Variables cibles
    y_regression = df_features['revenue'] if 'revenue' in df_features.columns else None
    y_classification = None
    
    if 'sentiment_score' in df_features.columns:
        if city_name == "Paris":
            # Seuil adapté à Paris (échelle 0-5)
            threshold = df_features['sentiment_score'].quantile(0.75)
        else:
            # Seuil adapté à Seattle (échelle 0-100) 
            threshold = df_features['sentiment_score'].quantile(0.70)
        
        y_classification = (df_features['sentiment_score'] >= threshold).astype(int)
    
    # Préparer X (exclure targets)
    exclude_cols = ['revenue', 'sentiment_score']
    X_cols = [col for col in df_features.columns if col not in exclude_cols]
    X = df_features[X_cols].copy()
    
    # Nettoyage des types
    for col in X.columns:
        if X[col].dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna('Unknown'))
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Preprocessing
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)
    
    print(f"  Features finales: {X_scaled.shape[1]}")
    print(f"  Échantillons: {len(X_scaled):,}")
    
    return X_scaled, y_regression, y_classification, imputer, scaler

def train_city_model(X, y, city_name, model_type='regression'):
    """
    Entraîner un modèle spécialisé pour une ville
    """
    print(f"\nEntraînement modèle {model_type} pour {city_name}...")
    
    if y is None or y.isnull().all():
        print(f"  Pas de données pour {model_type} - {city_name}")
        return None, None
    
    # Nettoyer les données
    mask = ~y.isnull()
    X_clean = X[mask]
    y_clean = y[mask]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42,
        stratify=y_clean if model_type == 'classification' and y_clean.nunique() > 1 else None
    )
    
    # Modèle optimisé par ville
    if model_type == 'regression':
        # Hyperparamètres optimisés par taille de dataset
        if city_name == "Paris":
            # Plus de données = modèle plus complexe
            model = lgb.LGBMRegressor(
                n_estimators=600,
                max_depth=12,
                learning_rate=0.05,
                num_leaves=100,
                random_state=42,
                verbose=-1
            )
        else:  # Seattle
            # Moins de données = modèle plus simple pour éviter l'overfitting
            model = lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                num_leaves=50,
                random_state=42,
                verbose=-1
            )
    else:  # classification
        from sklearn.ensemble import RandomForestClassifier
        if city_name == "Paris":
            model = RandomForestClassifier(
                n_estimators=400,
                max_depth=15,
                class_weight='balanced',
                random_state=42
            )
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            )
    
    # Entraînement
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Métriques
    if model_type == 'regression':
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  R²: {r2:.3f}")
        print(f"  CV R²: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        metrics = {'rmse': rmse, 'r2': r2, 'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()}
    
    else:  # classification
        accuracy = model.score(X_test, y_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  CV Accuracy: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        metrics = {'accuracy': accuracy, 'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()}
    
    return model, metrics

def compare_unified_vs_specialized():
    """
    Comparaison entre modèle unifié et modèles spécialisés
    """
    print("\nCOMPARAISON MODÈLE UNIFIÉ VS SPÉCIALISÉS")
    print("="*55)
    
    # Charger données
    paris_data, seattle_data = load_city_data()
    
    results = {}
    
    # Modèles spécialisés pour chaque ville
    for city_data, city_name in [(paris_data, "Paris"), (seattle_data, "Seattle")]:
        print(f"\n--- {city_name.upper()} ---")
        
        # Préparer features
        X, y_reg, y_clf, imputer, scaler = create_city_features(city_data, city_name)
        
        # Modèle de régression
        reg_model, reg_metrics = train_city_model(X, y_reg, city_name, 'regression')
        
        # Modèle de classification
        clf_model, clf_metrics = train_city_model(X, y_clf, city_name, 'classification')
        
        # Sauvegarder les modèles spécialisés
        if reg_model:
            joblib.dump(reg_model, f"{MODEL_DIR}/{city_name.lower()}_price_model.pkl")
            joblib.dump(scaler, f"{MODEL_DIR}/{city_name.lower()}_scaler.pkl")
            joblib.dump(imputer, f"{MODEL_DIR}/{city_name.lower()}_imputer.pkl")
        
        results[city_name] = {
            'regression': reg_metrics,
            'classification': clf_metrics,
            'samples': len(X)
        }
    
    # Comparaison avec le modèle unifié (Jour 5)
    unified_r2_paris = 0.212  # Performance globale du modèle unifié
    unified_r2_seattle = 0.212
    
    print(f"\n{'='*60}")
    print("COMPARAISON DES PERFORMANCES")
    print(f"{'='*60}")
    
    for city in ["Paris", "Seattle"]:
        if city in results and results[city]['regression']:
            specialized_r2 = results[city]['regression']['r2']
            unified_r2 = unified_r2_paris if city == "Paris" else unified_r2_seattle
            
            improvement = ((specialized_r2 - unified_r2) / abs(unified_r2)) * 100
            
            print(f"\n{city}:")
            print(f"  Modèle unifié:     R² = {unified_r2:.3f}")
            print(f"  Modèle spécialisé: R² = {specialized_r2:.3f}")
            print(f"  Amélioration:      {improvement:+.1f}%")
            
            if improvement > 20:
                print(f"  Statut: ✅ AMÉLIORATION SIGNIFICATIVE")
            elif improvement > 0:
                print(f"  Statut: 🔶 AMÉLIORATION MODÉRÉE")
            else:
                print(f"  Statut: ❌ DÉGRADATION")
    
    return results

def create_geographic_features():
    """
    PRIORITÉ 2 - Features géographiques avancées
    """
    print("\nFEATURES GÉOGRAPHIQUES AVANCÉES")
    print("="*40)
    
    paris_data, seattle_data = load_city_data()
    
    for city_data, city_name in [(paris_data, "Paris"), (seattle_data, "Seattle")]:
        print(f"\n{city_name} - Features géographiques:")
        
        if 'latitude' in city_data.columns and 'longitude' in city_data.columns:
            # Distance au centre-ville
            if city_name == "Paris":
                center_lat, center_lon = 48.8566, 2.3522  # Notre-Dame
            else:
                center_lat, center_lon = 47.6062, -122.3321  # Space Needle
            
            city_data['distance_center'] = np.sqrt(
                (city_data['latitude'] - center_lat)**2 + 
                (city_data['longitude'] - center_lon)**2
            ) * 111  # Approximation km
            
            # Densité locale (nombre de propriétés dans un rayon)
            from sklearn.neighbors import NearestNeighbors
            coords = city_data[['latitude', 'longitude']].dropna()
            if len(coords) > 100:
                nn = NearestNeighbors(n_neighbors=50, radius=0.01)
                nn.fit(coords)
                distances, indices = nn.kneighbors(coords)
                city_data.loc[coords.index, 'local_density'] = np.mean(distances, axis=1)
            
            print(f"  Distance centre: {city_data['distance_center'].mean():.2f} km moyenne")
            print(f"  Densité locale: calculée pour {len(coords)} propriétés")
        else:
            print(f"  Pas de coordonnées géographiques disponibles")

def temporal_validation():
    """
    PRIORITÉ 3 - Validation temporelle
    """
    print("\nVALIDATION TEMPORELLE")
    print("="*25)
    
    paris_data, seattle_data = load_city_data()
    
    for city_data, city_name in [(paris_data, "Paris"), (seattle_data, "Seattle")]:
        print(f"\n{city_name} - Analyse temporelle:")
        
        if 'month' in city_data.columns and 'revenue' in city_data.columns:
            # Performance par mois
            monthly_stats = city_data.groupby('month')['revenue'].agg(['mean', 'std', 'count'])
            
            print("  Prix moyen par mois:")
            for month, stats in monthly_stats.iterrows():
                if stats['count'] > 10:  # Au moins 10 observations
                    print(f"    Mois {month}: ${stats['mean']:.0f} ± ${stats['std']:.0f} ({stats['count']} prop.)")
            
            # Coefficient de variation temporelle
            cv_temporal = monthly_stats['std'].mean() / monthly_stats['mean'].mean()
            print(f"  Variabilité temporelle: {cv_temporal:.3f}")
            
            if cv_temporal > 0.3:
                print(f"  ⚠️  Forte variabilité saisonnière détectée")
            else:
                print(f"  ✅ Variabilité saisonnière modérée")

def main():
    """
    Workflow principal Jour 6
    """
    print("🚀 JOUR 6 - OPTIMISATIONS AVANCÉES")
    print("="*50)
    
    try:
        # Priorité 1: Modèles spécialisés par ville
        specialized_results = compare_unified_vs_specialized()
        
        # Priorité 2: Features géographiques
        create_geographic_features()
        
        # Priorité 3: Validation temporelle
        temporal_validation()
        
        print(f"\n🏆 JOUR 6 TERMINÉ")
        print("="*30)
        print(f"📁 Modèles spécialisés: {MODEL_DIR}")
        print(f"📊 Résultats: {OUTPUT_DIR}")
        
        return specialized_results
        
    except Exception as e:
        print(f"\n❌ Erreur Jour 6: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()