"""
ACTION 5 - FEATURE ENGINEERING AVANCE
=====================================
Création de nouvelles features et optimisation avancée
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
import time
import os

def load_clean_data():
    """Charge les données nettoyées"""
    print("Chargement des données nettoyées...")
    
    try:
        paris_df = pd.read_csv("data/paris_clean.csv")
        seattle_df = pd.read_csv("data/seattle_clean.csv")
        print(f"Paris: {paris_df.shape[0]:,} échantillons")
        print(f"Seattle: {seattle_df.shape[0]:,} échantillons")
        return paris_df, seattle_df
    except FileNotFoundError:
        print("❌ Fichiers de données nettoyées non trouvés")
        return None, None

def create_interaction_features(df, top_features):
    """Crée des features d'interaction entre les variables importantes"""
    print("Création des features d'interaction...")
    
    df_new = df.copy()
    
    # Interactions entre features numériques importantes
    numeric_features = [f for f in top_features if f in df.select_dtypes(include=[np.number]).columns]
    
    interaction_count = 0
    for feat1, feat2 in combinations(numeric_features[:8], 2):  # Top 8 pour limiter
        if feat1 in df.columns and feat2 in df.columns:
            # Produit
            df_new[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
            # Ratio (avec protection division par zéro)
            df_new[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-8)
            interaction_count += 2
    
    print(f"   {interaction_count} features d'interaction créées")
    return df_new

def create_polynomial_features(df, numeric_cols, degree=2):
    """Crée des features polynomiales"""
    print("Création des features polynomiales...")
    
    df_new = df.copy()
    
    # Sélectionner les top features numériques pour éviter l'explosion dimensionnelle
    important_numeric = numeric_cols[:6]  # Top 6 seulement
    
    for col in important_numeric:
        if col in df.columns:
            # Carré
            df_new[f'{col}_squared'] = df[col] ** 2
            # Racine carrée (pour valeurs positives)
            if df[col].min() >= 0:
                df_new[f'{col}_sqrt'] = np.sqrt(df[col] + 1e-8)
            # Log (pour valeurs positives)
            if df[col].min() > 0:
                df_new[f'{col}_log'] = np.log(df[col] + 1e-8)
    
    print(f"   {len(important_numeric) * 3} features polynomiales créées")
    return df_new

def create_binning_features(df, target_col='price'):
    """Crée des features par binning/discrétisation"""
    print("Création des features par binning...")
    
    df_new = df.copy()
    
    # Binning pour features continues importantes
    continuous_features = ['accommodates', 'bedrooms', 'bathrooms', 'beds']
    
    for feature in continuous_features:
        if feature in df.columns:
            # Quantile binning
            df_new[f'{feature}_bin'] = pd.qcut(df[feature], q=5, labels=False, duplicates='drop')
            
            # Binning basé sur la target (price)
            if target_col in df.columns:
                # Grouper par feature et calculer la moyenne de prix
                price_by_feature = df.groupby(feature)[target_col].mean()
                # Créer des bins basés sur les quantiles de prix
                df_new[f'{feature}_price_bin'] = pd.cut(
                    df[feature].map(price_by_feature), 
                    bins=5, labels=False, duplicates='drop'
                )
    
    print(f"   {len(continuous_features) * 2} features de binning créées")
    return df_new

def create_clustering_features(df):
    """Crée des features basées sur le clustering"""
    print("Création des features de clustering...")
    
    df_new = df.copy()
    
    # Sélectionner features pour clustering
    clustering_features = ['accommodates', 'bedrooms', 'bathrooms', 'beds', 'latitude', 'longitude']
    available_features = [f for f in clustering_features if f in df.columns]
    
    if len(available_features) >= 3:
        # Préparer les données
        X_cluster = df[available_features].fillna(df[available_features].mean())
        
        # Standardiser
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        # K-means clustering
        for n_clusters in [3, 5, 8]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_new[f'cluster_{n_clusters}'] = kmeans.fit_predict(X_scaled)
            
            # Distance au centroïde
            distances = kmeans.transform(X_scaled)
            df_new[f'distance_to_cluster_{n_clusters}'] = np.min(distances, axis=1)
    
    print(f"   6 features de clustering créées")
    return df_new

def create_statistical_features(df, group_cols=['neighbourhood_cleansed', 'property_type']):
    """Crée des features statistiques par groupe"""
    print("Création des features statistiques...")
    
    df_new = df.copy()
    target_col = 'price'
    
    if target_col not in df.columns:
        print("   ⚠️ Colonne price non trouvée, skip features statistiques")
        return df_new
    
    for group_col in group_cols:
        if group_col in df.columns:
            # Statistiques par groupe
            group_stats = df.groupby(group_col)[target_col].agg([
                'mean', 'median', 'std', 'min', 'max', 'count'
            ]).reset_index()
            
            # Renommer les colonnes
            group_stats.columns = [group_col] + [f'{group_col}_{stat}' for stat in ['mean', 'median', 'std', 'min', 'max', 'count']]
            
            # Merger avec le dataframe principal
            df_new = df_new.merge(group_stats, on=group_col, how='left')
            
            # Features relatives (écart à la moyenne du groupe)
            df_new[f'{group_col}_price_vs_mean'] = df_new[target_col] - df_new[f'{group_col}_mean']
            df_new[f'{group_col}_price_vs_median'] = df_new[target_col] - df_new[f'{group_col}_median']
    
    print(f"   {len(group_cols) * 8} features statistiques créées")
    return df_new

def create_temporal_features(df):
    """Crée des features temporelles si disponibles"""
    print("Création des features temporelles...")
    
    df_new = df.copy()
    
    # Si on a des colonnes de dates
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'last' in col.lower()]
    
    features_created = 0
    for date_col in date_columns:
        if date_col in df.columns:
            try:
                df_new[date_col] = pd.to_datetime(df_new[date_col], errors='coerce')
                
                # Extraire composantes temporelles
                df_new[f'{date_col}_year'] = df_new[date_col].dt.year
                df_new[f'{date_col}_month'] = df_new[date_col].dt.month
                df_new[f'{date_col}_dayofweek'] = df_new[date_col].dt.dayofweek
                df_new[f'{date_col}_quarter'] = df_new[date_col].dt.quarter
                
                # Temps depuis la date
                df_new[f'days_since_{date_col}'] = (pd.Timestamp.now() - df_new[date_col]).dt.days
                
                features_created += 5
            except:
                continue
    
    print(f"   {features_created} features temporelles créées")
    return df_new

def advanced_feature_selection(X, y, max_features=50):
    """Sélection avancée des features"""
    print(f"Sélection avancée des {max_features} meilleures features...")
    
    # 1. Filter method: SelectKBest
    selector_f = SelectKBest(score_func=f_regression, k=min(max_features*2, X.shape[1]))
    X_selected_f = selector_f.fit_transform(X, y)
    selected_features_f = X.columns[selector_f.get_support()].tolist()
    
    # 2. Wrapper method: RFE avec RandomForest
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    selector_rfe = RFE(estimator=rf, n_features_to_select=max_features, step=0.1)
    X_selected_rfe = selector_rfe.fit_transform(X, y)
    selected_features_rfe = X.columns[selector_rfe.get_support()].tolist()
    
    # 3. Embedded method: RandomForest feature importance
    rf.fit(X, y)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    selected_features_emb = feature_importance.head(max_features)['feature'].tolist()
    
    # Combinaison des méthodes (intersection + top features)
    # Intersection des 3 méthodes
    intersection = set(selected_features_f) & set(selected_features_rfe) & set(selected_features_emb)
    
    # Compléter avec les top features par importance
    final_features = list(intersection)
    remaining_slots = max_features - len(final_features)
    
    if remaining_slots > 0:
        additional_features = feature_importance[
            ~feature_importance['feature'].isin(final_features)
        ].head(remaining_slots)['feature'].tolist()
        final_features.extend(additional_features)
    
    print(f"   Features sélectionnées: {len(final_features)}")
    print(f"   Intersection des 3 méthodes: {len(intersection)}")
    
    return final_features, feature_importance

def prepare_advanced_data(df, city_name, top_baseline_features):
    """Prépare les données avec feature engineering avancé"""
    print(f"\n{'='*50}")
    print(f"FEATURE ENGINEERING AVANCÉ - {city_name.upper()}")
    print(f"{'='*50}")
    
    print(f"Données initiales: {df.shape}")
    
    # 1. Features d'interaction
    df = create_interaction_features(df, top_baseline_features)
    print(f"Après interactions: {df.shape}")
    
    # 2. Features polynomiales
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = create_polynomial_features(df, numeric_cols)
    print(f"Après polynomiales: {df.shape}")
    
    # 3. Features de binning
    df = create_binning_features(df)
    print(f"Après binning: {df.shape}")
    
    # 4. Features de clustering
    df = create_clustering_features(df)
    print(f"Après clustering: {df.shape}")
    
    # 5. Features statistiques
    df = create_statistical_features(df)
    print(f"Après statistiques: {df.shape}")
    
    # 6. Features temporelles
    df = create_temporal_features(df)
    print(f"Après temporelles: {df.shape}")
    
    return df

def evaluate_feature_engineering(X_train, X_test, y_train, y_test, city_name, baseline_performance):
    """Évalue l'impact du feature engineering"""
    print(f"\nÉVALUATION FEATURE ENGINEERING - {city_name.upper()}")
    print("="*50)
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=2000, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Entraînement {name}...")
        
        # Entraînement
        model.fit(X_train, y_train)
        
        # Prédiction
        y_pred = model.predict(X_test)
        
        # Métriques
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        
        results[name] = {
            'r2': r2,
            'rmse': rmse,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model': model
        }
        
        # Comparaison avec baseline
        baseline_r2 = baseline_performance[city_name.lower()]
        improvement = (r2 - baseline_r2) * 100
        
        print(f"   R²: {r2:.3f} (vs baseline {baseline_r2:.3f})")
        print(f"   RMSE: ${rmse:.2f}")
        print(f"   CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        print(f"   Amélioration: {improvement:+.1f}%")
        
        if improvement > 0:
            print("   ✅ Amélioration!")
        else:
            print("   ⚠️ Pas d'amélioration")
    
    return results

def create_feature_analysis_plots(feature_importance_paris, feature_importance_seattle, output_dir):
    """Crée les graphiques d'analyse des features"""
    print("Création graphiques d'analyse des features...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top features Paris
    top_paris = feature_importance_paris.head(15)
    ax1.barh(range(len(top_paris)), top_paris['importance'])
    ax1.set_yticks(range(len(top_paris)))
    ax1.set_yticklabels(top_paris['feature'])
    ax1.set_title('Top 15 Features - Paris')
    ax1.set_xlabel('Importance')
    
    # Top features Seattle
    top_seattle = feature_importance_seattle.head(15)
    ax2.barh(range(len(top_seattle)), top_seattle['importance'])
    ax2.set_yticks(range(len(top_seattle)))
    ax2.set_yticklabels(top_seattle['feature'])
    ax2.set_title('Top 15 Features - Seattle')
    ax2.set_xlabel('Importance')
    
    # Distribution importance Paris
    ax3.hist(feature_importance_paris['importance'], bins=30, alpha=0.7, edgecolor='black')
    ax3.set_title('Distribution Importance Features - Paris')
    ax3.set_xlabel('Importance')
    ax3.set_ylabel('Fréquence')
    
    # Distribution importance Seattle
    ax4.hist(feature_importance_seattle['importance'], bins=30, alpha=0.7, edgecolor='black')
    ax4.set_title('Distribution Importance Features - Seattle')
    ax4.set_xlabel('Importance')
    ax4.set_ylabel('Fréquence')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 60)
    print("ACTION 5 - FEATURE ENGINEERING AVANCÉ")
    print("=" * 60)
    
    start_time = time.time()
    
    # Créer dossier de résultats
    output_dir = "results/feature_engineering"
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger les données
    paris_df, seattle_df = load_clean_data()
    if paris_df is None:
        return
    
    # Features importantes du baseline (à récupérer du rapport précédent)
    top_baseline_features = [
        'accommodates', 'bedrooms', 'bathrooms', 'beds', 
        'people_per_bedroom', 'estimated_months_active',
        'availability_30', 'availability_365', 'longitude', 'number_of_reviews'
    ]
    
    # Performance baseline pour comparaison
    baseline_performance = {'paris': 0.455, 'seattle': 0.587}
    
    results_summary = {}
    
    # Traiter chaque ville
    for city_name, df in [('Paris', paris_df), ('Seattle', seattle_df)]:
        # Feature engineering
        df_enhanced = prepare_advanced_data(df, city_name, top_baseline_features)
        
        # Préparation pour modélisation
        target_col = 'price'
        if target_col not in df_enhanced.columns:
            print(f"❌ Colonne {target_col} non trouvée pour {city_name}")
            continue
        
        # Séparer features et target
        X = df_enhanced.drop(columns=[target_col])
        y = df_enhanced[target_col]
        
        # Gérer les valeurs manquantes et non-numériques
        X = X.select_dtypes(include=[np.number]).fillna(0)
        
        print(f"\nFeatures finales pour {city_name}: {X.shape[1]}")
        
        # Sélection avancée des features
        selected_features, feature_importance = advanced_feature_selection(X, y, max_features=30)
        X_selected = X[selected_features]
        
        # Split train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42
        )
        
        print(f"Split final - Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
        
        # Évaluation
        results = evaluate_feature_engineering(
            X_train, X_test, y_train, y_test, 
            city_name, baseline_performance
        )
        
        results_summary[city_name] = {
            'results': results,
            'feature_importance': feature_importance,
            'n_features_original': df.shape[1],
            'n_features_enhanced': df_enhanced.shape[1],
            'n_features_selected': len(selected_features)
        }
        
        # Sauvegarder les features sélectionnées
        selected_features_df = pd.DataFrame({
            'feature': selected_features,
            'rank': range(1, len(selected_features) + 1)
        })
        selected_features_df.to_csv(
            f"{output_dir}/selected_features_{city_name.lower()}.csv", 
            index=False
        )
    
    # Créer graphiques d'analyse
    if 'Paris' in results_summary and 'Seattle' in results_summary:
        create_feature_analysis_plots(
            results_summary['Paris']['feature_importance'],
            results_summary['Seattle']['feature_importance'],
            output_dir
        )
    
    # Rapport final
    duration = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("RÉSUMÉ FEATURE ENGINEERING")
    print(f"{'='*60}")
    print(f"Durée: {duration:.1f} secondes")
    
    for city_name, summary in results_summary.items():
        print(f"\n{city_name.upper()}:")
        print(f"  Features: {summary['n_features_original']} → {summary['n_features_enhanced']} → {summary['n_features_selected']}")
        
        best_model = max(summary['results'].keys(), 
                        key=lambda x: summary['results'][x]['r2'])
        best_r2 = summary['results'][best_model]['r2']
        baseline_r2 = baseline_performance[city_name.lower()]
        improvement = (best_r2 - baseline_r2) * 100
        
        print(f"  Meilleur modèle: {best_model}")
        print(f"  R²: {best_r2:.3f} (baseline: {baseline_r2:.3f})")
        print(f"  Amélioration: {improvement:+.1f}%")
    
    print(f"\nFichiers sauvegardés dans: {output_dir}/")
    print("Prêt pour Action 6 - Ensemble Methods")

if __name__ == "__main__":
    main()