"""
ACTION 5 - FEATURE ENGINEERING AVANCE (VERSION CORRIGÉE)
========================================================
Création de nouvelles features à partir des données baseline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
import time
import os

def load_baseline_data():
    """Charge les données préparées par le baseline"""
    print("Chargement des données baseline...")
    
    baseline_dir = "results/honest_baseline"
    
    try:
        # Charger train et test séparément puis les recombiner pour feature engineering
        X_train_paris = pd.read_csv(f"{baseline_dir}/X_train_paris.csv")
        X_test_paris = pd.read_csv(f"{baseline_dir}/X_test_paris.csv") 
        y_train_paris = pd.read_csv(f"{baseline_dir}/y_train_paris.csv").values.ravel()
        y_test_paris = pd.read_csv(f"{baseline_dir}/y_test_paris.csv").values.ravel()
        
        X_train_seattle = pd.read_csv(f"{baseline_dir}/X_train_seattle.csv")
        X_test_seattle = pd.read_csv(f"{baseline_dir}/X_test_seattle.csv")
        y_train_seattle = pd.read_csv(f"{baseline_dir}/y_train_seattle.csv").values.ravel()
        y_test_seattle = pd.read_csv(f"{baseline_dir}/y_test_seattle.csv").values.ravel()
        
        # Recombiner train et test avec target pour feature engineering
        paris_full = pd.concat([X_train_paris, X_test_paris], ignore_index=True)
        paris_target = np.concatenate([y_train_paris, y_test_paris])
        paris_full['price'] = paris_target
        
        seattle_full = pd.concat([X_train_seattle, X_test_seattle], ignore_index=True)
        seattle_target = np.concatenate([y_train_seattle, y_test_seattle])
        seattle_full['price'] = seattle_target
        
        print(f"✅ Données chargées avec succès")
        print(f"   Paris: {paris_full.shape[0]:,} échantillons, {paris_full.shape[1]:,} features")
        print(f"   Seattle: {seattle_full.shape[0]:,} échantillons, {seattle_full.shape[1]:,} features")
        
        return paris_full, seattle_full
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        print("Assurez-vous d'avoir exécuté honest_baseline.py avec sauvegarde")
        return None, None

def create_interaction_features(df, top_features):
    """Crée des features d'interaction entre les variables importantes"""
    print("Création des features d'interaction...")
    
    df_new = df.copy()
    
    # Features numériques importantes disponibles
    numeric_features = [f for f in top_features if f in df.select_dtypes(include=[np.number]).columns]
    
    interaction_count = 0
    # Limiter aux top 6 pour éviter explosion dimensionnelle
    for feat1, feat2 in combinations(numeric_features[:6], 2):
        if feat1 in df.columns and feat2 in df.columns:
            try:
                # Produit
                df_new[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                # Ratio (avec protection division par zéro)
                df_new[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-8)
                interaction_count += 2
            except:
                continue
    
    print(f"   {interaction_count} features d'interaction créées")
    return df_new

def create_polynomial_features(df, numeric_cols):
    """Crée des features polynomiales"""
    print("Création des features polynomiales...")
    
    df_new = df.copy()
    poly_count = 0
    
    # Top 5 features numériques pour éviter explosion
    important_numeric = numeric_cols[:5]
    
    for col in important_numeric:
        if col in df.columns and col != 'price':
            try:
                # Carré
                df_new[f'{col}_squared'] = df[col] ** 2
                poly_count += 1
                
                # Racine carrée (pour valeurs positives)
                if df[col].min() >= 0:
                    df_new[f'{col}_sqrt'] = np.sqrt(df[col] + 1e-8)
                    poly_count += 1
                
                # Log (pour valeurs strictement positives)
                if df[col].min() > 0:
                    df_new[f'{col}_log'] = np.log(df[col] + 1e-8)
                    poly_count += 1
            except:
                continue
    
    print(f"   {poly_count} features polynomiales créées")
    return df_new

def create_binning_features(df, target_col='price'):
    """Crée des features par binning/discrétisation"""
    print("Création des features par binning...")
    
    df_new = df.copy()
    bin_count = 0
    
    # Features continues importantes à binner
    continuous_features = ['accommodates', 'bedrooms', 'bathrooms', 'beds']
    available_features = [f for f in continuous_features if f in df.columns]
    
    for feature in available_features:
        try:
            # Éviter les doublons et valeurs manquantes
            feature_values = df[feature].dropna()
            if len(feature_values.unique()) > 5:  # Seulement si assez de valeurs uniques
                # Quantile binning
                df_new[f'{feature}_bin'] = pd.qcut(df[feature], q=5, labels=False, duplicates='drop')
                bin_count += 1
                
                # Binning basé sur la target si disponible
                if target_col in df.columns:
                    price_by_feature = df.groupby(feature)[target_col].mean()
                    df_new[f'{feature}_price_bin'] = pd.cut(
                        df[feature].map(price_by_feature), 
                        bins=5, labels=False, duplicates='drop'
                    )
                    bin_count += 1
        except:
            continue
    
    print(f"   {bin_count} features de binning créées")
    return df_new

def create_clustering_features(df):
    """Crée des features basées sur le clustering"""
    print("Création des features de clustering...")
    
    df_new = df.copy()
    cluster_count = 0
    
    # Features pour clustering (seulement numériques disponibles)
    potential_features = ['accommodates', 'bedrooms', 'bathrooms', 'beds', 'longitude', 'latitude']
    clustering_features = [f for f in potential_features if f in df.columns]
    
    if len(clustering_features) >= 3:
        try:
            # Préparer les données
            X_cluster = df[clustering_features].fillna(df[clustering_features].mean())
            
            # Standardiser
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)
            
            # K-means clustering avec différents nombres de clusters
            for n_clusters in [3, 5]:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df_new[f'cluster_{n_clusters}'] = kmeans.fit_predict(X_scaled)
                
                # Distance au centroïde le plus proche
                distances = kmeans.transform(X_scaled)
                df_new[f'distance_to_cluster_{n_clusters}'] = np.min(distances, axis=1)
                cluster_count += 2
        except Exception as e:
            print(f"   ⚠️ Erreur clustering: {e}")
    
    print(f"   {cluster_count} features de clustering créées")
    return df_new

def create_ratio_features(df):
    """Crée des features de ratios métier"""
    print("Création des features de ratios métier...")
    
    df_new = df.copy()
    ratio_count = 0
    
    try:
        # Ratios pertinents pour Airbnb
        if 'bedrooms' in df.columns and 'accommodates' in df.columns:
            df_new['people_per_bedroom_v2'] = df['accommodates'] / (df['bedrooms'] + 1e-8)
            ratio_count += 1
        
        if 'bathrooms' in df.columns and 'accommodates' in df.columns:
            df_new['people_per_bathroom'] = df['accommodates'] / (df['bathrooms'] + 1e-8)
            ratio_count += 1
        
        if 'beds' in df.columns and 'accommodates' in df.columns:
            df_new['people_per_bed'] = df['accommodates'] / (df['beds'] + 1e-8)
            ratio_count += 1
        
        # Ratios de densité si disponibles
        if all(col in df.columns for col in ['bedrooms', 'bathrooms', 'beds']):
            df_new['bed_bath_ratio'] = df['beds'] / (df['bathrooms'] + 1e-8)
            df_new['room_density'] = (df['bedrooms'] + df['bathrooms']) / (df['accommodates'] + 1e-8)
            ratio_count += 2
            
    except Exception as e:
        print(f"   ⚠️ Erreur ratios: {e}")
    
    print(f"   {ratio_count} features de ratios créées")
    return df_new

def advanced_feature_selection(X, y, max_features=25):
    """Sélection avancée des features"""
    print(f"Sélection avancée des {max_features} meilleures features...")
    
    # Nettoyer les données
    X_clean = X.select_dtypes(include=[np.number]).fillna(0)
    
    # 1. RandomForest feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_clean, y)
    
    feature_importance = pd.DataFrame({
        'feature': X_clean.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 2. Sélection des top features
    selected_features = feature_importance.head(max_features)['feature'].tolist()
    
    print(f"   Features sélectionnées: {len(selected_features)}")
    print("   Top 10 features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"     {i:2d}. {row['feature']}: {row['importance']:.4f}")
    
    return selected_features, feature_importance

def evaluate_enhanced_features(df, city_name):
    """Évalue l'impact du feature engineering"""
    print(f"\nÉVALUATION FEATURE ENGINEERING - {city_name.upper()}")
    print("="*60)
    
    # Séparer features et target
    target_col = 'price'
    if target_col not in df.columns:
        print("❌ Colonne price non trouvée")
        return None
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Sélection des features
    selected_features, feature_importance = advanced_feature_selection(X, y, max_features=25)
    X_selected = X[selected_features]
    
    print(f"Features après engineering: {X.shape[1]} → {len(selected_features)} sélectionnées")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    # Test des modèles avec nouvelles features
    models = {
        'LightGBM_Enhanced': LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, verbose=-1
        ),
        'CatBoost_Enhanced': CatBoostRegressor(
            iterations=200, depth=6, learning_rate=0.1,
            random_seed=42, verbose=False
        ),
        'RandomForest_Enhanced': RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        )
    }
    
    results = {}
    baseline_performance = {'paris': 0.470, 'seattle': 0.597}  # De l'Action 4
    
    for name, model in models.items():
        print(f"\nEntraînement {name}...")
        
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
        
        print(f"   R²: {r2:.3f} (baseline {baseline_r2:.3f})")
        print(f"   RMSE: ${rmse:.2f}")
        print(f"   CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        print(f"   Amélioration: {improvement:+.1f}%")
        
        if improvement > 0:
            print("   ✅ Amélioration!")
        else:
            print("   ⚠️ Pas d'amélioration")
    
    return {
        'results': results,
        'feature_importance': feature_importance,
        'selected_features': selected_features,
        'data_shapes': {
            'original': X.shape[1],
            'selected': len(selected_features),
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0]
        }
    }

def create_feature_analysis_plots(results_paris, results_seattle, output_dir):
    """Crée les graphiques d'analyse des features"""
    print("\n📊 Création graphiques d'analyse...")
    
    if results_paris is None or results_seattle is None:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top features Paris
    top_paris = results_paris['feature_importance'].head(15)
    ax1.barh(range(len(top_paris)), top_paris['importance'])
    ax1.set_yticks(range(len(top_paris)))
    ax1.set_yticklabels(top_paris['feature'], fontsize=10)
    ax1.set_title('Top 15 Features - Paris')
    ax1.set_xlabel('Importance')
    
    # Top features Seattle
    top_seattle = results_seattle['feature_importance'].head(15)
    ax2.barh(range(len(top_seattle)), top_seattle['importance'])
    ax2.set_yticks(range(len(top_seattle)))
    ax2.set_yticklabels(top_seattle['feature'], fontsize=10)
    ax2.set_title('Top 15 Features - Seattle')
    ax2.set_xlabel('Importance')
    
    # Comparaison performances modèles
    models = list(results_paris['results'].keys())
    paris_r2 = [results_paris['results'][m]['r2'] for m in models]
    seattle_r2 = [results_seattle['results'][m]['r2'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax3.bar(x - width/2, paris_r2, width, label='Paris', alpha=0.8)
    ax3.bar(x + width/2, seattle_r2, width, label='Seattle', alpha=0.8)
    ax3.set_xlabel('Modèles Enhanced')
    ax3.set_ylabel('R²')
    ax3.set_title('Performances avec Feature Engineering')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Amélioration vs baseline
    baseline_r2 = {'paris': 0.470, 'seattle': 0.597}
    paris_improvement = [(r2 - baseline_r2['paris']) * 100 for r2 in paris_r2]
    seattle_improvement = [(r2 - baseline_r2['seattle']) * 100 for r2 in seattle_r2]
    
    ax4.bar(x - width/2, paris_improvement, width, label='Paris', alpha=0.8)
    ax4.bar(x + width/2, seattle_improvement, width, label='Seattle', alpha=0.8)
    ax4.set_xlabel('Modèles')
    ax4.set_ylabel('Amélioration R² (%)')
    ax4.set_title('Amélioration vs Baseline Action 4')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_engineering_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Graphiques sauvegardés: {output_dir}/feature_engineering_analysis.png")

def generate_feature_engineering_report(results_paris, results_seattle, output_dir, duration):
    """Génère le rapport du feature engineering"""
    print("\n📝 Génération rapport feature engineering...")
    
    if results_paris is None or results_seattle is None:
        print("❌ Pas de résultats à rapporter")
        return
    
    report_path = f"{output_dir}/feature_engineering_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 🔧 RAPPORT FEATURE ENGINEERING AVANCÉ\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Durée d'exécution**: {duration:.1f} secondes\n\n")
        
        # Résultats par ville
        baseline_performance = {'paris': 0.470, 'seattle': 0.597}
        
        for city, results in [('Paris', results_paris), ('Seattle', results_seattle)]:
            f.write(f"## 📊 Résultats {city.upper()}\n\n")
            
            shapes = results['data_shapes']
            f.write(f"**Transformation des données:**\n")
            f.write(f"- Features originales: {shapes['original']}\n")
            f.write(f"- Features sélectionnées: {shapes['selected']}\n")
            f.write(f"- Échantillons train: {shapes['train_samples']:,}\n")
            f.write(f"- Échantillons test: {shapes['test_samples']:,}\n\n")
            
            f.write("**Performances des modèles:**\n")
            f.write("| Modèle | R² | RMSE | CV Score | Amélioration |\n")
            f.write("|--------|----|----- |----------|-------------|\n")
            
            baseline_r2 = baseline_performance[city.lower()]
            
            for model_name in sorted(results['results'].keys(), key=lambda x: results['results'][x]['r2'], reverse=True):
                result = results['results'][model_name]
                improvement = (result['r2'] - baseline_r2) * 100
                
                f.write(f"| {model_name} | {result['r2']:.3f} | ${result['rmse']:.2f} | ")
                f.write(f"{result['cv_mean']:.3f}±{result['cv_std']:.3f} | ")
                if improvement > 0:
                    f.write(f"**+{improvement:.1f}%** ✅ |\n")
                else:
                    f.write(f"{improvement:.1f}% ❌ |\n")
            
            f.write("\n**Top 10 Features:**\n")
            for i, (_, row) in enumerate(results['feature_importance'].head(10).iterrows(), 1):
                f.write(f"{i}. **{row['feature']}**: {row['importance']:.4f}\n")
            f.write("\n")
        
        # Analyse globale
        f.write("## 🎯 ANALYSE GLOBALE\n\n")
        
        # Meilleurs modèles
        best_paris = max(results_paris['results'].items(), key=lambda x: x[1]['r2'])
        best_seattle = max(results_seattle['results'].items(), key=lambda x: x[1]['r2'])
        
        f.write(f"### 🏆 Meilleurs modèles après Feature Engineering:\n")
        f.write(f"- **Paris**: {best_paris[0]} (R² = {best_paris[1]['r2']:.3f})\n")
        f.write(f"- **Seattle**: {best_seattle[0]} (R² = {best_seattle[1]['r2']:.3f})\n\n")
        
        # Améliorations
        paris_improvement = (best_paris[1]['r2'] - baseline_performance['paris']) * 100
        seattle_improvement = (best_seattle[1]['r2'] - baseline_performance['seattle']) * 100
        
        f.write("### 📈 Impact du Feature Engineering:\n")
        f.write(f"- **Paris**: {paris_improvement:+.2f}% d'amélioration\n")
        f.write(f"- **Seattle**: {seattle_improvement:+.2f}% d'amélioration\n\n")
        
        if paris_improvement > 0 and seattle_improvement > 0:
            f.write("✅ **Succès**: Feature engineering bénéfique pour les deux villes\n\n")
        elif paris_improvement > 0 or seattle_improvement > 0:
            f.write("⚠️ **Mitigé**: Amélioration sur une ville seulement\n\n")
        else:
            f.write("❌ **Limité**: Peu d'amélioration, modèles baseline déjà optimaux\n\n")
        
        # Recommandations
        f.write("## 🚀 RECOMMANDATIONS\n\n")
        f.write("### Prochaines étapes:\n")
        f.write("1. **Ensemble Methods**: Combiner les meilleurs modèles\n")
        f.write("2. **Feature Engineering ciblé**: Approfondir les interactions prometteuses\n")
        f.write("3. **Données externes**: Intégrer météo, événements, économie\n")
        f.write("4. **Optimisation temporelle**: Features saisonnières\n\n")
        
        f.write("### Pour la production:\n")
        if paris_improvement > 2 or seattle_improvement > 2:
            f.write("- **Adopter** les modèles enhanced pour production\n")
        else:
            f.write("- **Maintenir** modèles baseline en production\n")
            f.write("- **Continuer** recherche de nouvelles features\n")
        
        f.write("- **Monitoring** continu des performances\n")
        f.write("- **A/B Testing** pour validation\n")
    
    print(f"   Rapport sauvegardé: {report_path}")

def main():
    print("=" * 60)
    print("ACTION 5 - FEATURE ENGINEERING AVANCÉ")
    print("=" * 60)
    
    start_time = time.time()
    
    # Créer dossier de résultats
    output_dir = "results/feature_engineering"
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger les données
    paris_df, seattle_df = load_baseline_data()
    if paris_df is None or seattle_df is None:
        return
    
    # Features importantes du baseline (basées sur Action 3 et 4)
    top_baseline_features = [
        'accommodates', 'bedrooms', 'bathrooms', 'beds', 
        'longitude', 'availability_30'
    ]
    
    results_summary = {}
    
    # Traiter chaque ville
    for city_name, df in [('Paris', paris_df), ('Seattle', seattle_df)]:
        print(f"\n{'='*60}")
        print(f"FEATURE ENGINEERING - {city_name.upper()}")
        print(f"{'='*60}")
        
        print(f"Données initiales: {df.shape}")
        
        # Feature engineering progressif
        df_enhanced = df.copy()
        
        # 1. Features d'interaction
        df_enhanced = create_interaction_features(df_enhanced, top_baseline_features)
        print(f"Après interactions: {df_enhanced.shape}")
        
        # 2. Features polynomiales
        numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns.tolist()
        df_enhanced = create_polynomial_features(df_enhanced, numeric_cols)
        print(f"Après polynomiales: {df_enhanced.shape}")
        
        # 3. Features de binning
        df_enhanced = create_binning_features(df_enhanced)
        print(f"Après binning: {df_enhanced.shape}")
        
        # 4. Features de clustering
        df_enhanced = create_clustering_features(df_enhanced)
        print(f"Après clustering: {df_enhanced.shape}")
        
        # 5. Features de ratios métier
        df_enhanced = create_ratio_features(df_enhanced)
        print(f"Après ratios: {df_enhanced.shape}")
        
        # Évaluation
        results = evaluate_enhanced_features(df_enhanced, city_name)
        if results:
            results_summary[city_name] = results
    
    # Créer graphiques si on a des résultats
    if 'Paris' in results_summary and 'Seattle' in results_summary:
        create_feature_analysis_plots(
            results_summary['Paris'], 
            results_summary['Seattle'], 
            output_dir
        )
    
    # Générer rapport
    duration = time.time() - start_time
    generate_feature_engineering_report(
        results_summary.get('Paris'), 
        results_summary.get('Seattle'), 
        output_dir, 
        duration
    )
    
    # Résumé final
    print(f"\n{'='*60}")
    print("ACTION 5 TERMINÉE")
    print(f"{'='*60}")
    print(f"Durée: {duration:.1f} secondes")
    
    if results_summary:
        baseline_performance = {'paris': 0.470, 'seattle': 0.597}
        
        for city_name, summary in results_summary.items():
            best_model = max(summary['results'].keys(), key=lambda x: summary['results'][x]['r2'])
            best_r2 = summary['results'][best_model]['r2']
            baseline_r2 = baseline_performance[city_name.lower()]
            improvement = (best_r2 - baseline_r2) * 100
            
            print(f"\n{city_name.upper()}:")
            print(f"  Features: {summary['data_shapes']['original']} → {summary['data_shapes']['selected']}")
            print(f"  Meilleur: {best_model}")
            print(f"  R²: {best_r2:.3f} (baseline: {baseline_r2:.3f})")
            print(f"  Amélioration: {improvement:+.1f}%")
    
    print(f"\nFichiers sauvegardés dans: {output_dir}/")
    print("Prêt pour Action 6 - Ensemble Methods")

if __name__ == "__main__":
    main()