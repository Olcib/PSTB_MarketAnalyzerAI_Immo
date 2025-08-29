"""
ACTION 4 - OPTIMISATION AVANCEE DES MODELES
===========================================
Optimisation des hyperparamètres et techniques avancées
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def load_baseline_data():
    """Charge les données préparées par le baseline"""
    print("Chargement des données préparées...")
    
    # Charger les modèles et données du baseline
    baseline_dir = "results/honest_baseline"
    
    try:
        # Si les données préparées ont été sauvegardées
        X_train_paris = pd.read_csv(f"{baseline_dir}/X_train_paris.csv")
        X_test_paris = pd.read_csv(f"{baseline_dir}/X_test_paris.csv") 
        y_train_paris = pd.read_csv(f"{baseline_dir}/y_train_paris.csv").values.ravel()
        y_test_paris = pd.read_csv(f"{baseline_dir}/y_test_paris.csv").values.ravel()
        
        X_train_seattle = pd.read_csv(f"{baseline_dir}/X_train_seattle.csv")
        X_test_seattle = pd.read_csv(f"{baseline_dir}/X_test_seattle.csv")
        y_train_seattle = pd.read_csv(f"{baseline_dir}/y_train_seattle.csv").values.ravel()
        y_test_seattle = pd.read_csv(f"{baseline_dir}/y_test_seattle.csv").values.ravel()
        
        return {
            'paris': (X_train_paris, X_test_paris, y_train_paris, y_test_paris),
            'seattle': (X_train_seattle, X_test_seattle, y_train_seattle, y_test_seattle)
        }
    except:
        print("❌ Données préparées non trouvées. Veuillez d'abord exécuter honest_baseline.py")
        return None

def optimize_lightgbm(X_train, y_train, X_test, y_test, city_name):
    """Optimise LightGBM avec GridSearch"""
    print(f"\n🔧 Optimisation LightGBM - {city_name.upper()}")
    
    # Paramètres à optimiser
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, -1],
        'learning_rate': [0.05, 0.1, 0.15],
        'min_child_samples': [10, 20, 30],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    lgb_base = LGBMRegressor(random_state=42, verbose=-1)
    
    # RandomizedSearch pour efficacité
    lgb_search = RandomizedSearchCV(
        lgb_base, param_grid, 
        n_iter=50, cv=5, 
        scoring='r2', n_jobs=-1, 
        random_state=42, verbose=1
    )
    
    lgb_search.fit(X_train, y_train)
    
    # Évaluation
    y_pred = lgb_search.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"   Meilleurs paramètres: {lgb_search.best_params_}")
    print(f"   R²: {r2:.3f} | RMSE: ${rmse:.2f}")
    
    return lgb_search.best_estimator_, r2, rmse

def optimize_randomforest(X_train, y_train, X_test, y_test, city_name):
    """Optimise Random Forest"""
    print(f"\n🌳 Optimisation Random Forest - {city_name.upper()}")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    rf_search = RandomizedSearchCV(
        rf_base, param_grid,
        n_iter=30, cv=5,
        scoring='r2', n_jobs=-1,
        random_state=42, verbose=1
    )
    
    rf_search.fit(X_train, y_train)
    
    y_pred = rf_search.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"   Meilleurs paramètres: {rf_search.best_params_}")
    print(f"   R²: {r2:.3f} | RMSE: ${rmse:.2f}")
    
    return rf_search.best_estimator_, r2, rmse

def test_advanced_models(X_train, y_train, X_test, y_test, city_name):
    """Teste des modèles plus avancés"""
    print(f"\n🚀 Test modèles avancés - {city_name.upper()}")
    
    models = {}
    results = {}
    
    # XGBoost
    print("   Testing XGBoost...")
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    results['XGBoost'] = {
        'model': xgb,
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    # CatBoost
    print("   Testing CatBoost...")
    cat = CatBoostRegressor(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        verbose=False
    )
    cat.fit(X_train, y_train)
    y_pred = cat.predict(X_test)
    results['CatBoost'] = {
        'model': cat,
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    # Régularisées (Ridge, Lasso, ElasticNet)
    for name, model in [
        ('Ridge', Ridge(alpha=1.0)),
        ('Lasso', Lasso(alpha=1.0, max_iter=2000)),
        ('ElasticNet', ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=2000))
    ]:
        print(f"   Testing {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'model': model,
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
    
    return results

def create_comparison_plots(results_paris, results_seattle, output_dir):
    """Crée des graphiques de comparaison"""
    print("\n📊 Création graphiques comparatifs...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # R² comparison
    models = list(results_paris.keys())
    r2_paris = [results_paris[m]['r2'] for m in models]
    r2_seattle = [results_seattle[m]['r2'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, r2_paris, width, label='Paris', alpha=0.8)
    ax1.bar(x + width/2, r2_seattle, width, label='Seattle', alpha=0.8)
    ax1.set_xlabel('Modèles')
    ax1.set_ylabel('R²')
    ax1.set_title('Comparaison R² par ville')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RMSE comparison
    rmse_paris = [results_paris[m]['rmse'] for m in models]
    rmse_seattle = [results_seattle[m]['rmse'] for m in models]
    
    ax2.bar(x - width/2, rmse_paris, width, label='Paris', alpha=0.8)
    ax2.bar(x + width/2, rmse_seattle, width, label='Seattle', alpha=0.8)
    ax2.set_xlabel('Modèles')
    ax2.set_ylabel('RMSE ($)')
    ax2.set_title('Comparaison RMSE par ville')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Amélioration vs baseline
    baseline_r2 = {'paris': 0.455, 'seattle': 0.587}  # Du baseline
    improvement_paris = [(r2 - baseline_r2['paris']) * 100 for r2 in r2_paris]
    improvement_seattle = [(r2 - baseline_r2['seattle']) * 100 for r2 in r2_seattle]
    
    ax3.bar(x - width/2, improvement_paris, width, label='Paris', alpha=0.8)
    ax3.bar(x + width/2, improvement_seattle, width, label='Seattle', alpha=0.8)
    ax3.set_xlabel('Modèles')
    ax3.set_ylabel('Amélioration R² (%)')
    ax3.set_title('Amélioration vs Baseline')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Performance ranking
    combined_score = [(r2_paris[i] + r2_seattle[i])/2 for i in range(len(models))]
    sorted_indices = sorted(range(len(models)), key=lambda i: combined_score[i], reverse=True)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_scores = [combined_score[i] for i in sorted_indices]
    
    ax4.barh(range(len(sorted_models)), sorted_scores, alpha=0.8)
    ax4.set_yticks(range(len(sorted_models)))
    ax4.set_yticklabels(sorted_models)
    ax4.set_xlabel('Score R² moyen')
    ax4.set_title('Classement global des modèles')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/advanced_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_optimization_report(results_paris, results_seattle, output_dir, duration):
    """Génère le rapport d'optimisation"""
    print("\n📝 Génération rapport d'optimisation...")
    
    report_path = f"{output_dir}/optimization_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 🚀 RAPPORT D'OPTIMISATION AVANCÉE DES MODÈLES\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Durée d'exécution**: {duration:.1f} secondes\n\n")
        
        # Résultats par ville
        for city, results in [('Paris', results_paris), ('Seattle', results_seattle)]:
            f.write(f"## 📊 Résultats {city.upper()}\n\n")
            f.write("| Modèle | R² | RMSE | Amélioration |\n")
            f.write("|--------|----|----- |-------------|\n")
            
            baseline_r2 = 0.455 if city == 'Paris' else 0.587
            
            for model_name in sorted(results.keys(), key=lambda x: results[x]['r2'], reverse=True):
                r2 = results[model_name]['r2']
                rmse = results[model_name]['rmse']
                improvement = (r2 - baseline_r2) * 100
                
                f.write(f"| {model_name} | {r2:.3f} | ${rmse:.2f} | ")
                if improvement > 0:
                    f.write(f"**+{improvement:.1f}%** ✅ |\n")
                else:
                    f.write(f"{improvement:.1f}% |\n")
            f.write("\n")
        
        # Recommandations
        f.write("## 🎯 RECOMMANDATIONS\n\n")
        
        # Meilleur modèle global
        all_models = set(results_paris.keys()) & set(results_seattle.keys())
        best_global = max(all_models, 
                         key=lambda m: (results_paris[m]['r2'] + results_seattle[m]['r2'])/2)
        
        f.write(f"### Meilleur modèle global: **{best_global}**\n")
        f.write(f"- Paris: R² = {results_paris[best_global]['r2']:.3f}\n")
        f.write(f"- Seattle: R² = {results_seattle[best_global]['r2']:.3f}\n\n")
        
        f.write("### Prochaines étapes suggérées:\n")
        f.write("1. **Feature Engineering avancé**: Création d'interactions, transformations non-linéaires\n")
        f.write("2. **Ensemble Methods**: Stacking, Voting, Blending\n")
        f.write("3. **Optimisation fine**: Bayesian Optimization pour hyperparamètres\n")
        f.write("4. **Analyse des résidus**: Identification des patterns non capturés\n")
        f.write("5. **External Data**: Intégration de données externes (météo, événements)\n\n")
        
        f.write("## 📈 CONCLUSION\n\n")
        f.write("L'optimisation avancée des modèles a permis d'identifier les meilleures configurations ")
        f.write("pour chaque ville. Les résultats montrent les opportunités d'amélioration et guident ")
        f.write("vers les prochaines étapes d'optimisation.\n")
    
    print(f"Rapport sauvegardé: {report_path}")

def main():
    print("=" * 50)
    print("ACTION 4 - OPTIMISATION AVANCEE DES MODELES")
    print("=" * 50)
    
    start_time = time.time()
    
    # Créer dossier de résultats
    output_dir = "results/advanced_optimization"
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger les données
    data = load_baseline_data()
    if data is None:
        return
    
    results_paris = {}
    results_seattle = {}
    
    # Optimisation pour chaque ville
    for city_name, (X_train, X_test, y_train, y_test) in data.items():
        print(f"\n{'='*60}")
        print(f"OPTIMISATION {city_name.upper()}")
        print(f"{'='*60}")
        
        city_results = {}
        
        # Optimisation des modèles existants
        lgb_opt, lgb_r2, lgb_rmse = optimize_lightgbm(X_train, y_train, X_test, y_test, city_name)
        city_results['LightGBM_Optimized'] = {'model': lgb_opt, 'r2': lgb_r2, 'rmse': lgb_rmse}
        
        rf_opt, rf_r2, rf_rmse = optimize_randomforest(X_train, y_train, X_test, y_test, city_name)
        city_results['RandomForest_Optimized'] = {'model': rf_opt, 'r2': rf_r2, 'rmse': rf_rmse}
        
        # Test de nouveaux modèles
        advanced_results = test_advanced_models(X_train, y_train, X_test, y_test, city_name)
        city_results.update(advanced_results)
        
        # Sauvegarder les résultats
        if city_name == 'paris':
            results_paris = city_results
        else:
            results_seattle = city_results
        
        # Afficher le récap
        print(f"\n🏆 RÉSULTATS {city_name.upper()}:")
        for model_name in sorted(city_results.keys(), key=lambda x: city_results[x]['r2'], reverse=True):
            r2 = city_results[model_name]['r2']
            rmse = city_results[model_name]['rmse']
            print(f"   {model_name:20} | R²: {r2:.3f} | RMSE: ${rmse:.2f}")
    
    # Créer les graphiques comparatifs
    create_comparison_plots(results_paris, results_seattle, output_dir)
    
    # Générer le rapport
    duration = time.time() - start_time
    generate_optimization_report(results_paris, results_seattle, output_dir, duration)
    
    print(f"\n{'='*60}")
    print("ACTION 4 TERMINÉE")
    print(f"{'='*60}")
    print(f"Durée: {duration:.1f} secondes")
    
    # Meilleur modèle global
    all_models = set(results_paris.keys()) & set(results_seattle.keys())
    if all_models:
        best_global = max(all_models, 
                         key=lambda m: (results_paris[m]['r2'] + results_seattle[m]['r2'])/2)
        print(f"Meilleur modèle global: {best_global}")
        print(f"Paris: R² = {results_paris[best_global]['r2']:.3f}")
        print(f"Seattle: R² = {results_seattle[best_global]['r2']:.3f}")
    
    print("\nPrêt pour Action 5 - Feature Engineering avancé")

if __name__ == "__main__":
    main()