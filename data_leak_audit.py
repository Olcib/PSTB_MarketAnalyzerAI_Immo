import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "results/data_audit/"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots/", exist_ok=True)

def load_seattle_data():
    """
    Charge les données Seattle pour l'audit
    """
    print("AUDIT DES FUITES DE DONNÉES - SEATTLE")
    print("="*50)
    
    df_seattle = pd.read_csv("data/processed/enriched_seattle_fixed.csv")
    
    # Préparation identique au Jour 6
    seattle_clean = df_seattle.copy()
    if 'price' in seattle_clean.columns:
        seattle_clean['revenue'] = seattle_clean['price']
    if 'review_scores_rating' in seattle_clean.columns:
        seattle_clean['sentiment_score'] = seattle_clean['review_scores_rating']
    
    print(f"Seattle dataset: {seattle_clean.shape[0]:,} lignes × {seattle_clean.shape[1]} colonnes")
    
    return seattle_clean

def identify_suspicious_features(df):
    """
    Identifie les features suspectes corrélées au prix
    """
    print("\nIDENTIFICATION DES FEATURES SUSPECTES")
    print("="*45)
    
    if 'revenue' not in df.columns:
        print("Variable cible 'revenue' manquante")
        return None, None
    
    # Sélectionner features numériques seulement
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'revenue' in numeric_features:
        numeric_features.remove('revenue')
    
    print(f"Analyse de {len(numeric_features)} features numériques")
    
    # Calculer corrélations avec le prix
    correlations = {}
    for feature in numeric_features:
        if df[feature].notna().sum() > 100:  # Au moins 100 valeurs non-nulles
            corr = df[feature].corr(df['revenue'])
            if not np.isnan(corr):
                correlations[feature] = abs(corr)
    
    # Trier par corrélation décroissante
    sorted_correlations = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
    
    print("\nTop 20 corrélations avec le prix:")
    suspicious_features = []
    
    for i, (feature, corr) in enumerate(list(sorted_correlations.items())[:20], 1):
        status = "🚨" if corr > 0.95 else "⚠️" if corr > 0.8 else "📊"
        print(f"  {i:2d}. {feature}: {corr:.3f} {status}")
        
        if corr > 0.9:  # Corrélation suspecte
            suspicious_features.append(feature)
    
    print(f"\nFeatures SUSPECTES (corrélation > 0.9): {len(suspicious_features)}")
    for feature in suspicious_features:
        print(f"  🚨 {feature}: {sorted_correlations[feature]:.3f}")
    
    return sorted_correlations, suspicious_features

def analyze_feature_distributions(df, suspicious_features):
    """
    Analyse la distribution des features suspectes
    """
    print(f"\nANALYSE DES DISTRIBUTIONS SUSPECTES")
    print("="*40)
    
    if not suspicious_features:
        print("Aucune feature suspecte à analyser")
        return
    
    # Analyser les 5 features les plus suspectes
    top_suspicious = suspicious_features[:5]
    
    for feature in top_suspicious:
        print(f"\n--- {feature} ---")
        
        feature_data = df[feature].dropna()
        price_data = df.loc[feature_data.index, 'revenue']
        
        print(f"  Valeurs uniques: {feature_data.nunique():,}")
        print(f"  Min: {feature_data.min():.2f}")
        print(f"  Max: {feature_data.max():.2f}")
        print(f"  Moyenne: {feature_data.mean():.2f}")
        print(f"  Médiane: {feature_data.median():.2f}")
        
        # Vérifier si c'est une transformation directe du prix
        ratio_with_price = feature_data / price_data
        if ratio_with_price.std() < 0.01:  # Ratio quasi-constant
            print(f"  🚨 FUITE DÉTECTÉE: Ratio constant avec prix ({ratio_with_price.mean():.3f})")
        
        # Vérifier les valeurs identiques
        identical_values = (feature_data == price_data).sum()
        if identical_values > len(feature_data) * 0.8:
            print(f"  🚨 FUITE DÉTECTÉE: {identical_values} valeurs identiques au prix")

def check_feature_construction(df):
    """
    Vérifie la construction des features pour identifier les fuites
    """
    print(f"\nVÉRIFICATION CONSTRUCTION DES FEATURES")
    print("="*45)
    
    # Chercher des patterns suspects dans les noms de colonnes
    price_related_patterns = ['price', 'revenue', 'cost', 'amount', 'dollar']
    
    suspicious_names = []
    for col in df.columns:
        col_lower = col.lower()
        for pattern in price_related_patterns:
            if pattern in col_lower and col != 'revenue':
                suspicious_names.append(col)
                break
    
    print(f"Colonnes avec noms suspects: {len(suspicious_names)}")
    for col in suspicious_names:
        if col in df.columns and df[col].notna().sum() > 0:
            corr_with_price = df[col].corr(df['revenue']) if 'revenue' in df.columns else 0
            print(f"  📋 {col}: corrélation = {corr_with_price:.3f}")
    
    # Analyser les features de rolling/lag pour détecter des fuites temporelles
    temporal_features = [col for col in df.columns if any(pattern in col.lower() 
                        for pattern in ['rolling', 'lag', 'shift'])]
    
    print(f"\nFeatures temporelles: {len(temporal_features)}")
    
    temporal_leaks = []
    for col in temporal_features[:10]:  # Analyser les 10 premières
        if col in df.columns and df[col].notna().sum() > 100:
            corr = df[col].corr(df['revenue']) if 'revenue' in df.columns else 0
            if abs(corr) > 0.8:
                temporal_leaks.append((col, corr))
    
    if temporal_leaks:
        print("Features temporelles suspectes:")
        for col, corr in temporal_leaks:
            print(f"  ⏰ {col}: {corr:.3f}")
    else:
        print("Aucune fuite temporelle détectée")

def create_correlation_heatmap(df, top_features=20):
    """
    Crée une heatmap des corrélations
    """
    print(f"\nCRÉATION HEATMAP DES CORRÉLATIONS")
    print("="*40)
    
    # Sélectionner les features numériques les plus corrélées
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if 'revenue' not in numeric_cols:
        print("Pas de variable revenue pour la heatmap")
        return
    
    # Calculer corrélations avec revenue
    correlations_with_revenue = df[numeric_cols].corrwith(df['revenue']).abs().sort_values(ascending=False)
    
    # Prendre les top features
    top_corr_features = correlations_with_revenue.head(top_features).index.tolist()
    
    if len(top_corr_features) > 1:
        # Matrice de corrélation
        correlation_matrix = df[top_corr_features].corr()
        
        # Créer la heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   fmt='.2f',
                   cmap='RdYlBu_r',
                   center=0,
                   square=True,
                   cbar_kws={"shrink": .8})
        
        plt.title(f'Heatmap Corrélations - Top {top_features} Features vs Revenue')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/plots/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Heatmap sauvegardée: {OUTPUT_DIR}/plots/correlation_heatmap.png")
    else:
        print("Pas assez de features pour créer une heatmap")

def simulate_clean_model(df, clean_features_only=True):
    """
    Simule un modèle sans fuites de données
    """
    print(f"\nSIMULATION MODÈLE PROPRE")
    print("="*30)
    
    if clean_features_only:
        # Features de base seulement (sans fuites potentielles)
        safe_features = [
            'occupancy_rate', 'nb_amenities', 'latitude', 'longitude', 
            'month', 'day_of_week'
        ]
        
        # Vérifier disponibilité
        available_safe = [f for f in safe_features if f in df.columns]
        print(f"Features sûres disponibles: {available_safe}")
        
        if len(available_safe) < 3:
            print("Pas assez de features sûres pour l'évaluation")
            return None
        
        # Préparer les données
        X = df[available_safe].copy()
        y = df['revenue'] if 'revenue' in df.columns else None
        
        if y is None:
            print("Variable cible manquante")
            return None
        
        # Nettoyage
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str).fillna('Unknown'))
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Imputation et normalisation
        imputer = SimpleImputer(strategy='median')
        X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_clean), columns=X_clean.columns)
        
        # Split et test rapide
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
        import lightgbm as lgb
        
        # Nettoyer les données
        mask = ~y.isnull()
        X_final = X_scaled[mask]
        y_final = y[mask]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final, test_size=0.3, random_state=42
        )
        
        # Modèle simple
        model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Performance modèle PROPRE:")
        print(f"  Features utilisées: {len(available_safe)}")
        print(f"  R²: {r2:.3f}")
        print(f"  RMSE: ${rmse:.2f}")
        
        # Comparaison avec modèle suspect
        print(f"\nComparaison:")
        print(f"  Modèle suspect (Jour 6): R² = 0.974")
        print(f"  Modèle propre (Audit): R² = {r2:.3f}")
        
        performance_drop = ((0.974 - r2) / 0.974) * 100
        print(f"  Chute de performance: -{performance_drop:.1f}%")
        
        if r2 < 0.5:
            print("  🔍 CONFIRMATION: La performance suspecte était due à des fuites")
        else:
            print("  ⚠️ Performance propre reste élevée - investigation nécessaire")
        
        return r2, rmse
    
def main():
    """
    Audit principal des fuites de données
    """
    print("🔍 AUDIT PRIORITÉ 1 - FUITES DE DONNÉES SEATTLE")
    print("="*60)
    
    try:
        # Charger les données Seattle
        df_seattle = load_seattle_data()
        
        # Identifier les features suspectes
        correlations, suspicious_features = identify_suspicious_features(df_seattle)
        
        if suspicious_features:
            # Analyser les distributions
            analyze_feature_distributions(df_seattle, suspicious_features)
        
        # Vérifier la construction des features
        check_feature_construction(df_seattle)
        
        # Créer la heatmap
        create_correlation_heatmap(df_seattle)
        
        # Tester un modèle propre
        clean_performance = simulate_clean_model(df_seattle)
        
        print(f"\n🏁 AUDIT TERMINÉ")
        print("="*25)
        print(f"📁 Résultats: {OUTPUT_DIR}")
        
        if suspicious_features and len(suspicious_features) > 3:
            print("🚨 FUITE DE DONNÉES CONFIRMÉE")
            print("   Action requise: Reconstruction du modèle Seattle")
        else:
            print("🔍 Pas de fuite évidente détectée")
            print("   Investigation supplémentaire nécessaire")
        
        return correlations, suspicious_features
        
    except Exception as e:
        print(f"\n❌ Erreur durant l'audit: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    correlations, suspicious = main()