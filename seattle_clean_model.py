import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def load_seattle_clean():
    """
    Charge Seattle avec exclusion stricte des fuites
    """
    print("MODÈLE SEATTLE SANS FUITES DE DONNÉES")
    print("="*45)
    
    df_seattle = pd.read_csv("data/processed/enriched_seattle_fixed.csv")
    
    # Variable cible
    if 'price' in df_seattle.columns:
        df_seattle['revenue'] = df_seattle['price']
    
    print(f"Dataset Seattle: {df_seattle.shape}")
    
    return df_seattle

def select_safe_features(df):
    """
    Sélection stricte des features sans fuites
    """
    print("\nSÉLECTION DES FEATURES SÛRES")
    print("="*35)
    
    # Features INTERDITES (fuites potentielles)
    forbidden_patterns = [
        'price', 'revenue', 'cost', 'amount', 'dollar', 'euro', 
        'weekly_price', 'monthly_price', 'security_deposit', 
        'cleaning_fee', 'extra_people'
    ]
    
    # Features AUTORISÉES (caractéristiques intrinsèques)
    safe_base_features = [
        'nb_amenities', 'accommodates', 'bedrooms', 'beds', 'bathrooms',
        'latitude', 'longitude', 'month', 'day_of_week',
        'minimum_nights', 'maximum_nights', 'availability_365',
        'number_of_reviews', 'reviews_per_month'
    ]
    
    # Features de review scores (moyennes, pas individuelles)
    review_patterns = [
        'review_scores_rating', 'review_scores_accuracy', 
        'review_scores_cleanliness', 'review_scores_checkin',
        'review_scores_communication', 'review_scores_location',
        'review_scores_value'
    ]
    
    # Construire liste des features sûres
    safe_features = []
    
    # Features de base
    for feature in safe_base_features:
        if feature in df.columns:
            safe_features.append(feature)
    
    # Features de review (moyennes seulement, pas de lag/diff/rolling qui pourraient créer des fuites)
    for pattern in review_patterns:
        if pattern in df.columns:
            safe_features.append(pattern)
    
    # Vérifier qu'aucune feature interdite n'est incluse
    final_features = []
    for feature in safe_features:
        is_forbidden = any(forbidden in feature.lower() for forbidden in forbidden_patterns)
        if not is_forbidden:
            final_features.append(feature)
        else:
            print(f"  EXCLU (fuite potentielle): {feature}")
    
    print(f"\nFeatures sûres sélectionnées: {len(final_features)}")
    for i, feature in enumerate(final_features, 1):
        print(f"  {i:2d}. {feature}")
    
    return final_features

def prepare_clean_data(df, safe_features):
    """
    Préparation des données sans fuites
    """
    print(f"\nPRÉPARATION DES DONNÉES PROPRES")
    print("="*40)
    
    # Sélectionner features + target
    feature_cols = [f for f in safe_features if f in df.columns]
    
    if 'revenue' not in df.columns:
        print("Variable cible 'revenue' manquante")
        return None, None
    
    X = df[feature_cols].copy()
    y = df['revenue'].copy()
    
    print(f"Features finales: {len(feature_cols)}")
    print(f"Échantillons: {len(X):,}")
    
    # Nettoyage des features
    for col in X.columns:
        if X[col].dtype == 'object':
            # Encoder les variables catégorielles
            try:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str).fillna('Unknown'))
                print(f"  Encodé: {col}")
            except:
                X[col] = 0
        else:
            # Variables numériques
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Vérifier les corrélations avec le prix (détection finale)
    print(f"\nVÉRIFICATION FINALE DES CORRÉLATIONS:")
    suspicious_count = 0
    for col in X.columns:
        if X[col].notna().sum() > 100:
            try:
                corr = X[col].corr(y)
                if abs(corr) > 0.9:
                    print(f"  ALERTE: {col} corrélation = {corr:.3f}")
                    suspicious_count += 1
                elif abs(corr) > 0.7:
                    print(f"  Élevée: {col} corrélation = {corr:.3f}")
            except:
                pass
    
    if suspicious_count == 0:
        print("  ✅ Aucune corrélation suspecte détectée")
    else:
        print(f"  ⚠️ {suspicious_count} corrélations suspectes restantes")
    
    # Imputation
    imputer = SimpleImputer(strategy='median')
    X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_clean), columns=X_clean.columns)
    
    return X_scaled, y

def train_clean_seattle_model(X, y):
    """
    Entraînement du modèle Seattle sans fuites
    """
    print(f"\nENTRAÎNEMENT MODÈLE SEATTLE PROPRE")
    print("="*40)
    
    # Nettoyer les données
    mask = ~y.isnull()
    X_clean = X[mask]
    y_clean = y[mask]
    
    print(f"Données d'entraînement: {len(X_clean):,} échantillons")
    
    # Split avec une proportion plus grande pour le test (données limitées)
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.3, random_state=42
    )
    
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Modèle conservateur pour éviter l'overfitting
    model = lgb.LGBMRegressor(
        n_estimators=100,      # Peu d'arbres
        max_depth=5,           # Profondeur limitée
        learning_rate=0.1,     # Learning rate standard
        num_leaves=20,         # Peu de feuilles
        min_child_samples=20,  # Minimum d'échantillons par feuille
        subsample=0.8,         # Subsampling pour régularisation
        random_state=42,
        verbose=-1
    )
    
    # Entraînement
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Métriques
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Validation croisée
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')  # 3-fold seulement
    
    print(f"\nPERFORMANCE MODÈLE PROPRE:")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  R²: {r2:.3f}")
    print(f"  CV R²: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
    
    # Comparaison avec modèle contaminé
    print(f"\nCOMPARAISON:")
    print(f"  Modèle contaminé (Jour 6): R² = 0.974")
    print(f"  Modèle propre (Actuel):    R² = {r2:.3f}")
    
    performance_drop = ((0.974 - r2) / 0.974) * 100
    print(f"  Chute de performance: -{performance_drop:.1f}%")
    
    if performance_drop > 70:
        print("  ✅ CONFIRMATION: Fuites de données éliminées")
    else:
        print("  ⚠️ Performance encore élevée - investigation supplémentaire requise")
    
    # Feature importance
    print(f"\nIMPORTANCE DES FEATURES (Top 10):")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']}: {row['importance']:.3f}")
    
    return model, r2, rmse

def main():
    """
    Reconstruction complète du modèle Seattle
    """
    print("🔧 RECONSTRUCTION MODÈLE SEATTLE SANS FUITES")
    print("="*60)
    
    try:
        # Charger données
        df_seattle = load_seattle_clean()
        
        # Sélectionner features sûres
        safe_features = select_safe_features(df_seattle)
        
        if len(safe_features) < 5:
            print("Pas assez de features sûres disponibles")
            return None
        
        # Préparer données
        X, y = prepare_clean_data(df_seattle, safe_features)
        
        if X is None:
            print("Échec de la préparation des données")
            return None
        
        # Entraîner modèle propre
        model, r2, rmse = train_clean_seattle_model(X, y)
        
        print(f"\n🏁 RECONSTRUCTION TERMINÉE")
        print("="*35)
        
        if r2 > 0.3:
            print(f"✅ Performance légitime: R² = {r2:.3f}")
            print("   Modèle utilisable en production")
        elif r2 > 0.1:
            print(f"🔶 Performance modérée: R² = {r2:.3f}")
            print("   Amélioration possible avec plus de features")
        else:
            print(f"❌ Performance faible: R² = {r2:.3f}")
            print("   Données insuffisantes pour Seattle")
        
        return model, r2, rmse
        
    except Exception as e:
        print(f"\n❌ Erreur durant la reconstruction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()