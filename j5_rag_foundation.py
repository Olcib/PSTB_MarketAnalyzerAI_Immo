"""
J5 APRÈS-MIDI - Foundation RAG Pipeline
======================================
Code à exécuter aujourd'hui 16h-20h
"""

# 1. Setup environnement
"""
pip install sentence-transformers faiss-cpu streamlit pandas numpy
pip install textblob  # pour sentiment basique
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
from pathlib import Path
import json

class AirbnbRAG:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.documents = []
        self.metadata = []
        
    def load_airbnb_data(self):
        """Charge vos données ML existantes"""
        try:
            # Adapter aux chemins de vos fichiers
            self.paris_data = pd.read_csv("results/honest_baseline/X_train_paris.csv")
            self.paris_target = pd.read_csv("results/honest_baseline/y_train_paris.csv")
            
            self.seattle_data = pd.read_csv("results/honest_baseline/X_train_seattle.csv") 
            self.seattle_target = pd.read_csv("results/honest_baseline/y_train_seattle.csv")
            
            print(f"Données chargées: Paris {len(self.paris_data)}, Seattle {len(self.seattle_data)}")
            return True
        except Exception as e:
            print(f"Erreur chargement: {e}")
            return False
    
    def create_knowledge_base(self):
        """Crée base de connaissances textuelles pour RAG"""
        documents = []
        metadata = []
        
        # Stats Paris
        paris_stats = {
            'ville': 'Paris',
            'nb_logements': len(self.paris_data),
            'prix_moyen': float(self.paris_target['price'].mean()),
            'prix_median': float(self.paris_target['price'].median()),
            'quartiers_top': self.get_top_areas('paris')
        }
        
        doc_paris = f"""
        Données Airbnb Paris:
        - Nombre de logements analysés: {paris_stats['nb_logements']}
        - Prix moyen par nuit: {paris_stats['prix_moyen']:.0f}€
        - Prix médian: {paris_stats['prix_median']:.0f}€
        - Capacité moyenne: {self.paris_data['accommodates'].mean():.1f} personnes
        - Chambres moyennes: {self.paris_data['bedrooms'].mean():.1f}
        """
        documents.append(doc_paris)
        metadata.append(paris_stats)
        
        # Stats Seattle  
        seattle_stats = {
            'ville': 'Seattle',
            'nb_logements': len(self.seattle_data),
            'prix_moyen': float(self.seattle_target['price'].mean()),
            'prix_median': float(self.seattle_target['price'].median()),
        }
        
        doc_seattle = f"""
        Données Airbnb Seattle:
        - Nombre de logements analysés: {seattle_stats['nb_logements']}
        - Prix moyen par nuit: {seattle_stats['prix_moyen']:.0f}$
        - Prix médian: {seattle_stats['prix_median']:.0f}$
        - Capacité moyenne: {self.seattle_data['accommodates'].mean():.1f} personnes
        - Chambres moyennes: {self.seattle_data['bedrooms'].mean():.1f}
        """
        documents.append(doc_seattle)
        metadata.append(seattle_stats)
        
        # Insights ML
        ml_insights = """
        Résultats modèles Machine Learning:
        - Paris: R² final 0.473 (Weighted Ensemble)
        - Seattle: R² final 0.598 (Weighted Ensemble) 
        - Feature la plus importante: bedrooms × bathrooms
        - Modèles performants: LightGBM, CatBoost, XGBoost
        - Pipeline: Baseline → Optimisation → Feature Engineering → Ensemble
        """
        documents.append(ml_insights)
        metadata.append({'type': 'ml_results'})
        
        self.documents = documents
        self.metadata = metadata
        return documents
    
    def build_faiss_index(self):
        """Construit index vectoriel FAISS"""
        if not self.documents:
            self.create_knowledge_base()
            
        # Embeddings
        embeddings = self.encoder.encode(self.documents)
        
        # Index FAISS
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normaliser pour cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Index FAISS créé: {len(self.documents)} documents, dimension {dimension}")
        return True
    
    def retrieve(self, question, k=2):
        """Récupère documents pertinents"""
        if self.index is None:
            self.build_faiss_index()
            
        # Embedding question
        q_embedding = self.encoder.encode([question])
        faiss.normalize_L2(q_embedding)
        
        # Recherche
        scores, indices = self.index.search(q_embedding.astype('float32'), k)
        
        # Résultats
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid index
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(score),
                    'rank': i + 1
                })
        
        return results
    
    def generate_answer(self, question, retrieved_docs):
        """Génère réponse basée sur documents récupérés (version simple)"""
        if not retrieved_docs:
            return "Désolé, je n'ai pas trouvé d'information pertinente."
        
        # Template réponse simple
        context = "\n".join([doc['document'] for doc in retrieved_docs])
        
        # Logique simple selon mots-clés
        question_lower = question.lower()
        
        if 'prix' in question_lower and 'paris' in question_lower:
            paris_data = next((doc['metadata'] for doc in retrieved_docs if doc['metadata'].get('ville') == 'Paris'), None)
            if paris_data:
                return f"À Paris, le prix moyen est de {paris_data['prix_moyen']:.0f}€ par nuit (médian: {paris_data['prix_median']:.0f}€) sur {paris_data['nb_logements']} logements analysés."
        
        elif 'prix' in question_lower and 'seattle' in question_lower:
            seattle_data = next((doc['metadata'] for doc in retrieved_docs if doc['metadata'].get('ville') == 'Seattle'), None)
            if seattle_data:
                return f"À Seattle, le prix moyen est de {seattle_data['prix_moyen']:.0f}$ par nuit (médian: {seattle_data['prix_median']:.0f}$) sur {seattle_data['nb_logements']} logements analysés."
        
        elif 'modèle' in question_lower or 'ml' in question_lower or 'performance' in question_lower:
            return "Nos modèles ML atteignent R² = 0.473 pour Paris et R² = 0.598 pour Seattle. Les ensembles (Weighted) combinent LightGBM, CatBoost et XGBoost. La feature la plus prédictive est l'interaction bedrooms × bathrooms."
        
        else:
            # Réponse générique
            return f"Voici les informations disponibles:\n{retrieved_docs[0]['document'][:300]}..."
    
    def get_top_areas(self, city):
        """Analyse des quartiers (placeholder)"""
        return "Centre, Quartier historique, Zone business"

# Test basique
if __name__ == "__main__":
    rag = AirbnbRAG()
    if rag.load_airbnb_data():
        rag.build_faiss_index()
        
        # Tests
        questions = [
            "Quel est le prix moyen à Paris?",
            "Comment performent vos modèles ML?",
            "Combien de logements à Seattle?"
        ]
        
        for q in questions:
            print(f"\nQ: {q}")
            docs = rag.retrieve(q)
            answer = rag.generate_answer(q, docs)
            print(f"A: {answer}")