"""
J5 - 18h-20h: Interface Streamlit Chatbot MVP
===========================================
"""

import streamlit as st
import sys
sys.path.append('.')
from j5_rag_foundation import AirbnbRAG

st.set_page_config(
    page_title="MarketAnalyzer AI - Assistant Airbnb",
    page_icon="🏠",
    layout="wide"
)

# Initialize RAG system
@st.cache_resource
def load_rag_system():
    """Charge le système RAG en cache"""
    rag = AirbnbRAG()
    if rag.load_airbnb_data():
        rag.build_faiss_index()
        return rag
    return None

def main():
    st.title("🏠 MarketAnalyzer AI - Assistant Airbnb")
    st.markdown("### Votre copilote IA pour l'analyse du marché locatif")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        tab = st.selectbox("Choisissez une section:", [
            "💬 Chat Assistant",
            "📊 Résultats ML",
            "ℹ️ À propos"
        ])
    
    # Load RAG system
    rag = load_rag_system()
    if not rag:
        st.error("Erreur de chargement des données. Vérifiez les fichiers CSV.")
        return
    
    # Chat Tab
    if tab == "💬 Chat Assistant":
        st.header("Assistant Conversationnel")
        st.markdown("Posez vos questions sur les données Airbnb Paris/Seattle")
        
        # Questions prédéfinies
        st.subheader("Questions suggérées:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Prix moyen Paris?"):
                st.session_state['question'] = "Quel est le prix moyen à Paris?"
        
        with col2:
            if st.button("Performance modèles ML?"):
                st.session_state['question'] = "Comment performent vos modèles ML?"
                
        with col3:
            if st.button("Données Seattle?"):
                st.session_state['question'] = "Combien de logements à Seattle?"
        
        # Zone de saisie
        question = st.text_input(
            "Votre question:", 
            value=st.session_state.get('question', ''),
            placeholder="Ex: Quel est le prix médian à Seattle?"
        )
        
        if st.button("Envoyer") or question:
            if question:
                with st.spinner("Recherche en cours..."):
                    # Récupération documents
                    retrieved_docs = rag.retrieve(question, k=2)
                    
                    # Génération réponse
                    answer = rag.generate_answer(question, retrieved_docs)
                    
                    # Affichage
                    st.success("Réponse:")
                    st.write(answer)
                    
                    # Debug info
                    with st.expander("Détails techniques (debug)"):
                        st.write("Documents récupérés:")
                        for i, doc in enumerate(retrieved_docs):
                            st.write(f"**Document {i+1}** (score: {doc['score']:.3f})")
                            st.write(doc['document'][:200] + "...")
            else:
                st.warning("Veuillez poser une question.")
    
    # ML Results Tab  
    elif tab == "📊 Résultats ML":
        st.header("Résultats Machine Learning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Paris")
            st.metric("R² Final", "0.473", "+1.8%")
            st.metric("RMSE", "$163.64")
            st.metric("Logements", f"{len(rag.paris_data):,}")
            st.metric("Prix moyen", f"{rag.paris_target['price'].mean():.0f}€")
        
        with col2:
            st.subheader("Seattle") 
            st.metric("R² Final", "0.598", "+1.1%")
            st.metric("RMSE", "$59.81")
            st.metric("Logements", f"{len(rag.seattle_data):,}")
            st.metric("Prix moyen", f"{rag.seattle_target['price'].mean():.0f}$")
        
        st.subheader("Architecture Modèles")
        st.write("""
        **Meilleurs modèles:** Weighted Ensemble
        - Paris: LightGBM + CatBoost + XGBoost
        - Seattle: CatBoost + LightGBM + RandomForest
        
        **Feature top:** bedrooms × bathrooms (interaction)
        
        **Pipeline:** Baseline → Optimisation → Feature Engineering → Ensemble
        """)
    
    # About Tab
    elif tab == "ℹ️ À propos":
        st.header("À propos de MarketAnalyzer AI")
        st.write("""
        **Mission:** Fournir aux professionnels de la location un assistant IA 
        capable d'analyser le marché et d'émettre des recommandations data-driven.
        
        **Technologies:**
        - Machine Learning: LightGBM, CatBoost, XGBoost
        - NLP: Sentence Transformers, FAISS
        - Interface: Streamlit
        - Déploiement: Streamlit Cloud
        
        **Données analysées:**
        - Paris: 53,455 logements Airbnb
        - Seattle: 3,818 logements Airbnb
        - Source: Inside Airbnb (données réelles)
        """)

if __name__ == "__main__":
    main()