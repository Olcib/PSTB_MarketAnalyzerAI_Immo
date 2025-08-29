import streamlit as st

st.set_page_config(page_title="MarketAnalyzerAI Immo", layout="wide")
st.title("MarketAnalyzerAI Immo Strategic AI Co-Pilot")

tabs = st.tabs(["EDA", "Forecasts", "Chatbot (RAG)", "What-if Simulator"])

with tabs[0]:
    st.subheader("Exploratory Data Analysis")
    st.info("Add plots and dataset summaries here.")

with tabs[1]:
    st.subheader("ML/DL Forecasts")
    st.info("Show model predictions and KPIs (occupancy, revenue).")

with tabs[2]:
    st.subheader("RAG Chatbot")
    st.info("Ask domain questions grounded in your documents.")

with tabs[3]:
    st.subheader("What-if Simulator")
    price = st.slider("Price (â‚¬/night)", 30, 300, 120, 5)
    promo = st.slider("Promotion (%)", 0, 50, 10, 1)
    amenities = st.multiselect("Added amenities", ["A/C", "Washer", "Dryer", "Dishwasher", "High-speed WiFi"])
    st.success(f"Simulated: price={price}, promo={promo}%, amenities={amenities}")
