from dotenv import load_dotenv
load_dotenv()

import json, asyncio
import streamlit as st
from mcpuse_agent import make_client, make_agent, run_once, register_client_for_cleanup

st.set_page_config(page_title="Ollama + MCP Use Agent", layout="wide")
st.title("Ollama + MCP Use • Agent Chat")

with st.sidebar:
    st.header("Configuration")
    cfg_path = st.text_input("Config MCP (browser_mcp.json)", "browser_mcp.json")
    force_tools = st.toggle("Forcer l'usage d'outils", value=True)
    max_steps = st.number_input("Max steps", 4, 60, 24, 1)
    st.caption("Node/npm requis pour npx. Ollama doit être en cours d'exécution (port 11434)." )

if "mcp_use_client" not in st.session_state:
    st.session_state.mcp_use_client = None
if "mcp_use_agent" not in st.session_state:
    st.session_state.mcp_use_agent = None
if "history" not in st.session_state:
    st.session_state.history = []

colA, colB = st.columns(2)
with colA:
    if st.button("Initialiser l'agent", type="primary"):
        try:
            client = make_client(cfg_path)
            agent = make_agent(client, require_tools=force_tools, max_steps=int(max_steps))
            st.session_state.mcp_use_client = client
            st.session_state.mcp_use_agent = agent
            register_client_for_cleanup(client)
            st.success("Agent prêt.")
        except Exception as e:
            st.error(f"Init échouée: {e}")

with colB:
    if st.button("Arrêter les serveurs"):
        try:
            c = st.session_state.get("mcp_use_client")
            if c and getattr(c, "sessions", None):
                asyncio.run(c.close_all_sessions())
            st.session_state.mcp_use_client = None
            st.session_state.mcp_use_agent = None
            st.info("Arrêté.")
        except Exception as e:
            st.error(f"Arrêt échoué: {e}")

st.markdown("---")
st.caption("Prompt de test qui déclenche un outil :")
st.code('''Liste les outils disponibles, puis appelle duckduckgo-search.search avec {"query":"tendances prix immobilier Paris 10e 2025","max_results":5} et synthétise en 5 puces avec sources.''')

prompt = st.text_area("Message", "Trouve 3 sources récentes sur les tendances des prix dans le 10e arrondissement de Paris et résume.")

if st.button("Lancer la requête", type="primary"):
    ag = st.session_state.get("mcp_use_agent")
    if not ag:
        st.warning("Initialise d'abord l'agent dans la barre latérale.")
    else:
        with st.spinner("Réflexion de l'agent…"):
            try:
                body, trace = run_once(ag, prompt)
                st.session_state.history.append(("user", prompt))
                st.session_state.history.append(("assistant", body))
                st.subheader("Réponse")
                st.markdown(body)
                if trace:
                    st.subheader("Tool trace")
                    st.code(json.dumps(trace, indent=2, ensure_ascii=False))
            except Exception as e:
                st.error(str(e))

# History display
if st.session_state.history:
    st.markdown("---")
    st.subheader("Historique")
    for role, content in st.session_state.history[-10:]:
        with st.chat_message(role):
            st.markdown(content)
