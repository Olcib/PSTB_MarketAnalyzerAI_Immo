# gemini_agent.py
import os, json, time
from typing import Dict, Any, List, Optional, Tuple, Any as _Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent


# ───────────── Config
def _get_gemini_key() -> Optional[str]:
    return os.getenv("GOOGLE_API_KEY")


def _get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


# ───────────── Tools
def _build_tools(
    mcp_tools: List[Dict[str, Any]],
    mcp_call_fn,
    trace: List[Dict[str, Any]],
):
    """
    Expose ONLY schema-safe tools:
      - mcp_call(server: str, tool: str, args_json: str) -> str(JSON)
      - mcp_catalog() -> str
    """

    def _mcp_call(server: str, tool: str, args_json: str) -> str:
        started = time.time()
        norm_tool = (tool or "")
        if "__" in norm_tool:
            norm_tool = norm_tool.split("__", 1)[-1]

        # ---- arguments: corriger Airbnb (location) si l'appel fournit query
        args = {}
        if args_json:
            try:
                args = json.loads(args_json)
            except Exception as e:
                raise ValueError(f"args_json invalid JSON: {e}")
        if server == "airbnb" and norm_tool == "airbnb_search":
            if "location" not in args and "query" in args:
                args["location"] = args.pop("query")

        try:
            res = mcp_call_fn(server, norm_tool, args) or {}
            ok = True
        except Exception as e:
            res = {"error": str(e), "called_tool": norm_tool}
            ok = False

        # ---- FLATTEN: texte + JSON → 'joined_text' et stats simples
        flat_texts: List[str] = []
        listings = []
        contents = res.get("content") if isinstance(res, dict) else None
        if isinstance(contents, list):
            for c in contents:
                if isinstance(c, dict):
                    # texte direct
                    if c.get("type") == "text" and c.get("text"):
                        flat_texts.append(str(c["text"]).strip())
                    # JSON structuré
                    data = c.get("data")
                    if c.get("type") in ("json", "application/json") or isinstance(data, (dict, list)):
                        try:
                            payload = data if isinstance(data, (dict, list)) else json.loads(c.get("text","{}"))
                            flat_texts.append(json.dumps(payload, ensure_ascii=False)[:4000])
                            # Heuristique Airbnb: extraire prix/nuit si présents
                            if isinstance(payload, dict):
                                items = payload.get("results") or payload.get("listings") or []
                            elif isinstance(payload, list):
                                items = payload
                            else:
                                items = []
                            for it in items:
                                price = it.get("price") or it.get("pricePerNight") or it.get("nightly_price")
                                if isinstance(price, (int, float)):
                                    listings.append(float(price))
                                elif isinstance(price, str):
                                    # extraire chiffres dans "€123"
                                    import re
                                    m = re.search(r"(\d+[.,]?\d*)", price)
                                    if m:
                                        listings.append(float(m.group(1).replace(",", ".")))
                        except Exception:
                            pass

        summary = {}
        if listings:
            import statistics as _st
            summary = {
                "n_listings": len(listings),
                "avg_price": float(round(_st.mean(listings), 2)),
                "median_price": float(round(_st.median(listings), 2)),
                "min_price": float(round(min(listings), 2)),
                "max_price": float(round(max(listings), 2)),
            }

        normalized = {
            "ok": ok,
            "server": server,
            "tool": norm_tool,
            "args": args,
            "joined_text": "\n".join(flat_texts)[:8000] if flat_texts else "",
            "stats": summary,
            "raw": res,
        }

        latency = int((time.time() - started) * 1000)
        trace.append({
            "server": server, "tool": norm_tool, "success": ok,
            "latency_ms": latency, "args": args, "via": "generic"
        })
        return json.dumps(normalized, ensure_ascii=False)

    mcp_call_tool = StructuredTool.from_function(
        name="mcp_call",
        description=(
            "Appelle un outil MCP. Fournis:\n"
            "- server: ex. 'airbnb', 'duckduckgo-search', 'playwright'\n"
            "- tool: nom exact de l’outil (ex. 'search', 'airbnb_search')\n"
            "- args_json: JSON string des arguments (ex: '{\"query\":\"paris\"}')\n"
            "La réponse JSON contient 'joined_text' (extrait texte agrégé) et 'raw' (résultat brut)."
        ),
        func=_mcp_call,
    )

    # Catalog helper (for the model and debugging)
    cats = []
    for t in mcp_tools:
        server = t.get("server", "srv")
        name = t.get("name", "tool")
        keys = ", ".join((t.get("schema", {}).get("properties") or {}).keys())
        cats.append(f"{server}__{name}({keys})")

    def _mcp_catalog() -> str:
        return "\n".join(cats) if cats else "Aucun outil MCP chargé."

    mcp_catalog_tool = StructuredTool.from_function(
        name="mcp_catalog",
        description="Liste lisible des outils MCP disponibles.",
        func=lambda: _mcp_catalog(),
    )

    return [mcp_call_tool, mcp_catalog_tool]


# ───────────── System/User prompts
def _build_system(mcp_tools: List[Dict[str, Any]], base_system: str) -> str:
    if not base_system:
        base_system = (
            "Tu es un agent immobilier outillé. Utilise mcp_catalog pour voir les outils, "
            "puis mcp_call(server, tool, args_json) pour obtenir des données réelles. "
            "Réponds en français, concis et chiffré. Ne fabrique pas de chiffres."
        )
    cats = []
    for t in mcp_tools:
        server = t.get("server", "srv")
        name = t.get("name", "tool")
        keys = ", ".join((t.get("schema", {}).get("properties") or {}).keys())
        cats.append(f"{server}__{name}({keys})")
    cat_txt = "Outils MCP: " + "; ".join(cats) if cats else "Aucun outil MCP chargé."

    # Key instruction: read joined_text
    how_to_read = (
        "Quand tu reçois la réponse de mcp_call, LIS le champ 'joined_text' et synthétise des chiffres et sources à partir de ce texte. "
        "Si 'joined_text' est vide, inspecte 'raw'. N'invente rien."
    )

    # Procedure to force tool use
    procedure = (
        "Procédure:\n"
        "1) Appelle mcp_catalog.\n"
        "2) Choisis un outil pertinent.\n"
        "3) Appelle mcp_call(server, tool, args_json) au moins une fois avant toute conclusion.\n"
        "4) Résume en puces chiffrées avec 2–3 sources."
    )

    return f"{base_system}\n{cat_txt}\n{how_to_read}\n{procedure}"


def _build_user(user_text: str) -> str:
    return (
        f"Tâche: {user_text}\n"
        "IMPORTANT: N'écris pas de réponse finale avant d'avoir utilisé mcp_call."
    )


# ───────────── Public entrypoint (Streamlit uses this)
def run_gemini_with_mcp(
    user_messages: List[Dict[str, str]],
    mcp_tools: List[Dict[str, Any]],
    mcp_call_fn,
    model_name: Optional[str] = None,
    max_turns: int = 6,  # kept for signature; LangGraph handles the loop
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    user_messages: [{"role":"system"|"user"|"assistant", "content":"..."}]
    mcp_tools: [{"server":, "name":, "description":, "schema":{...}}]
    mcp_call_fn: callable(server:str, tool:str, args:dict) -> dict
    returns: (final_text, trace)
    """
    api_key = _get_gemini_key()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY missing.")
    model_id = model_name or _get_gemini_model()

    trace: List[Dict[str, Any]] = []
    tools = _build_tools(mcp_tools, mcp_call_fn, trace)

    llm = ChatGoogleGenerativeAI(
        model=model_id,
        google_api_key=api_key,
        temperature=0.3,
    )

    base_system = next(
        (m["content"] for m in user_messages if m.get("role") == "system" and m.get("content")), ""
    )
    user_txt = next(
        (m["content"] for m in reversed(user_messages) if m.get("role") == "user" and m.get("content")),
        "Analyse la requête.",
    )

    system_msg = SystemMessage(content=_build_system(mcp_tools, base_system))
    user_msg = HumanMessage(content=_build_user(user_txt))

    agent = create_react_agent(
        llm,
        tools=tools,
        state_modifier=system_msg,
    )

    # Attempt 1
    final: Dict[str, _Any] = agent.invoke({"messages": [user_msg]})
    msgs: List[BaseMessage] = final.get("messages", [])
    used_tools = any(t.get("via") for t in trace)

    # If no tool used, retry once with stronger instruction
    if not used_tools:
        stronger = HumanMessage(
            content=(
                "Tu n'as utilisé aucun outil. Relance:\n"
                "1) Appelle mcp_catalog maintenant.\n"
                "2) Choisis un outil adapté.\n"
                "3) Appelle mcp_call(server, tool, args_json) avec des arguments JSON valides.\n"
                "4) Puis synthétise à partir de 'joined_text'."
            )
        )
        final = agent.invoke({"messages": [user_msg, stronger]})
        msgs = final.get("messages", [])

    text_out = ""
    if msgs:
        last = msgs[-1]
        if isinstance(last, AIMessage):
            if isinstance(last.content, str):
                text_out = last.content
            else:
                # flatten parts if any
                text_out = "".join(
                    [p.get("text", "") if isinstance(p, dict) else str(p) for p in last.content]  # type: ignore
                )
        else:
            text_out = getattr(last, "content", "") or str(last)

    return (text_out.strip(), trace)
