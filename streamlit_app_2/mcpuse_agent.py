# mcpuse_agent.py (Ollama + MCP Use • finalized)
import os, re, json, asyncio, atexit
from typing import Any, Dict, Optional, Tuple, List

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from mcp_use import MCPAgent, MCPClient

load_dotenv()

# Force tool use via prompt (controlled by make_agent)
_FORCE_TOOL_USE: bool = True

OUTPUT_GUIDE_BASE = (
"""
Tu es un agent MCP. Si la tâche implique web/listings/recherche, **appelle au moins un outil**.
N'AFFICHE PAS "Thought", "Action", "Observation". Fournis uniquement les sections ci-dessous.

## TL;DR
- 2–5 puces clés

## Réponse
Texte clair, sections si utile, chiffres et sources si obtenus via outils.

## Comment j'ai travaillé
- Outils utilisés: `server.tool` + une ligne

```tool_trace
{"tools_used":[]}
```
"""
).strip()

_TRACE_RE = re.compile(r"```tool_trace\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)


def _extract_trace_and_body(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Extract optional tool_trace JSON and return (body_without_block, trace_dict|None)."""
    m = _TRACE_RE.search(text or "")
    if not m:
        return text, None
    try:
        js = json.loads(m.group(1))
    except Exception:
        js = None
    body = text.replace(m.group(0), "").rstrip()
    return body, js


def _strip_cot(text: str) -> str:
    """Remove Thought/Action/Observation lines and inline JSON tool calls from the visible answer."""
    lines = (text or "").splitlines()
    out: List[str] = []
    for ln in lines:
        l = ln.strip()
        if not l:
            out.append(ln); continue
        if l.startswith(("Thought:", "Action:", "Observation:")):
            continue
        # drop bare tool-call JSON lines
        if l.startswith("{") and l.endswith("}") and '"name"' in l and '"parameters"' in l:
            try:
                json.loads(l)
                continue
            except Exception:
                pass
        out.append(ln)
    res = "\n".join(out).strip()
    if "## TL;DR" not in res and "## Réponse" not in res:
        res = "## Réponse\n" + res
    return res


def _parse_tools_from_react(text: str) -> List[Dict[str, Any]]:
    """Best-effort parser to collect called tools when the model didn't emit tool_trace."""
    tools: List[Dict[str, Any]] = []

    # 1) JSON tool call lines: {"name": "...", "parameters": {...}}
    for ln in (text or "").splitlines():
        l = ln.strip()
        if l.startswith("{") and l.endswith("}") and '"name"' in l:
            try:
                obj = json.loads(l)
                name = obj.get("name"); params = obj.get("parameters", {})
                if name:
                    tools.append({"name": name, "args": params})
            except Exception:
                pass

    # 2) ReAct-style: Action: <tool> Action Input: <args...>
    pat = re.compile(r"Action:\s*([A-Za-z0-9_\.\-]+)\s*Action Input:\s*(.+)", re.IGNORECASE)
    for m in pat.finditer(text or ""):
        name = m.group(1).strip()
        raw = m.group(2).strip()
        args: Any = raw
        try:
            if raw.startswith("{") and raw.endswith("}"):
                args = json.loads(raw)
            else:
                kv = {}
                for seg in re.split(r"[,\s]+", raw):
                    if "=" in seg:
                        k, v = seg.split("=", 1)
                        v = v.strip().strip('"').strip("'")
                        if re.fullmatch(r"\d+", v): v = int(v)
                        elif re.fullmatch(r"\d+\.\d+", v): v = float(v)
                        kv[k.strip()] = v
                if kv: args = kv
        except Exception:
            pass
        tools.append({"name": name, "args": args})

    dedup: Dict[str, Dict[str, Any]] = {}
    for t in tools:
        key = f"{t.get('name')}::{json.dumps(t.get('args', {}), sort_keys=True, ensure_ascii=False)}"
        dedup[key] = t
    return list(dedup.values())


def make_llm():
    """Local LLM via Ollama."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    return ChatOllama(model=model, base_url=base_url, temperature=0.1)


def make_client(cfg_path: str) -> MCPClient:
    return MCPClient.from_config_file(cfg_path)


def make_agent(client: MCPClient, require_tools: bool = True, max_steps: int = 24) -> MCPAgent:
    global _FORCE_TOOL_USE
    _FORCE_TOOL_USE = bool(require_tools)
    llm = make_llm()
    return MCPAgent(llm=llm, client=client, max_steps=max_steps, memory_enabled=True)


def _tool_force_instructions() -> str:
    if not _FORCE_TOOL_USE:
        return ""
    return (
        "ÉTAPES OBLIGATOIRES AVANT RÉPONSE FINALE:\n"
        "1) Appelle `mcp_catalog` pour lister les outils.\n"
        "2) Choisis un outil pertinent et appelle-le (ex: `duckduckgo_web_search`, `airbnb_search`, `browser_navigate`).\n"
        "3) N'affiche pas ta réflexion. Fournis la synthèse finale au format demandé.\n"
    )


def run_once(agent: MCPAgent, user_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    guide = OUTPUT_GUIDE_BASE
    force = _tool_force_instructions()
    prompt_tools = f"{guide}\n\n{force}\nQuestion:\n{user_text}".strip()

    out1: str = asyncio.run(agent.run(prompt_tools))
    body1, trace1 = _extract_trace_and_body(out1)
    body1 = _strip_cot(body1)
    trace_obj = trace1 or {"tools_used": _parse_tools_from_react(out1)}

    needs_wrap = ("## TL;DR" not in body1) or ("## Réponse" not in body1)
    if needs_wrap:
        finalize = ("Rédige maintenant UNIQUEMENT la réponse finale au format exigé (## TL;DR, ## Réponse, ## Comment j'ai travaillé). "
                    "N'utilise aucun outil supplémentaire. N'affiche pas ta réflexion.")
        out2: str = asyncio.run(agent.run(finalize))
        body2, _ = _extract_trace_and_body(out2)
        body2 = _strip_cot(body2)
        if "## Réponse" in body2:
            body1 = body2

    return body1, trace_obj


# Clean shutdown for Streamlit
_CLIENTS: List[MCPClient] = []

def register_client_for_cleanup(c: MCPClient):
    _CLIENTS.append(c)

@atexit.register
def _cleanup():
    try:
        for c in _CLIENTS:
            if getattr(c, "sessions", None):
                asyncio.run(c.close_all_sessions())
    except Exception:
        pass
