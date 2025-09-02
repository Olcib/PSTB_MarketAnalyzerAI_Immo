import asyncio, json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from contextlib import AsyncExitStack

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except Exception as e:
    MCP_AVAILABLE = False
    _MCP_IMPORT_ERROR = e

@dataclass
class ToolInfo:
    server: str
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = None

class MCPServerConnection:
    def __init__(self, server_name: str, command: str, args: List[str]):
        self.server_name = server_name
        self.command = command
        self.args = args
        self.exit_stack: Optional[AsyncExitStack] = None
        self.session: Optional[ClientSession] = None

    async def start(self):
        if not MCP_AVAILABLE:
            raise RuntimeError(f"MCP SDK not available: {_MCP_IMPORT_ERROR}")
        self.exit_stack = AsyncExitStack()
        await self.exit_stack.__aenter__()
        params = StdioServerParameters(command=self.command, args=self.args, env=None)
        read, write = await self.exit_stack.enter_async_context(stdio_client(params))
        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

    async def list_tools(self) -> List[ToolInfo]:
        assert self.session is not None, "Server not started"
        tools_resp = await self.session.list_tools()
        items: List[ToolInfo] = []
        for t in tools_resp.tools:
            items.append(ToolInfo(
                server=self.server_name,
                name=getattr(t, 'name', ''),
                description=getattr(t, 'description', '') or "",
                input_schema=getattr(t, 'inputSchema', None) or {}
            ))
        return items

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        assert self.session is not None, "Server not started"
        result = await self.session.call_tool(tool_name, arguments=arguments or {})
        out: List[Dict[str, Any]] = []
        for c in (result.content or []):
            ctype = getattr(c, 'type', None) or (isinstance(c, dict) and c.get('type'))
            if ctype == 'text':
                text = getattr(c, 'text', None) or (isinstance(c, dict) and c.get('text'))
                out.append({"type": "text", "text": text})
            elif ctype == 'image':
                out.append({
                    "type": "image",
                    "data": getattr(c, 'data', None) or (isinstance(c, dict) and c.get('data')),
                    "mimeType": getattr(c, 'mimeType', None) or (isinstance(c, dict) and c.get('mimeType')),
                    "uri": getattr(c, 'uri', None) or (isinstance(c, dict) and c.get('uri'))
                })
            else:
                try:
                    out.append(json.loads(json.dumps(c, default=lambda o: getattr(o, '__dict__', str(o)))))
                except Exception:
                    out.append({"type": str(ctype or 'unknown')})
        return {"server": self.server_name, "tool": tool_name, "content": out}

    async def close(self):
        if self.exit_stack:
            try:
                await self.exit_stack.aclose()
            finally:
                self.exit_stack = None
                self.session = None

class MCPManager:
    def __init__(self):
        self.servers: Dict[str, MCPServerConnection] = {}

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> 'MCPManager':
        mgr = MCPManager()
        for name, spec in (cfg.get("mcpServers") or {}).items():
            cmd = spec.get("command")
            args = spec.get("args", [])
            mgr.servers[name] = MCPServerConnection(name, cmd, args)
        return mgr

    async def start_all(self):
        await asyncio.gather(*(s.start() for s in self.servers.values()))

    async def stop_all(self):
        await asyncio.gather(*(s.close() for s in self.servers.values()))

    async def list_all_tools(self) -> List[ToolInfo]:
        tools_nested = await asyncio.gather(*(s.list_tools() for s in self.servers.values()))
        flat: List[ToolInfo] = []
        for lst in tools_nested:
            flat.extend(lst)
        return flat

    async def call(self, server: str, tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if server not in self.servers:
            raise KeyError(f"Unknown server: {server}")
        return await self.servers[server].call_tool(tool, arguments)
