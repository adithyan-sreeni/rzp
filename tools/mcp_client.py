import os
import logging
from typing import List, Any
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

RAZORPAY_MCP_URL = "https://mcp.razorpay.com/mcp"

async def get_mcp_client(mcp_token: str = None) -> MultiServerMCPClient:
    token = mcp_token or os.getenv("RAZORPAY_MCP_TOKEN", "")
    headers = {}
    if token:
        if not token.startswith("Basic "):
            headers["Authorization"] = f"Basic {token}"
        else:
            headers["Authorization"] = token

    client = MultiServerMCPClient({
        "razorpay": {
            "url": RAZORPAY_MCP_URL,
            "transport": "streamable_http",
            "headers": headers,
            "timeout": 30.0,
            "sse_read_timeout": 300.0
        }
    })
    return client

async def load_razorpay_mcp_tools(mcp_token: str = None) -> List[Any]:
    try:
        client = await get_mcp_client(mcp_token)
        tools = await client.get_tools()
        logger.info(f"Loaded {len(tools)} tools from Razorpay MCP server.")
        return tools
    except Exception as e:
        logger.error(f"Failed to load tools from Razorpay MCP server: {e}")
        return []
