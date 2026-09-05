import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
from typing import List, Any
from tools.synthetic_tools import get_synthetic_settlements, get_synthetic_payments, get_synthetic_settlement_recon
from tools.mcp_client import load_razorpay_mcp_tools

logger = logging.getLogger(__name__)

async def get_all_settlement_tools() -> List[Any]:
    use_synthetic = os.getenv("USE_SYNTHETIC", "true").lower() == "true"
    synthetic_tools = [get_synthetic_settlements, get_synthetic_settlement_recon]
    
    if use_synthetic:
        logger.info("Using synthetic settlement tools based on USE_SYNTHETIC=true environment setting.")
        return synthetic_tools
    
    mcp_tools = await load_razorpay_mcp_tools()
    if not mcp_tools:
        logger.warning("No MCP tools returned; falling back to synthetic settlement tools.")
        return synthetic_tools
        
    return mcp_tools + synthetic_tools

async def get_all_transaction_tools() -> List[Any]:
    use_synthetic = os.getenv("USE_SYNTHETIC", "true").lower() == "true"
    synthetic_tools = [get_synthetic_payments]
    
    if use_synthetic:
        logger.info("Using synthetic transaction tools based on USE_SYNTHETIC=true environment setting.")
        return synthetic_tools
        
    mcp_tools = await load_razorpay_mcp_tools()
    if not mcp_tools:
        logger.warning("No MCP tools returned; falling back to synthetic transaction tools.")
        return synthetic_tools
        
    return mcp_tools + synthetic_tools
