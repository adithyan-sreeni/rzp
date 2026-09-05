import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
from typing import List, Any, Tuple
from tools.synthetic_tools import get_synthetic_settlements, get_synthetic_payments, get_synthetic_settlement_recon
from tools.mcp_client import load_razorpay_mcp_tools

logger = logging.getLogger(__name__)

# ─── Tool Selection Logic ──────────────────────────────────────────────────────
#
# USE_SYNTHETIC=true  → ONLY synthetic tools, no MCP attempted
# USE_SYNTHETIC=false → ONLY MCP tools. Synthetic tools are NOT given to the agent.
#                       If MCP returns 0 tools, raise clearly so the agent/UI can
#                       surface the problem instead of silently hallucinating.
#
# This prevents the agent answering questions about synthetic setl_XXXXXX IDs
# when the user has live Razorpay data and USE_SYNTHETIC=false.
# ──────────────────────────────────────────────────────────────────────────────

async def get_all_settlement_tools() -> Tuple[List[Any], str]:
    """
    Returns (tools, data_source) where data_source is 'live_mcp' or 'synthetic'.
    Raises RuntimeError if USE_SYNTHETIC=false and MCP connection fails.
    """
    use_synthetic = os.getenv("USE_SYNTHETIC", "true").lower() == "true"

    if use_synthetic:
        logger.info("USE_SYNTHETIC=true — using synthetic settlement tools only.")
        return [get_synthetic_settlements, get_synthetic_settlement_recon], "synthetic"

    # USE_SYNTHETIC=false → attempt live MCP only
    mcp_tools = await load_razorpay_mcp_tools()
    if not mcp_tools:
        raise RuntimeError(
            "USE_SYNTHETIC=false but Razorpay MCP returned no tools. "
            "Check your RAZORPAY_MCP_TOKEN and network connectivity. "
            "Set USE_SYNTHETIC=true to use synthetic fallback data."
        )

    # Inject custom tool for calculating exact gross amounts and platform fees
    from tools.custom_tools import get_settlement_gross_details
    mcp_tools.append(get_settlement_gross_details)

    logger.info(f"USE_SYNTHETIC=false — using {len(mcp_tools)} live MCP settlement tools.")
    return mcp_tools, "live_mcp"


async def get_all_transaction_tools() -> Tuple[List[Any], str]:
    """
    Returns (tools, data_source) where data_source is 'live_mcp' or 'synthetic'.
    Raises RuntimeError if USE_SYNTHETIC=false and MCP connection fails.
    """
    use_synthetic = os.getenv("USE_SYNTHETIC", "true").lower() == "true"

    if use_synthetic:
        logger.info("USE_SYNTHETIC=true — using synthetic transaction tools only.")
        return [get_synthetic_payments], "synthetic"

    mcp_tools = await load_razorpay_mcp_tools()
    if not mcp_tools:
        raise RuntimeError(
            "USE_SYNTHETIC=false but Razorpay MCP returned no tools. "
            "Check your RAZORPAY_MCP_TOKEN and network connectivity. "
            "Set USE_SYNTHETIC=true to use synthetic fallback data."
        )

    logger.info(f"USE_SYNTHETIC=false — using {len(mcp_tools)} live MCP transaction tools.")
    return mcp_tools, "live_mcp"
