import os
import sys

# Ensure project root directory is present in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import re
import json
from typing import TypedDict, List, Dict, Any, Optional, Union
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from langgraph.types import Send
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, BaseMessage
from agents.coordinator import route_question
from agents.settlement_agent import SETTLEMENT_SYSTEM_PROMPT, get_settlement_agent
from agents.transaction_agent import TRANSACTION_SYSTEM_PROMPT, get_transaction_agent
from tools.settlement_tools import get_all_settlement_tools, get_all_transaction_tools


class AgentState(TypedDict):
    question: str
    route: str
    context: str
    settlement_answer: str
    transaction_answer: str
    final_answer: str
    cited_records: List[str]
    confidence: str
    exceptions: List[str]
    data_source: str   # 'live_mcp' | 'synthetic' | 'error'
    # Annotated with add_messages reducer so LangGraph handles list merging correctly
    messages: Annotated[List[BaseMessage], add_messages]


def _extract_text(content: Any) -> str:
    """
    Safely coerce LangChain response.content to a plain string.
    Gemini models with bound tools may return content as a list of content blocks
    e.g. [{'type': 'text', 'text': '...'}] instead of a plain string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # LangChain content block format: {'type': 'text', 'text': '...'}
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return " ".join(p for p in parts if p).strip()
    return str(content)


# ─── Nodes ────────────────────────────────────────────────────────────────────

def coordinator_node(state: AgentState) -> Dict[str, Any]:
    question = state.get("question", "")
    routing_result = route_question(question)
    return {
        "route": routing_result.get("route", "settlement_agent"),
        "context": routing_result.get("context", ""),
        "messages": [HumanMessage(content=question)],
    }


async def settlement_node(state: AgentState) -> Dict[str, Any]:
    question = state.get("question", "")
    context = state.get("context", "")

    try:
        tools, data_source = await get_all_settlement_tools()
    except RuntimeError as e:
        return {
            "settlement_answer": f"⚠️ Data source error: {e}",
            "data_source": "error",
        }

    llm_with_tools = get_settlement_agent(tools=tools)

    # Local message list — each node manages its own LLM conversation
    messages = [
        SystemMessage(content=SETTLEMENT_SYSTEM_PROMPT),
        HumanMessage(content=f"Context: {context}\nQuestion: {question}"),
    ]
    response = await llm_with_tools.ainvoke(messages)

    # Tool-call loop
    tool_map = {t.name: t for t in tools}
    while getattr(response, "tool_calls", None):
        messages.append(response)
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            if tool_name in tool_map:
                try:
                    tool_output = await tool_map[tool_name].ainvoke(tool_args)
                except Exception as e:
                    tool_output = json.dumps({"error": str(e)})
            else:
                tool_output = json.dumps({"error": f"Tool '{tool_name}' not available"})
            messages.append(
                ToolMessage(content=str(tool_output), tool_call_id=tc["id"])
            )
        response = await llm_with_tools.ainvoke(messages)

    return {
        "settlement_answer": _extract_text(response.content),
        "data_source": data_source,
    }


async def transaction_node(state: AgentState) -> Dict[str, Any]:
    question = state.get("question", "")
    context = state.get("context", "")

    try:
        tools, data_source = await get_all_transaction_tools()
    except RuntimeError as e:
        return {
            "transaction_answer": f"⚠️ Data source error: {e}",
            "data_source": "error",
        }

    llm_with_tools = get_transaction_agent(tools=tools)

    messages = [
        SystemMessage(content=TRANSACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"Context: {context}\nQuestion: {question}"),
    ]
    response = await llm_with_tools.ainvoke(messages)

    tool_map = {t.name: t for t in tools}
    while getattr(response, "tool_calls", None):
        messages.append(response)
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            if tool_name in tool_map:
                try:
                    tool_output = await tool_map[tool_name].ainvoke(tool_args)
                except Exception as e:
                    tool_output = json.dumps({"error": str(e)})
            else:
                tool_output = json.dumps({"error": f"Tool '{tool_name}' not available"})
            messages.append(
                ToolMessage(content=str(tool_output), tool_call_id=tc["id"])
            )
        response = await llm_with_tools.ainvoke(messages)

    return {
        "transaction_answer": _extract_text(response.content),
        "data_source": data_source,
    }


def merge_node(state: AgentState) -> Dict[str, Any]:
    settlement_ans = _extract_text(state.get("settlement_answer", "")).strip()
    transaction_ans = _extract_text(state.get("transaction_answer", "")).strip()

    if settlement_ans and transaction_ans:
        final = (
            f"### Settlement Analysis\n{settlement_ans}\n\n"
            f"### Transaction Analysis\n{transaction_ans}"
        )
    elif settlement_ans:
        final = settlement_ans
    elif transaction_ans:
        final = transaction_ans
    else:
        final = "I cannot determine this from the available settlement data."

    # Extract cited record IDs from the answer text
    cited_setl  = re.findall(r"setl_[A-Za-z0-9]+", final)
    cited_pay   = re.findall(r"pay_[A-Za-z0-9]+",  final)
    cited_order = re.findall(r"order_[A-Za-z0-9]+", final)
    cited_records = list(set(cited_setl + cited_pay + cited_order))

    confidence = "high"
    if "cannot determine" in final.lower() or "missing" in final.lower():
        confidence = "medium" if cited_records else "low"

    exceptions: List[str] = []
    if "cannot determine" in final.lower():
        exceptions.append("Some requested data could not be determined.")

    return {
        "final_answer": final,
        "cited_records": cited_records,
        "confidence": confidence,
        "exceptions": exceptions,
    }


# ─── Routing ──────────────────────────────────────────────────────────────────

def route_decision(state: AgentState) -> Union[str, List[Send]]:
    """
    Returns the next node name, or a list of Send objects for parallel fan-out
    when the coordinator routes to 'both' agents simultaneously.
    """
    route = state.get("route", "settlement_agent")

    if route == "both":
        # Fan out to both agents in parallel; both feed into merge_node
        return [
            Send("settlement_node", state),
            Send("transaction_node", state),
        ]
    elif route == "transaction_agent":
        return "transaction_node"
    else:
        return "settlement_node"


# ─── Graph ────────────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("coordinator",      coordinator_node)
    builder.add_node("settlement_node",  settlement_node)
    builder.add_node("transaction_node", transaction_node)
    builder.add_node("merge",            merge_node)

    builder.add_edge(START, "coordinator")

    builder.add_conditional_edges(
        "coordinator",
        route_decision,
        {
            "settlement_node":  "settlement_node",
            "transaction_node": "transaction_node",
        },
    )

    builder.add_edge("settlement_node",  "merge")
    builder.add_edge("transaction_node", "merge")
    builder.add_edge("merge", END)

    return builder.compile()
