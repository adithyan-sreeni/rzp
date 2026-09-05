import os
import sys

# Ensure project root directory is present in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import re
import json
from typing import TypedDict, List, Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
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
    messages: List[Any]

def _extract_text(content: Any) -> str:
    """
    Safely coerce LangChain response content to a plain string.
    Gemini models with bound tools may return content as a list of content blocks
    (e.g. [{'type': 'text', 'text': '...'}]) instead of a plain string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return " ".join(parts).strip()
    return str(content) if content is not None else ""

def coordinator_node(state: AgentState) -> Dict[str, Any]:
    question = state.get("question", "")
    routing_result = route_question(question)
    return {
        "route": routing_result.get("route", "settlement_agent"),
        "context": routing_result.get("context", ""),
        "messages": [HumanMessage(content=question)]
    }

async def settlement_node(state: AgentState) -> Dict[str, Any]:
    question = state.get("question", "")
    context = state.get("context", "")
    tools = await get_all_settlement_tools()
    llm_with_tools = get_settlement_agent(tools=tools)
    
    # Execute agent call with tool execution loop if needed
    messages = [SystemMessage(content=SETTLEMENT_SYSTEM_PROMPT), HumanMessage(content=f"Context: {context}\nQuestion: {question}")]
    response = llm_with_tools.invoke(messages)
    
    tool_map = {tool.name: tool for tool in tools}
    while getattr(response, "tool_calls", None):
        messages.append(response)
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            if tool_name in tool_map:
                try:
                    tool_output = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    tool_output = json.dumps({"error": str(e)})
            else:
                tool_output = json.dumps({"error": f"Tool {tool_name} unavailable"})
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
        response = llm_with_tools.invoke(messages)
        
    return {
        "settlement_answer": _extract_text(response.content)
    }

async def transaction_node(state: AgentState) -> Dict[str, Any]:
    question = state.get("question", "")
    context = state.get("context", "")
    tools = await get_all_transaction_tools()
    llm_with_tools = get_transaction_agent(tools=tools)
    
    messages = [SystemMessage(content=TRANSACTION_SYSTEM_PROMPT), HumanMessage(content=f"Context: {context}\nQuestion: {question}")]
    response = llm_with_tools.invoke(messages)
    
    tool_map = {tool.name: tool for tool in tools}
    while getattr(response, "tool_calls", None):
        messages.append(response)
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            if tool_name in tool_map:
                try:
                    tool_output = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    tool_output = json.dumps({"error": str(e)})
            else:
                tool_output = json.dumps({"error": f"Tool {tool_name} unavailable"})
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
        response = llm_with_tools.invoke(messages)
        
    return {
        "transaction_answer": _extract_text(response.content)
    }

def merge_node(state: AgentState) -> Dict[str, Any]:
    settlement_ans = _extract_text(state.get("settlement_answer", "")).strip()
    transaction_ans = _extract_text(state.get("transaction_answer", "")).strip()
    
    if settlement_ans and transaction_ans:
        final = f"### Settlement Analysis\n{settlement_ans}\n\n### Transaction Analysis\n{transaction_ans}"
    elif settlement_ans:
        final = settlement_ans
    elif transaction_ans:
        final = transaction_ans
    else:
        final = "I cannot determine this from the available settlement data."
        
    # Extract cited records matching setl_XXXX or pay_XXXX or order_XXXX
    cited_setl = re.findall(r"setl_[A-Za-z0-9]+", final)
    cited_pay = re.findall(r"pay_[A-Za-z0-9]+", final)
    cited_order = re.findall(r"order_[A-Za-z0-9]+", final)
    cited_records = list(set(cited_setl + cited_pay + cited_order))
    
    # Assess confidence
    confidence = "high"
    if "cannot determine" in final.lower() or "missing" in final.lower():
        confidence = "medium" if cited_records else "low"
        
    exceptions = []
    if "cannot determine" in final.lower():
        exceptions.append("Some requested data could not be determined.")
        
    return {
        "final_answer": final,
        "cited_records": cited_records,
        "confidence": confidence,
        "exceptions": exceptions
    }

def route_decision(state: AgentState) -> str:
    route = state.get("route", "settlement_agent")
    if route == "transaction_agent":
        return "transaction_node"
    elif route == "both":
        return "both"
    return "settlement_node"

def build_graph():
    builder = StateGraph(AgentState)
    
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("settlement_node", settlement_node)
    builder.add_node("transaction_node", transaction_node)
    builder.add_node("merge", merge_node)
    
    builder.add_edge(START, "coordinator")
    
    builder.add_conditional_edges(
        "coordinator",
        route_decision,
        {
            "settlement_node": "settlement_node",
            "transaction_node": "transaction_node",
            "both": "settlement_node"  # For 'both', first settlement then transaction or direct path
        }
    )
    
    # Connect nodes to merge
    builder.add_edge("settlement_node", "merge")
    builder.add_edge("transaction_node", "merge")
    builder.add_edge("merge", END)
    
    return builder.compile()
