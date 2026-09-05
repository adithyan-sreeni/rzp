import json
import os
from typing import Dict, Any, Literal
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

class RoutingDecision(BaseModel):
    route: Literal["settlement_agent", "transaction_agent", "both"] = Field(
        ..., description="Target specialist agent route"
    )
    question: str = Field(..., description="Original question")
    context: str = Field("", description="Extracted IDs, filters, or context")

COORDINATOR_SYSTEM_PROMPT = """You are a coordinator agent for a merchant settlement assistant.
Your ONLY job is to route incoming questions to the correct specialist agent.
You do NOT answer questions yourself.

Routing rules:
- Questions about settlement amounts, fees, deductions, reconciliation, settlement status, UTR numbers → route to: settlement_agent
- Questions about individual payments, transaction failures, refund status, payment methods → route to: transaction_agent
- Questions that need both → route to: both

Output format:
Respond strictly with a JSON object matching this schema:
{
  "route": "settlement_agent" | "transaction_agent" | "both",
  "question": "<original question>",
  "context": "<any relevant IDs or filters you extracted from the question>"
}

Do not add explanation. Do not answer the question. Just route."""

def get_coordinator_model(model_name: str = "gemini-2.0-flash"):
    api_key = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.0
    )

def route_question(question: str, model_name: str = "gemini-2.0-flash") -> Dict[str, Any]:
    llm = get_coordinator_model(model_name)
    
    prompt = f"{COORDINATOR_SYSTEM_PROMPT}\n\nQuestion: {question}"
    try:
        structured_llm = llm.with_structured_output(RoutingDecision)
        decision = structured_llm.invoke(prompt)
        return decision.model_dump()
    except Exception as e:
        # Fallback raw parsing
        response = llm.invoke(prompt)
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        try:
            parsed = json.loads(content)
            return {
                "route": parsed.get("route", "settlement_agent"),
                "question": question,
                "context": parsed.get("context", "")
            }
        except Exception:
            return {
                "route": "settlement_agent",
                "question": question,
                "context": ""
            }
