# Razorpay Settlement Q&A Agent — Implementation Plan
## Razorpay AI Buildathon | Track 04: AI Finance Controller

---

## 1. Project Goal

Build a multi-agent Settlement Q&A system that:
- Accepts natural language questions from a merchant about their settlements
- Routes queries through a coordinator to specialized sub-agents
- Pulls live data from Razorpay's official MCP server
- Falls back to synthetic settlement data for evaluation purposes
- Runs a measurable eval harness on 25 ground truth Q&A pairs
- Reports accuracy %, failed cases, and honest exception list

This is NOT a chatbot demo. The eval harness with honest numbers is the differentiator.

---

## 2. Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| LLM | Gemini Flash 2.0 (`gemini-2.0-flash`) | Free tier, cheap at scale, fast |
| Agent Framework | LangGraph | Extends LangChain (already familiar), native multi-agent support |
| MCP Integration | `langchain-mcp-adapters` | Bridges LangGraph with Razorpay MCP server |
| Razorpay Data | Razorpay Remote MCP Server | Official, hosted, no infrastructure needed |
| Synthetic Data | Python-generated JSON/CSV | For eval harness when test mode returns empty settlements |
| UI | Streamlit | Fast to build, clean enough for demo |
| Eval Scoring | `sentence-transformers` + exact match | Semantic similarity + keyword accuracy |
| Language | Python 3.11+ | |

Do NOT use Flutter. Do NOT use LangChain alone (use LangGraph). Do NOT use the Anthropic API (budget constraint).

---

## 3. Folder Structure

```
razorpay-settlement-agent/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── synthetic_settlements.json      # 80 synthetic settlement records
│   ├── synthetic_payments.json         # linked payment records
│   └── ground_truth_qa.json           # 25 Q&A pairs with expected answers
│
├── agents/
│   ├── __init__.py
│   ├── coordinator.py                  # Coordinator agent — router only
│   ├── settlement_agent.py             # Settlement-specific agent
│   └── transaction_agent.py           # Transaction/payment-level agent
│
├── tools/
│   ├── __init__.py
│   ├── mcp_client.py                   # Razorpay MCP connection setup
│   ├── settlement_tools.py             # Direct API fallback tools
│   └── synthetic_tools.py             # Tools that query synthetic data
│
├── graph/
│   ├── __init__.py
│   └── agent_graph.py                  # LangGraph graph definition
│
├── eval/
│   ├── __init__.py
│   ├── run_eval.py                     # Main eval script
│   └── scorer.py                       # Scoring logic
│
├── ui/
│   └── app.py                          # Streamlit app
│
└── scripts/
    └── generate_synthetic_data.py      # One-time data generation script
```

---

## 4. Environment Variables

```env
# .env
GOOGLE_API_KEY=your_gemini_api_key
RAZORPAY_KEY_ID=rzp_test_xxxxxxxxxxxx
RAZORPAY_KEY_SECRET=your_secret
RAZORPAY_MCP_TOKEN=base64(key_id:key_secret)   # echo KEY:SECRET | base64
USE_SYNTHETIC=true   # set false if live MCP returns real settlement data
```

---

## 5. Razorpay MCP Server Setup

**Endpoint:** `https://mcp.razorpay.com/mcp`  
**Auth:** `Authorization: Basic <base64(KEY_ID:KEY_SECRET)>`  
**Old SSE endpoint is deprecated** as of August 13, 2025. Use the streamable HTTP endpoint above.

**How to generate token:**
```bash
echo "rzp_test_KEY:SECRET" | base64
```

**Relevant MCP tools to use (bind only these to agents):**

For Settlement Agent:
- fetch settlement by ID
- list settlements (with filters: from, to, count)
- fetch settlement recon/combined report

For Transaction Agent:
- fetch payment by ID
- list payments (with filters)
- fetch order details

Do NOT bind payment link creation, payout, or QR tools to any agent. Scope is read-only Q&A.

**MCP connection in Python using langchain-mcp-adapters:**
```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "razorpay": {
        "url": "https://mcp.razorpay.com/mcp",
        "transport": "streamable_http",
        "headers": {
            "Authorization": f"Basic {os.getenv('RAZORPAY_MCP_TOKEN')}"
        }
    }
})
tools = await client.get_tools()
```

If MCP returns no settlement data in test mode (expected), fall back to synthetic_tools.py which queries synthetic_settlements.json. Be transparent about this in the UI.

---

## 6. Synthetic Data Schema

Generate 80 settlement records in this structure (mirrors Razorpay API response):

```json
{
  "id": "setl_XXXXXXXXXX",
  "entity": "settlement",
  "amount": 95000,
  "status": "processed",
  "fees": 2500,
  "tax": 450,
  "utr": "HDFC2025091234567",
  "description": null,
  "created_at": 1725000000,
  "settled_at": 1725086400,
  "payments_count": 12,
  "payment_ids": ["pay_XXXX", "pay_YYYY"],
  "_anomaly": {
    "type": "unexplained_deduction",
    "description": "Fee exceeds expected 2.5% rate",
    "ground_truth_explanation": "Additional GST applied due to interstate transaction"
  }
}
```

**Anomaly distribution across 80 records:**
- 10% (8 records): unexplained deductions (fee > expected rate)
- 10% (8 records): timing mismatches (settled_at - created_at > 5 days)
- 5% (4 records): duplicate UTR numbers
- 5% (4 records): settlement amount doesn't match sum of payment amounts
- 70% (56 records): clean, normal records

Remove `_anomaly` field before feeding to agents — it's only for eval ground truth.

---

## 7. Ground Truth Q&A Pairs (25 total)

Structure for `ground_truth_qa.json`:

```json
[
  {
    "id": "q001",
    "question": "Why is settlement setl_ABC123 lower than expected?",
    "expected_answer_keywords": ["fee", "deduction", "GST", "2.5%"],
    "relevant_record_ids": ["setl_ABC123"],
    "category": "deduction_explanation",
    "difficulty": "medium"
  }
]
```

**Categories to cover (spread across 25 questions):**
- deduction_explanation (5 questions)
- settlement_status (4 questions)
- timing_query (4 questions)
- anomaly_detection (4 questions)
- aggregation (sum, count, average) (4 questions)
- comparison_across_settlements (4 questions)

---

## 8. Agent Definitions

### 8.1 Coordinator Agent

**Role:** Router only. Never answers directly. Reads the user question, decides which sub-agent handles it, passes the full question and relevant context.

**System Prompt:**
```
You are a coordinator agent for a merchant settlement assistant.
Your ONLY job is to route incoming questions to the correct specialist agent.
You do NOT answer questions yourself.

Routing rules:
- Questions about settlement amounts, fees, deductions, reconciliation, 
  settlement status, UTR numbers → route to: settlement_agent
- Questions about individual payments, transaction failures, refund status, 
  payment methods → route to: transaction_agent
- Questions that need both → route to settlement_agent first, then transaction_agent

Output format (strictly follow this):
{
  "route": "settlement_agent" | "transaction_agent" | "both",
  "question": "<original question>",
  "context": "<any relevant IDs or filters you extracted from the question>"
}

Do not add explanation. Do not answer the question. Just route.
```

**LangGraph node behavior:** Parse the routing output, pass state to the correct node.

---

### 8.2 Settlement Agent

**Role:** Answers all settlement-level questions. Cites specific record IDs in every answer. Explicitly states what it cannot determine.

**System Prompt:**
```
You are a settlement analyst agent for Razorpay merchants.
You have access to tools that fetch settlement records and reconciliation data.

Your responsibilities:
- Answer questions about settlement amounts, fees, tax deductions, UTR numbers
- Explain discrepancies between expected and actual settlement amounts
- Identify timing anomalies in settlement cycles
- Surface duplicate settlements or reconciliation mismatches

Rules you must follow:
1. Always cite the settlement ID (setl_XXXX) for every claim you make
2. If you find an anomaly, explain it clearly — do not hide it
3. If you cannot determine the answer from available data, say so explicitly:
   "I cannot determine this from the available settlement data."
4. Never fabricate numbers. If a record is missing, say it is missing.
5. Format amounts in INR (e.g., ₹950.00 means amount field value / 100)
6. Amounts in Razorpay API are in paise. Always convert to rupees in your answer.

Output format:
- Direct answer to the question
- Supporting evidence: list of settlement IDs referenced
- Confidence: high / medium / low
- Exceptions: anything you could not resolve
```

**Tools bound to this agent:** settlement fetch, settlement list, settlement recon, synthetic settlement query.

---

### 8.3 Transaction Agent

**Role:** Handles payment-level detail. Surfaces failure reasons, refund status, payment method breakdown.

**System Prompt:**
```
You are a transaction analyst agent for Razorpay merchants.
You have access to tools that fetch individual payment records and order details.

Your responsibilities:
- Explain why specific payments failed or were captured
- Show payment method breakdown for a settlement
- Track refund status for specific transactions
- Link payment IDs to their parent settlement

Rules you must follow:
1. Always cite payment ID (pay_XXXX) and order ID (order_XXXX) for every claim
2. Distinguish between payment failed vs payment captured vs payment refunded
3. If a payment ID doesn't exist in available data, say so explicitly
4. Never guess at failure reasons — only report what the data shows
5. Amounts are in paise — always convert to rupees in your output

Output format:
- Direct answer
- Payment IDs referenced
- Confidence: high / medium / low
- Exceptions: unresolved items
```

**Tools bound to this agent:** payment fetch, payment list, order fetch, synthetic payment query.

---

## 9. LangGraph Graph Definition

```python
# graph/agent_graph.py structure

from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

class AgentState(TypedDict):
    question: str
    route: str
    context: str
    settlement_answer: str
    transaction_answer: str
    final_answer: str
    cited_records: list
    confidence: str
    exceptions: list

# Nodes: coordinator_node, settlement_node, transaction_node, merge_node
# Edges: START → coordinator → (conditional) → settlement or transaction or both → merge → END

# Conditional routing function reads state["route"] and returns next node name
```

State is passed between nodes. Each node reads from state, adds its output, passes forward. The merge node combines answers when both agents are called.

---

## 10. Eval Harness

**File:** `eval/run_eval.py`

**What it does:**
1. Loads all 25 questions from `ground_truth_qa.json`
2. Runs each question through the full LangGraph pipeline
3. Scores each answer using two methods:
   - Keyword match: does the answer contain expected keywords?
   - Semantic similarity: cosine similarity using `sentence-transformers` (threshold 0.75)
4. Logs pass/fail per question
5. Outputs final report

**Output report structure:**
```
=== EVAL RESULTS ===
Total questions: 25
Passed (keyword): 18/25 (72%)
Passed (semantic): 20/25 (80%)
Failed cases:
  - q003: Expected 'duplicate UTR' — agent said 'timing mismatch'
  - q011: Agent returned 'cannot determine' — data was available
Exception list:
  - q017: MCP returned empty response, fell back to synthetic
  - q022: Agent timed out after 30s
Average confidence: medium
Average latency: 4.2s per question
```

This report goes in the README and the pitch video. Show the real numbers including failures. Do not hide them.

**Scoring script:**
```python
# eval/scorer.py
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def keyword_score(answer: str, keywords: list) -> bool:
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)

def semantic_score(answer: str, expected: str, threshold=0.75) -> bool:
    emb1 = model.encode(answer, convert_to_tensor=True)
    emb2 = model.encode(expected, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    return score >= threshold
```

---

## 11. Streamlit UI

**File:** `ui/app.py`

**Layout:**
- Left main panel: question input box, submit button, answer display with cited record IDs, confidence badge
- Right sidebar: eval results summary (accuracy %, top 3 failed cases), MCP connection status (live/synthetic), agent trace log (which agents were called)

**Key UI behaviors:**
- Show which agent(s) handled the query (coordinator decision visible)
- Show cited settlement/payment IDs as clickable tags
- Show confidence level as colored badge (high=green, medium=yellow, low=red)
- Show "Data source: Live MCP / Synthetic" clearly — do not hide this

---

## 12. README Structure (for public repo)

```
# Razorpay Settlement Q&A Agent
## Razorpay AI Buildathon | Track 04: AI Finance Controller

### Problem
### Architecture (embed diagram image)
### Tech Stack
### Eval Results (paste actual numbers)
### Honest Limitations
### Setup Instructions
### Demo Video Link
```

Honest limitations section must include: test mode settlement data is sparse so eval runs on synthetic data; MCP settlement tool coverage for granular reconciliation is limited; semantic scoring threshold is conservative at 0.75.

---

## 13. Architecture Diagram (Excalidraw)

Draw exactly this flow, nothing more:

```
[Merchant Question]
        ↓
[Streamlit UI]
        ↓
[Coordinator Agent - Gemini Flash]
  "route only, never answer"
        ↓
   ┌────┴────┐
   ↓         ↓
[Settlement  [Transaction
  Agent]       Agent]
   ↓         ↓
[Razorpay Remote MCP Server]
[https://mcp.razorpay.com/mcp]
        ↓
[Synthetic Data Fallback]
[if MCP returns empty]
        ↓
[Answer + Cited Record IDs]
        ↓
[Eval Harness - 25 Q&A pairs]
[Accuracy % + Exception List]
```

Label what you built in solid lines. Label "aspirational/production" items in dotted lines (PII redaction, WAF, full observability). This distinction is respected by evaluators.

---

## 14. Pitch Video Script (5 minutes)

- 0:00–0:30 — Problem: merchants can't interpret settlement statements
- 0:30–1:00 — Architecture walkthrough (show diagram)
- 1:00–3:00 — Live demo: ask 3 questions (one normal, one anomaly, one aggregation)
- 3:00–4:00 — Eval results: show the script running, show the output report with real numbers
- 4:00–5:00 — Honest limitations + what production version would add

Record in one take if possible. Don't over-edit.

---

## 15. Things to Explicitly Skip

Do NOT build these — mention them in diagram as aspirational:
- WAF / rate limiting / DDoS protection
- PII redaction middleware
- Full observability (just log agent steps to a .txt file)
- Cost tracker (mention it, don't build it)
- MCP server for each agent (use one shared MCP client)
- Authentication / merchant login (not needed for demo)

---

## 16. Packages to Install

```
langchain
langchain-google-genai
langgraph
langchain-mcp-adapters
streamlit
sentence-transformers
razorpay
python-dotenv
pandas
httpx
```

---

## 17. Build Order (One Day)

1. Set up repo, .env, install packages
2. Generate synthetic data (delegate to coding agent)
3. Build MCP client, test Razorpay connection, confirm which settlement tools are available
4. Build tools/synthetic_tools.py as fallback
5. Build all three agents with system prompts exactly as defined above
6. Build LangGraph graph wiring them together
7. Build eval harness and run against all 25 questions
8. Build Streamlit UI
9. End-to-end test, fix breaks
10. Architecture diagram + README + pitch video

Do not add features after step 7. Scope is locked.

---

## 18. Critical Rules for Coding Agent

- Amounts from Razorpay API are always in paise. Always divide by 100 for display.
- The MCP endpoint is `https://mcp.razorpay.com/mcp` — NOT the old SSE endpoint.
- The coordinator must output valid JSON for routing — add output parsing with error fallback.
- Every agent answer must include cited record IDs — enforce this in system prompt and parse output.
- Eval harness must run all 25 questions automatically — no manual cherry-picking.
- The `_anomaly` field in synthetic data is for ground truth only — strip it before feeding to agents.
- Use `USE_SYNTHETIC=true` env flag to toggle between live MCP and synthetic data cleanly.
- Do not commit API keys. Use .env and add it to .gitignore.
