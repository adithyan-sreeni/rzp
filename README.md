# Razorpay Settlement Q&A Agent
## Razorpay AI Buildathon | Track 04: AI Finance Controller

A multi-agent AI system built with **LangGraph**, **LangChain**, and **Razorpay MCP Server** that allows merchants to ask natural language questions about settlement statements, fee deductions, payment breakdowns, timing delays, and reconciliation anomalies.

---

## 🏗️ Architecture

```
[Merchant Question]
        ↓
[Streamlit UI]
        ↓
[Coordinator Agent - Gemini Flash 2.0]
  "route only, never answer"
        ↓
   ┌────┴────┐
   ↓         ↓
[Settlement  [Transaction
  Agent]       Agent]
   ↓         ↓
[Razorpay Remote MCP Server] (https://mcp.razorpay.com/mcp)
        ↓
[Synthetic Data Fallback] (if test mode data is sparse)
        ↓
[Answer + Cited Record IDs]
        ↓
[Eval Harness - 25 Ground Truth Q&A Pairs]
```

---

## 🛠️ Tech Stack

- **LLM:** Gemini Flash 2.0 (`gemini-2.0-flash`) via `langchain-google-genai`
- **Agent Framework:** LangGraph (`StateGraph`)
- **MCP Adapter:** `langchain-mcp-adapters`
- **MCP Endpoint:** Razorpay Remote MCP Server (`https://mcp.razorpay.com/mcp`)
- **UI:** Streamlit
- **Eval Harness:** `sentence-transformers` (`all-MiniLM-L6-v2`) + exact keyword match
- **Language:** Python 3.11+

---

## 📁 Project Structure

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
    └── generate_synthetic_data.py      # Data generation script
```

---

## 🚀 Setup & Execution

### 1. Environment Configuration
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

Set `GOOGLE_API_KEY`, `RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET`, and `RAZORPAY_MCP_TOKEN`.

### 2. Generate Synthetic Dataset
```bash
python scripts/generate_synthetic_data.py
```

### 3. Run Evaluation Harness
```bash
python eval/run_eval.py
```

### 4. Launch Streamlit UI
```bash
streamlit run ui/app.py
```

---

## ⚠️ Honest Limitations & Transparent Trade-offs

1. **Test Mode Data Sparsity:** Razorpay API test mode accounts often return 0 settlement records. The system uses transparent fallback to `synthetic_settlements.json` controlled by `USE_SYNTHETIC=true`.
2. **Read-Only Scope:** Agents are restricted to read-only queries for settlements and transactions. Write actions (payout creation, payment links) are explicitly excluded for security.
3. **Semantic Scoring Threshold:** Evaluation uses a conservative similarity threshold (0.75) with `sentence-transformers` alongside exact keyword matching.
