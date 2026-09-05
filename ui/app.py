import os
import sys
import asyncio
import streamlit as st

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.agent_graph import build_graph

st.set_page_config(
    page_title="Razorpay Settlement Q&A Agent",
    page_icon="💳",
    layout="wide"
)

# Header
st.title("💳 Razorpay Settlement Q&A Agent")
st.caption("AI Finance Controller — Track 04 | Razorpay AI Buildathon")

# Sidebar Configuration & Telemetry
with st.sidebar:
    st.header("⚙️ Data & Environment Status")
    
    use_synthetic = os.getenv("USE_SYNTHETIC", "true").lower() == "true"
    if use_synthetic:
        st.warning("⚠️ Data Source: Synthetic Fallback Mode (`USE_SYNTHETIC=true`)")
    else:
        st.success("🟢 Data Source: Live Razorpay MCP Endpoint (`https://mcp.razorpay.com/mcp`)")
        
    st.divider()
    st.header("📊 Eval Benchmark Summary")
    st.metric(label="Ground Truth Benchmark", value="25 Q&A Pairs")
    st.metric(label="Target Keyword Pass Rate", value="75%+")
    st.metric(label="Target Semantic Pass Rate", value="80%+")
    
    st.divider()
    st.markdown("### 🛠️ Architecture Trace")
    st.info("Coordinator Node → Settlement / Transaction Agent → Synthesis Merge Node")

# Main Query Interface
col_main, col_detail = st.columns([2, 1])

with col_main:
    st.subheader("Ask a Settlement Question")
    question = st.text_input(
        "Enter query (e.g. 'Why is settlement setl_000001 lower than expected?' or 'What is the status of UTR HDFC2025090000001?'):",
        key="query_input"
    )
    submit_btn = st.button("Analyze Question", type="primary")

    if submit_btn and question:
        with st.status("Executing Multi-Agent Workflow...", expanded=True) as status_container:
            st.write("1. Coordinator Agent classifying intent...")
            graph = build_graph()
            
            st.write("2. Routing to Specialist Agent(s) & querying MCP / data sources...")
            
            async def run_workflow():
                result_state = None
                async for event in graph.astream_events({"question": question}, version="v2"):
                    kind = event["event"]
                    name = event.get("name", "")
                    
                    if kind == "on_chat_model_end":
                        # Attempt to get text output (thinking) from model before it makes tool calls
                        output = event.get("data", {}).get("output", None)
                        if output and hasattr(output, "content") and output.content:
                            st.markdown(f"🧠 **Agent Thinking:**\n\n{output.content}")
                            
                    elif kind == "on_tool_start":
                        args = event['data'].get('input', {})
                        st.info(f"🛠️ **Tool Call**: `{name}`\n\n```json\n{args}\n```")
                        
                    elif kind == "on_tool_end":
                        output = event['data'].get('output', {})
                        st.success(f"✅ **Tool Ended**: `{name}`\n\n```json\n{output}\n```")
                        
                    elif kind == "on_chain_end" and name == "LangGraph":
                        result_state = event["data"]["output"]
                return result_state
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_workflow())
            
            st.write("3. Synthesizing citations & confidence scoring...")
            status_container.update(label="Analysis Complete!", state="complete", expanded=False)
            
        st.divider()
        st.subheader("Analysis Response")

        # ── Data Source Banner ── shown BEFORE the answer, never hidden ──
        data_source = result.get("data_source", "unknown")
        if data_source == "live_mcp":
            st.success("🟢 **Data Source: Live Razorpay MCP** — answer is based on your real account data")
        elif data_source == "synthetic":
            st.warning("🟡 **Data Source: Synthetic Fallback** — answer is based on generated test data, NOT your real Razorpay account")
        elif data_source == "error":
            st.error("🔴 **Data Source: Error** — could not connect to Razorpay MCP. Check your credentials or set USE_SYNTHETIC=true")
        else:
            st.info(f"ℹ️ Data Source: {data_source}")

        # Display response
        st.markdown(result.get("final_answer", ""))

        # Display Metadata
        st.divider()
        m_col1, m_col2, m_col3 = st.columns(3)

        with m_col1:
            conf = result.get("confidence", "low").upper()
            if conf == "HIGH":
                st.success(f"Confidence: {conf}")
            elif conf == "MEDIUM":
                st.warning(f"Confidence: {conf}")
            else:
                st.error(f"Confidence: {conf}")
                
        with m_col2:
            route = result.get("route", "N/A")
            st.info(f"Routed Agent: {route}")
            
        with m_col3:
            cited = result.get("cited_records", [])
            st.markdown(f"**Cited Records ({len(cited)}):**")
            if cited:
                st.write(", ".join([f"`{c}`" for c in cited]))
            else:
                st.write("None")
                
        exceptions = result.get("exceptions", [])
        if exceptions:
            st.error(f"Exceptions / Unresolved: {', '.join(exceptions)}")

        st.divider()
        with st.expander("🔍 Detailed Execution State (JSON Log)"):
            # Clean up the messages list for JSON serialization (LangChain objects might not be JSON serializable out of the box)
            # but we can just use string representation or dictionary if it fails
            import json
            try:
                # result is an AgentState dict
                st.json(result)
            except Exception as e:
                st.write(str(result))

with col_detail:
    st.subheader("💡 Sample Test Queries")
    st.markdown("""
    - **Deduction Anomaly:**  
      *Why are the fees higher for settlement setl_000001?*
      
    - **Timing Delay:**  
      *Was settlement setl_000002 delayed beyond the expected processing timeframe?*
      
    - **Status Lookup:**  
      *What is the status and UTR number of settlement setl_000009?*
      
    - **Aggregation:**  
      *How many settlements were processed in the dataset?*
    """)
