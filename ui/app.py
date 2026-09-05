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
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(graph.ainvoke({"question": question}))
            
            st.write("3. Synthesizing citations & confidence scoring...")
            status_container.update(label="Analysis Complete!", state="complete", expanded=False)
            
        st.divider()
        st.subheader("Analysis Response")
        
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
