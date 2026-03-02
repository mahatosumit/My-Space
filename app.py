import time
import os
import sqlite3
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="Uni-Doc-Intel",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate CSS
st.markdown("""
<style>
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        border-bottom: 2px solid #0056b3;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .source-box {
        font-size: 0.85em;
        background-color: rgba(0, 86, 179, 0.05);
        padding: 10px;
        border-left: 4px solid #0056b3;
        margin-top: 10px;
        border-radius: 4px;
    }
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        border: 1px solid #0056b3;
        color: #0056b3;
        background-color: transparent;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0056b3;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

from hardware.device_manager import get_device
from processor.hybrid_rag import HybridRAGBuilder
from processor.llm_brain import LLMBrain
import config

# Cache offline instances so they don't reload every click
@st.cache_resource(show_spinner="Booting Offline Knowledge Cores...")
def load_rag_backend():
    device_info = get_device()
    rag = HybridRAGBuilder(device_info)
    llm = LLMBrain() # Requires GGuf to exist
    return rag, llm

@st.cache_data(ttl=30)
def get_dashboard_metrics():
    metrics = {"docs": 0, "chunks": 0, "vectors": 0}
    try:
        conn = sqlite3.connect(config.SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        metrics["chunks"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM processed_files")
        metrics["docs"] = cursor.fetchone()[0]
        conn.close()
    except:
        pass
    
    # Try FAISS metric
    try:
        import faiss
        if os.path.exists(config.FAISS_INDEX_PATH):
            index = faiss.read_index(config.FAISS_INDEX_PATH)
            metrics["vectors"] = index.ntotal
    except:
        pass
        
    return metrics

# Render Sidebar
with st.sidebar:
    st.markdown("### Uni-Doc-Intel")
    st.markdown("_Enterprise Offline Knowledge_")
    st.markdown("---")
    
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.markdown("### Vault Telemetry")
    metrics = get_dashboard_metrics()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", metrics.get("docs", 0))
    with col2:
        st.metric("Vector Size", metrics.get("vectors", 0))
        
    st.caption(f"Network isolated shards: {metrics.get('chunks', 0):,}")

# Render Main Layout
st.markdown("<h1 class='main-header'>Uni-Doc-Intel Assistant</h1>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View Cited Logic"):
                for src in message["sources"]:
                    st.markdown(f"""
                    <div class="source-box">
                    <strong>📄 {src['filename']}</strong><br>
                    <strong>ID:</strong> Shard #{src['chunk_id']}<br>
                    <em>{src['content'][:250]}...</em>
                    </div>
                    """, unsafe_allow_html=True)

# Process User Chat Input
if prompt := st.chat_input("Query documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke Logic
    rag_system, llm_system = load_rag_backend()

    with st.chat_message("assistant"):
        with st.spinner("Executing FAISS+BM25 Hybrid Search & Re-Ranking..."):
            retrieval_start = time.time()
            contexts = rag_system.retrieve(query=prompt, top_k=4)
            search_latency = time.time() - retrieval_start
            
        with st.spinner("Synthesizing answer securely on CPU..."):
             answer = llm_system.run_rag_inference(prompt_instruction=prompt, contexts=contexts)
        
        ui_response = f"{answer}\n\n`Latencies -> Search/Re-Rank: {search_latency:.3f}s`"
        st.markdown(ui_response)
        
        if contexts:
            with st.expander("View Cited Logic"):
                for src in contexts:
                    st.markdown(f"""
                    <div class="source-box">
                    <strong>📄 {src['filename']}</strong><br>
                    <strong>ID:</strong> Shard #{src['chunk_id']}<br>
                    <em>{src['content'][:250]}...</em>
                    </div>
                    """, unsafe_allow_html=True)
                    
        st.session_state.messages.append({"role": "assistant", "content": ui_response, "sources": contexts})
