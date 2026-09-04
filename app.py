import os
from pathlib import Path

import streamlit as st

from agent import run_agent, model_status
from rag import build_knowledge_base, collection_stats, clear_knowledge_base, retrieve_context

st.set_page_config(
    page_title="Agentic RAG Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Keep Streamlit from repeatedly inspecting the large Transformers package.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return os.getenv(name, value or default)


GROQ_AVAILABLE = bool(_secret("GROQ_API_KEY"))

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

with st.sidebar:
    st.title("⚙️ Control Panel")

    default_mode = "Cloud — Groq" if GROQ_AVAILABLE else "Local — Ollama"
    mode = st.radio(
        "LLM mode",
        ["Local — Ollama", "Cloud — Groq"],
        index=1 if default_mode.startswith("Cloud") else 0,
        help="Local mode is ideal for development. Cloud mode is intended for a public Streamlit deployment.",
    )

    st.divider()
    st.subheader("📚 Knowledge Base")
    uploads = st.file_uploader(
        "Upload research documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="PDF, TXT and Markdown files. Documents are chunked, embedded and stored in ChromaDB.",
    )

    if uploads and st.button("Build / update knowledge base", type="primary", use_container_width=True):
        with st.spinner("Chunking and embedding documents..."):
            try:
                stats = build_knowledge_base(uploads)
                st.success(f"Indexed {stats['chunks']} chunks from {stats['documents']} document(s).")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    stats = collection_stats()
    c1, c2 = st.columns(2)
    c1.metric("Chunks", stats["chunks"])
    c2.metric("Sources", stats["sources"])

    if stats["chunks"] and st.button("Clear knowledge base", use_container_width=True):
        clear_knowledge_base()
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()

    st.divider()
    st.subheader("🧠 Agent capabilities")
    st.markdown(
        """
        - **RAG retrieval** — searches your uploaded knowledge base
        - **Calculator** — safely evaluates arithmetic expressions
        - **Web research** — searches the web for current/recent questions
        - **LLM synthesis** — produces a grounded final answer
        - **LangGraph workflow** — routes work through explicit agent nodes
        """
    )

    if mode.startswith("Local"):
        status = model_status()
        if status["ok"]:
            st.success(f"Ollama online · {status['model_count']} model(s) installed")
        else:
            st.warning("Ollama is not reachable. Start Ollama or switch to Cloud — Groq.")
    else:
        if GROQ_AVAILABLE:
            st.success("Groq API key detected")
        else:
            st.warning("No GROQ_API_KEY detected. Add it in Streamlit Secrets for cloud deployment.")

    st.divider()
    st.caption("Portfolio project · RAG + Agentic AI + Local/Cloud LLM")

st.title("🤖 Agentic RAG Research Assistant")
st.caption("A production-style portfolio demo for grounded research over your own documents.")

# Hero metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Architecture", "RAG + Agent")
m2.metric("Vector Store", "ChromaDB")
m3.metric("Embeddings", "MiniLM")
m4.metric("LLM", "Ollama / Groq")

st.info(
    "💡 **Try it:** upload a PDF → build the knowledge base → ask a question. "
    "For an agent demo, ask for a calculation or a current/recent topic."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("🔎 Retrieved sources"):
                for src in message["sources"]:
                    st.markdown(
                        f"**{src['source']}** · page {src.get('page', '?')} · "
                        f"similarity {src['similarity']:.2f}"
                    )

query = st.chat_input("Ask about your documents, calculate something, or request current research…")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Agent is working…"):
            try:
                result = run_agent(query, mode="groq" if mode.startswith("Cloud") else "ollama")
                answer = result["answer"]
                sources = result.get("sources", [])
                route = result.get("route", "generate")

                st.markdown(answer)
                st.caption(f"Agent route: `{route}`")

                if sources:
                    with st.expander("🔎 Retrieved sources", expanded=True):
                        for src in sources:
                            st.markdown(
                                f"**{src['source']}** · page {src.get('page', '?')} · "
                                f"similarity {src['similarity']:.2f}"
                            )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:
                error = str(exc)
                st.error(error)
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ {error}", "sources": []}
                )
