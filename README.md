# 🤖 Agentic RAG Research Assistant

A portfolio-ready **Retrieval-Augmented Generation (RAG) + Agentic AI** application that lets users upload documents and ask grounded questions. It combines local embeddings, ChromaDB retrieval, a LangGraph workflow, safe calculator and web-research tools, and either a local Ollama model or optional Groq cloud inference.

## 🚀 Live Demo

> Add your deployed Streamlit URL here after deployment.

`https://YOUR-APP.streamlit.app`

## ✨ Features

- 📄 PDF / TXT / Markdown ingestion
- ✂️ Recursive chunking with overlap
- 🔎 Semantic retrieval with Sentence Transformers
- 🗄️ Persistent ChromaDB vector store
- 🧠 **LangGraph agent workflow**: retrieve → route → tool → generate
- 🧮 Safe arithmetic tool using Python AST (no raw `eval`)
- 🌐 Web research tool for current/recent questions
- 🤖 Local **Qwen3 4B via Ollama** for $0 local inference
- ☁️ Optional **Groq** backend for public cloud deployment
- 📌 Retrieved source, page and similarity information
- 🔐 API keys kept out of Git with Streamlit Secrets
- 🖥️ Streamlit UI suitable for portfolio demos

## 🏗️ Architecture

```text
                    ┌──────────────────────┐
                    │   Streamlit UI       │
                    └──────────┬───────────┘
                               │
                         User question
                               │
                    ┌──────────▼───────────┐
                    │ LangGraph Workflow   │
                    │ retrieve → route     │
                    │ → tool → generate    │
                    └──────┬────────┬──────┘
                           │        │
                ┌──────────▼───┐ ┌──▼─────────────┐
                │ ChromaDB     │ │ Agent Tools    │
                │ + MiniLM     │ │ Calculator/Web │
                └──────────────┘ └──────┬─────────┘
                                        │
                              ┌─────────▼─────────┐
                              │ Ollama / Groq LLM │
                              └───────────────────┘
```

## 🧪 Local setup — $0 LLM cost

Requires macOS with Ollama installed for the local LLM.

```bash
git clone YOUR_REPO_URL
cd agentic-rag-research-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen3:4b
streamlit run app.py
```

If Streamlit's file watcher causes Transformer warnings, the project already disables the watcher in `.streamlit/config.toml`.

## ☁️ Deploy to Streamlit Community Cloud

The Streamlit Community Cloud app can be connected to a GitHub repository. For a public deployment, use **Cloud — Groq** because your Mac's local Ollama server is not available inside Streamlit's cloud container.

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud and create an app from the repository.
3. Set the main file to `app.py`.
4. In the app's **Advanced settings / Secrets**, add:

```toml
GROQ_API_KEY = "your_key_here"
GROQ_MODEL = "llama-3.1-8b-instant"
```

5. Deploy.

Never commit `secrets.toml`, `.env`, or API keys to GitHub.

## 🎯 Demo script for recruiters

Upload a CV, research paper, policy document, or technical PDF and demonstrate:

1. **Document QA:** “What are the main findings?”
2. **Grounding:** ask for a fact that exists in the document and show its source/page.
3. **Calculator:** “Calculate 125 * 48 + 250.”
4. **Agent routing:** “What are the latest developments in RAG?”
5. Explain that LangGraph routes the request and the final LLM synthesizes the tool/retrieval output.

## 🧰 Tech stack

**Python · Streamlit · LangGraph · LangChain · ChromaDB · Sentence Transformers · Ollama · Qwen3 · Groq · PyPDF · DDGS**

## 📌 Resume bullet

> Built and deployed an agentic RAG research assistant using LangGraph, ChromaDB and Sentence Transformers, with PDF ingestion, semantic retrieval, source attribution, safe calculator/web tools, and local Ollama/Groq LLM backends.

## 🔮 Future improvements

- Reranking for higher retrieval precision
- Hybrid keyword + semantic retrieval
- Streaming token output
- Evaluation with RAGAS / custom retrieval metrics
- Authentication and multi-user workspaces
- Observability with LangSmith

## License

MIT
