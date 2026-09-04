# 🤖 Agentic RAG Research Assistant

An **Agentic Retrieval-Augmented Generation (RAG)** application that enables users to upload documents, retrieve relevant information, perform calculations, and research current topics through an agentic workflow.

The system combines **LangGraph, ChromaDB, Sentence Transformers, and LLM tool calling** to determine how a user's query should be handled.

---

## 🚀 Project Overview

Traditional RAG systems generally follow:

> Query → Retrieve Documents → Generate Answer

This project extends that approach with an **agentic workflow**:

> Query → Retrieve → Route → Use Tool if Required → Generate Grounded Answer

The agent can:

- Retrieve information from uploaded documents
- Answer questions using retrieved context
- Perform mathematical calculations safely
- Research current/recent information using a web-search tool
- Provide source/page information for retrieved document content
- Run locally using Ollama with **$0 API cost**

---

## ✨ Key Features

- 📄 **Multi-format document ingestion**
  - PDF
  - TXT
  - Markdown

- ✂️ **Recursive text chunking**
  - Configurable chunk size
  - Chunk overlap for contextual continuity

- 🔎 **Semantic retrieval**
  - Sentence Transformers embeddings
  - ChromaDB vector database
  - Persistent local vector store

- 🧠 **Agentic workflow**
  - Implemented with LangGraph
  - Retrieval and tool routing
  - Conditional execution
  - Final response generation

- 🧮 **Safe calculator tool**
  - Python AST-based arithmetic evaluation
  - Avoids raw `eval()`

- 🌐 **Web research tool**
  - Handles current/recent information requests

- 📌 **Source attribution**
  - Retrieved document content
  - Page information
  - Similarity/retrieval information

- 🤖 **Local LLM inference**
  - Qwen3 4B through Ollama
  - No API cost for local inference

- ☁️ **Cloud LLM support**
  - Optional Groq backend
  - Designed for cloud deployment

- 🖥️ **Streamlit interface**
  - Document upload
  - Knowledge-base creation
  - Query interface
  - Agent mode selection

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │     Streamlit UI     │
                    └──────────┬───────────┘
                               │
                         User Question
                               │
                    ┌──────────▼───────────┐
                    │   LangGraph Agent    │
                    │                      │
                    │ Retrieve → Route     │
                    │ → Tool → Generate    │
                    └──────┬────────┬──────┘
                           │        │
              ┌────────────▼───┐ ┌──▼─────────────┐
              │    ChromaDB    │ │  Agent Tools   │
              │                │ │                │
              │ MiniLM         │ │ Calculator     │
              │ Embeddings     │ │ Web Search     │
              └────────────────┘ └──────┬─────────┘
                                        │
                              ┌─────────▼─────────┐
                              │     LLM Layer     │
                              │                   │
                              │ Ollama / Groq     │
                              └───────────────────┘
