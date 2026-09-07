import os
import tempfile
from pathlib import Path
from typing import Iterable

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from chromadb.config import Settings as ChromaSettings

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION = "research_documents"
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )


def _load_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise ValueError(f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        path = tmp.name

    try:
        if suffix == ".pdf":
            docs = PyPDFLoader(path).load()
        else:
            docs = TextLoader(path, encoding="utf-8").load()
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
        return docs
    finally:
        Path(path).unlink(missing_ok=True)


def build_knowledge_base(files: Iterable) -> dict:
    all_docs = []
    names = []
    for file in files:
        names.append(file.name)
        all_docs.extend(_load_uploaded_file(file))

    if not all_docs:
        raise ValueError("No readable documents were provided.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    # Avoid accidentally carrying stale copies of the same uploaded source.
    vs = get_vectorstore()
    existing = vs.get(include=["metadatas"])
    ids_to_delete = []
    for idx, meta in zip(existing.get("ids", []), existing.get("metadatas", [])):
        if meta and meta.get("source") in names:
            ids_to_delete.append(idx)
    if ids_to_delete:
        vs.delete(ids=ids_to_delete)

    vs.add_documents(chunks)
    return {"documents": len(names), "chunks": len(chunks)}


def retrieve_context(query: str, k: int = 4):
    vs = get_vectorstore()
    pairs = vs.similarity_search_with_score(query, k=k)
    results = []
    for doc, distance in pairs:
        # With normalized embeddings and Chroma distance, lower is better.
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        results.append(
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "?"),
                "similarity": similarity,
            }
        )
    return results


def collection_stats() -> dict:
    try:
        vs = get_vectorstore()
        data = vs.get(include=["metadatas"])
        metas = data.get("metadatas", [])
        sources = {m.get("source") for m in metas if m and m.get("source")}
        return {"chunks": len(metas), "sources": len(sources)}
    except Exception:
        return {"chunks": 0, "sources": 0}


def clear_knowledge_base():
    vs = get_vectorstore()
    data = vs.get()
    ids = data.get("ids", [])
    if ids:
        vs.delete(ids=ids)