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
    """Return a Chroma vector store pointing at CHROMA_DIR.

    Ensures the underlying Chroma collection is configured to use cosine
    (hnsw:space = 'cosine') so distance values returned by Chroma match the
    code's expectation of cosine-style distances.
    """
    # Create/connect a chromadb client that uses the same persist directory and
    # ensure the collection exists with the correct HNSW space metadata. This
    # avoids accidentally creating a collection that uses L2 distance.
    settings = ChromaSettings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR)
    client = chromadb.Client(settings=settings)

    # Ask chroma to create the collection with cosine HNSW space if it doesn't exist.
    try:
        client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
    except Exception:
        # Be tolerant: if the client or collection already exists with other settings
        # this call may raise — in that case continue and rely on the existing store.
        pass

    # Return the LangChain Chroma wrapper pointed at the same persist directory.
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
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
        # Chroma may be using either a "cosine"-style distance (distance = 1 - cos)
        # or an L2 (Euclidean) distance depending on how the collection was created.
        # When embeddings are normalized (normalize_embeddings=True) the relation
        # between L2 and cosine is: ||u-v||^2 = 2 - 2*cos -> cos = 1 - 0.5 * ||u-v||^2
        # To be robust we compute both conversions and take the one that yields the
        # higher similarity (after clipping to [0,1]). This handles collections
        # created with either metric.
        try:
            d = float(distance)
        except Exception:
            d = 1.0

        # Option A: assume Chroma returned cosine-style distance (distance = 1 - cos)
        cos_from_cosine = 1.0 - d
        # Option B: assume Chroma returned L2 distance between normalized vectors
        cos_from_l2 = 1.0 - 0.5 * (d ** 2)

        similarity = max(0.0, min(1.0, max(cos_from_cosine, cos_from_l2)))

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
