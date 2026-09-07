import ast
import json
import os
import operator as op
from typing import TypedDict

import requests
from ddgs import DDGS
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from rag import retrieve_context


class AgentState(TypedDict, total=False):
    query: str
    context: list
    route: str
    tool_output: str
    answer: str
    sources: list
    mode: str


_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_ALLOWED_UNARY = {ast.UAdd: op.pos, ast.USub: op.neg}


def safe_calculate(expression: str):
    tree = ast.parse(expression, mode="eval")

    def visit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](visit(node.operand))
        raise ValueError("Only basic arithmetic is allowed.")

    return visit(tree.body)


def _ollama():
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.1,
    )


def _groq():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            key = None
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to Streamlit Secrets or use Local — Ollama.")
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0.1,
        api_key=key,
    )


def model_status():
    try:
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(f"{base}/api/tags", timeout=2)
        response.raise_for_status()
        return {"ok": True, "model_count": len(response.json().get("models", []))}
    except Exception:
        return {"ok": False, "model_count": 0}


def _needs_web(query: str) -> bool:
    q = query.lower()
    return any(word in q for word in ["latest", "current", "today", "recent", "this week", "news", "online"])


def _looks_like_math(query: str) -> bool:
    q = query.lower().strip()
    if len(q) > 100:
        return False
    return any(char.isdigit() for char in q) and any(char in q for char in "+-*/%")


def _extract_math(query: str) -> str:
    allowed = "0123456789.+-*/()% "
    expression = "".join(ch for ch in query if ch in allowed).strip()
    if not expression or not any(ch.isdigit() for ch in expression):
        raise ValueError("Could not find a safe arithmetic expression.")
    return expression


def route_node(state: AgentState):
    query = state["query"]
    if _needs_web(query):
        route = "web_research"
    elif _looks_like_math(query):
        route = "calculator"
    else:
        route = "rag"
    return {"route": route}


def _route_decision(state: AgentState) -> str:
    # Reads the route chosen by route_node and picks which branch to run next.
    return state["route"]


def retrieve_node(state: AgentState):
    context = retrieve_context(state["query"], k=4)
    return {"context": context, "sources": context}


def tool_node(state: AgentState):
    route = state["route"]
    query = state["query"]

    if route == "calculator":
        expression = _extract_math(query)
        return {"tool_output": f"Calculator result for `{expression}` = {safe_calculate(expression)}"}

    if route == "web_research":
        try:
            results = DDGS().text(query, max_results=5)
            lines = []
            for item in results:
                title = item.get("title", "Untitled")
                body = item.get("body", "")
                href = item.get("href", "")
                lines.append(f"- {title}\n  {body}\n  URL: {href}")
            return {"tool_output": "\n".join(lines) or "No web results found."}
        except Exception as exc:
            return {"tool_output": f"Web search failed: {exc}"}

    return {"tool_output": "Use the retrieved document context."}


def generate_node(state: AgentState):
    context_text = "\n\n".join(
        f"SOURCE: {item['source']} (page {item['page']})\n{item['text']}"
        for item in state.get("context", [])
    )
    prompt = f"""You are a careful research assistant.

User question:
{state['query']}

Retrieved document context:
{context_text or '[No relevant document context found.]'}

Tool output:
{state.get('tool_output', '[No extra tool output.]')}

Rules:
1. Answer directly and concisely.
2. Prefer the retrieved context/tool output over unsupported claims.
3. If the documents do not contain the answer, explicitly say that the provided documents do not establish it.
4. For web results, treat snippets as potentially incomplete and avoid pretending they are verified facts.
5. Do not invent citations. Source names/pages are shown separately by the application.
"""

    llm = _groq() if state.get("mode") == "groq" else _ollama()
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return {"answer": answer}


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("tool", tool_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        _route_decision,
        {
            "rag": "retrieve",
            "calculator": "tool",
            "web_research": "tool",
        },
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("tool", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


GRAPH = _build_graph()


def run_agent(query: str, mode: str = "ollama"):
    result = GRAPH.invoke({"query": query, "mode": mode})
    return {
        "answer": result.get("answer", "No answer generated."),
        "sources": result.get("sources", []),
        "route": result.get("route", "unknown"),
    }