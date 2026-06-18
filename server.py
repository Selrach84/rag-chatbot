"""
vault-chat — Web UI chatbot for vault-rag + DeepSeek synthesis.

RAG retrieval from obsidian-vault-rag, LLM synthesis via DeepSeek API.
Streams answers to a web chat UI via SSE.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import re
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn

# ── Config ────────────────────────────────────────────────────────────
VAULT_PATH = os.environ.get(
    "VAULT_CHAT_VAULT",
    "/Volumes/External 500 Gb/OBSIDIAN 5.17.26",
)
VAULT_RAG = os.path.join(VAULT_PATH, "vault-rag.py")
RAG_CLIENT = os.path.join(VAULT_PATH, "rag_client.py")
PORT = int(os.environ.get("VAULT_CHAT_PORT", "8080"))
HOST = os.environ.get("VAULT_CHAT_HOST", "127.0.0.1")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_BASE = "https://api.deepseek.com"
K_DEFAULT = int(os.environ.get("VAULT_CHAT_K", "6"))
HOPS_DEFAULT = int(os.environ.get("VAULT_CHAT_HOPS", "1"))

HERE = Path(__file__).parent

app = FastAPI(title="Vault Chat")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── DeepSeek client ──────────────────────────────────────────────────
DEEPSEEK_CLIENT = httpx.AsyncClient(
    base_url=DEEPSEEK_BASE,
    headers={
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    },
    timeout=60.0,
)

# ── Vault-RAG query ───────────────────────────────────────────────────
def vault_rag_query(q: str, k: int = K_DEFAULT, hops: int = HOPS_DEFAULT) -> dict:
    """Query vault-rag and return parsed results."""
    # Prefer warm daemon via rag_client.py if available
    if os.path.exists(RAG_CLIENT):
        try:
            result = subprocess.run(
                [sys.executable, RAG_CLIENT, q, "--k", str(k), "--hops", str(hops)],
                capture_output=True, text=True, timeout=15, cwd=VAULT_PATH,
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout.strip())
        except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as e:
            print(f"[vault-rag] rag_client fallback: {e}")

    # Fallback to direct query
    try:
        result = subprocess.run(
            [sys.executable, VAULT_RAG, "query", q, "--k", str(k), "--hops", str(hops), "--agent"],
            capture_output=True, text=True, timeout=30, cwd=VAULT_PATH,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout.strip())
        if result.stderr:
            print(f"[vault-rag] stderr: {result.stderr}")
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as e:
        print(f"[vault-rag] query failed: {e}")

    return {"results": [], "connected": [], "total_notes": 0, "warm": False}


def format_context(rag_result: dict) -> str:
    """Format vault-rag results into a context block for the LLM."""
    results = rag_result.get("results", [])
    if not results:
        return ""

    ctx_parts = []
    ctx_parts.append("# Retrieved Notes from Personal Knowledge Base\n")
    for i, r in enumerate(results, 1):
        path = r.get("path", "unknown")
        score = r.get("score", 0)
        snippet = r.get("snippet", "")
        tags = r.get("tags", [])

        ctx_parts.append(f"## Source {i}: {path} (relevance: {score:.2f})")
        if tags:
            ctx_parts.append(f"Tags: {', '.join(tags)}")
        if snippet:
            clean = re.sub(r'\s+', ' ', snippet).strip()[:800]
            ctx_parts.append(f"Content: {clean}")
        ctx_parts.append("")

    # Add connected notes (1-hop graph neighbors)
    connected = rag_result.get("connected", [])
    if connected:
        ctx_parts.append("## Related Notes (knowledge graph neighbors):")
        for r in connected[:5]:
            ctx_parts.append(f"- {r.get('path', 'unknown')} (degree: {r.get('degree', 0)})")
        ctx_parts.append("")

    return "\n".join(ctx_parts)


SYSTEM_PROMPT = """You are a helpful AI assistant answering questions based on the user's personal knowledge base (Obsidian vault). You have access to retrieved notes that are most relevant to the question.

## Instructions
1. Answer the user's question based SOLELY on the retrieved notes provided below. If the retrieved notes don't contain enough information, say so clearly.
2. Cite your sources inline using the format `[Source N]` where N is the source number from the context.
3. Be concise, direct, and helpful. Use the user's own notes and terminology.
4. If the question is a greeting or general chat (not needing retrieval), respond naturally.
5. Format responses with proper Markdown for readability."""


def build_messages(rag_result: dict, chat_history: list, user_msg: str) -> list:
    """Build the messages array for the DeepSeek API."""
    context = format_context(rag_result)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add chat history (up to last 10 turns to avoid context blowup)
    for msg in chat_history[-20:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add context as a separate assistant message if there's retrieved content
    if context:
        messages.append({
            "role": "user",
            "content": f"[The following notes from your vault are relevant to this question]\n\n{context}\n\nNow answer this question (cite sources as [Source N]): {user_msg}",
        })
    else:
        # No relevant notes found — just answer from knowledge
        messages.append({
            "role": "user",
            "content": user_msg,
        })

    return messages


async def stream_deepseek(messages: list) -> AsyncGenerator[str, None]:
    """Stream a chat completion from DeepSeek via SSE."""
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    try:
        async with DEEPSEEK_CLIENT.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                yield f"data: {json.dumps({'error': f'API error {resp.status_code}: {error_body.decode()}'})}\n\n"
                return

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload_str = line[6:].strip()
                if payload_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield f"data: {json.dumps({'content': content})}\n\n"
                except json.JSONDecodeError:
                    continue

            yield "data: [DONE]\n\n"
    except httpx.RequestError as e:
        yield f"data: {json.dumps({'error': f'Connection error: {e}'})}\n\n"


# ── Chat history (in-memory, per-session) ─────────────────────────────
chat_sessions: dict[str, list] = {}


@app.get("/api/health")
async def health():
    return {"status": "ok", "vault": VAULT_PATH, "model": DEEPSEEK_MODEL}


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    user_msg = body.get("message", "").strip()
    session_id = body.get("session_id", "default")
    k = body.get("k", K_DEFAULT)
    hops = body.get("hops", HOPS_DEFAULT)

    if not user_msg:
        return JSONResponse({"error": "Message is required"}, status_code=400)

    if not DEEPSEEK_API_KEY:
        return JSONResponse({"error": "DEEPSEEK_API_KEY not configured"}, status_code=500)

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    chat_sessions[session_id].append({"role": "user", "content": user_msg})

    # Step 1: Retrieve from vault-rag
    rag_result = vault_rag_query(user_msg, k=k, hops=hops)
    no_sources = len(rag_result.get("results", [])) == 0

    # Step 2: Build messages with context
    messages = build_messages(rag_result, chat_sessions[session_id], user_msg)

    # Step 3: Stream the response
    async def generate():
        # Send metadata first
        metadata = {
            "rag_results": [
                {
                    "path": r.get("path", ""),
                    "score": r.get("score", 0),
                    "snippet": (re.sub(r'\s+', ' ', r.get("snippet", "")).strip()[:200] if r.get("snippet") else ""),
                    "tags": r.get("tags", []),
                }
                for r in rag_result.get("results", [])
            ],
            "total_notes": rag_result.get("total_notes", 0),
            "warm": rag_result.get("warm", False),
        }
        yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"

        full_response = ""
        async for chunk in stream_deepseek(messages):
            yield chunk
            if chunk.startswith("data: ") and chunk[6:].strip() != "[DONE]":
                try:
                    parsed = json.loads(chunk[6:])
                    if "content" in parsed:
                        full_response += parsed["content"]
                except json.JSONDecodeError:
                    pass

        # Save assistant response to history
        if full_response:
            chat_sessions[session_id].append({"role": "assistant", "content": full_response})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": list(chat_sessions.keys()), "count": len(chat_sessions)}


@app.post("/api/sessions/clear")
async def clear_session(request: Request):
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    session_id = body.get("session_id", "default")
    if session_id in chat_sessions:
        chat_sessions[session_id] = []
    return {"status": "cleared", "session": session_id}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = HERE / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Vault Chat</h1><p>Frontend not found.</p>")


def main():
    print(f"╔══════════════════════════════════════════╗")
    print(f"║        Vault Chat — RAG Chatbot          ║")
    print(f"╠══════════════════════════════════════════╣")
    print(f"║ Vault:   {VAULT_PATH:<34}║")
    print(f"║ Model:   {DEEPSEEK_MODEL:<34}║")
    print(f"║ Port:    http://{HOST}:{PORT:<27}║")
    print(f"║ K/hops:  {K_DEFAULT}/{HOPS_DEFAULT:<31}║")
    print(f"╚══════════════════════════════════════════╝")

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
