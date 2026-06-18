# RAG Chatbot

Chat with your Obsidian vault using hybrid RAG + DeepSeek LLM synthesis.

Built on top of [obsidian-vault-rag](https://github.com/Selrach84/obsidian-vault-rag) — retrieves the most relevant notes from your second brain, then synthesizes a natural-language answer using DeepSeek.

## Architecture

```
You ──► Web UI ──► FastAPI ──► vault-rag.py (retrieval)
                            └─► DeepSeek API (synthesis, SSE streaming)
```

## Quick Start

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Set DeepSeek API key (if not already set)
export DEEPSEEK_API_KEY=sk-...

# 3. Configure vault path
export RAG_CHATBOT_VAULT="/Volumes/External 500 Gb/OBSIDIAN 5.17.26"

# 4. Start
python server.py

# 5. Open http://localhost:8080
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | — | DeepSeek API key (required) |
| `RAG_CHATBOT_VAULT` | auto-detect | Path to Obsidian vault root |
| `RAG_CHATBOT_PORT` | `8080` | Web UI + API port |
| `RAG_CHATBOT_HOST` | `127.0.0.1` | Bind address |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | Model for synthesis |
| `RAG_CHATBOT_K` | `6` | Number of notes to retrieve |
| `RAG_CHATBOT_HOPS` | `1` | Knowledge graph hop depth |

## How It Works

1. You type a question in the web UI
2. Backend calls `vault-rag.py query` to find the top-K relevant notes (BM25 + semantic + graph signals)
3. Retrieved chunks + context are injected into a system prompt
4. DeepSeek synthesizes an answer grounded in those notes
5. Answer streams back via SSE to the web UI — sources are shown inline

## License

MIT
