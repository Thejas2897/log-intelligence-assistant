# Log Intelligence Assistant

Natural language querying over system logs and runbooks using RAG.

Given a question like "why did the payment service fail at 2 AM?", the system
retrieves the relevant log entries and runbook context, then generates a
structured incident summary with root cause and recommended actions.

## How it works

1. Ingest — log files and runbook documents are chunked and stored as
   embeddings in ChromaDB
2. Retrieve — a natural language query is embedded and matched against
   stored chunks using vector similarity
3. Generate — the top matching chunks are passed to Gemini with the query,
   which returns a structured incident summary

## Stack

Python 3 · ChromaDB · Gemini API · LangChain

## Structure

    log-intelligence-assistant/
        ingest.py            — chunking, embedding, ChromaDB storage
        query.py             — retrieval, Gemini call, structured output
        config.py            — chunk size, top-k, model, environment config
        langchain_version.py — LangChain variant of the same pipeline
        data/
            sample_logs.txt
            runbook.txt

    tools/
        chatbot.py           — multi-turn terminal chatbot with retry logic
        token_counter.py     — context window usage visualiser

## Design decisions

- ChromaDB for local vector storage — no infrastructure overhead
- Chunk size and top-k configurable in config.py without code changes
- Exponential backoff on all Gemini API calls
- API keys in environment variables only — never hardcoded
