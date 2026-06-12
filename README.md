# Log Intelligence Assistant

Natural language querying over system logs and runbooks using RAG.

Given a question like "why did the payment service fail at 2 AM?", the system
retrieves the relevant log entries and runbook context, then generates a
structured incident summary with root cause and recommended actions.

## How it works

1. **Ingest** — log files and runbook documents are chunked and stored as
   embeddings in ChromaDB
2. **Retrieve** — a natural language query is embedded and matched against
   stored chunks using vector similarity
3. **Generate** — the top matching chunks are passed to Gemini with the query,
   which returns a structured incident summary
4. **Stream** — stream_simulator.py feeds live log events into the pipeline
   continuously, returning an incident summary per event

## Stack

Python 3 · ChromaDB · Gemini API · LangChain

## Structure

    log-intelligence-assistant/
        ingest.py              — chunking, embedding, ChromaDB storage
        query.py               — retrieval, Gemini call, structured output
        config.py              — chunk size, top-k, model, environment config
        stream_simulator.py    — live event stream simulation mode
        anomaly_detector.py    — z-score and rolling average anomaly detection
        vllm_simulator.py      — vLLM-style request batching simulation
        langchain_version.py   — LangChain variant of the same pipeline
        FAILURES.md            — documented failure modes and diagnosis paths
        data/
            sample_logs.txt
            runbook.txt
    tools/
        chatbot.py             — multi-turn terminal chatbot with retry logic
        token_counter.py       — context window usage visualiser

## Design decisions

**Why ChromaDB over Pinecone or Weaviate**
ChromaDB runs locally with no infrastructure overhead. For a single-machine
deployment handling thousands of log chunks, it is the right choice. At scale —
millions of chunks across multiple services — a managed vector store like
Pinecone would be the correct upgrade path.

**Why chunk size 500 characters**
Each chunk must contain exactly one complete semantic unit — one log event or
one runbook step. At 500 characters, a typical log line plus surrounding context
fits cleanly. Smaller chunks lose context. Larger chunks introduce noise from
unrelated events and degrade retrieval precision.

**Why top-k 3**
Three chunks provides enough context for Gemini to generate a grounded answer
without overloading the prompt. The constraint is: TOP_K × average chunk tokens
must stay well under the model context limit. At TOP_K=3 and 500-character chunks,
the retrieved context is roughly 400 tokens — safe headroom.

**Anti-hallucination prompt design**
The generation prompt explicitly instructs Gemini to answer only from retrieved
context and say so when context is insufficient. This is the core reliability
mechanism in any production RAG system.

## What I learned breaking this

Three failure modes were deliberately introduced and diagnosed. Full diagnosis
paths are in FAILURES.md.

**Break 1 — Chunk size too small (CHUNK_SIZE = 50)**
The ingest process was killed by the OS. 50-character chunks produce ~96 chunks
from the same data that 500-character chunks handle in 12, each requiring a
separate embedding API call. Failure mode: resource exhaustion and API quota
burn before ingest completes. At production scale this would exhaust budget.

**Break 2 — TOP_K too high (TOP_K = 50)**
On a 12-chunk test set, the query completed but returned all chunks regardless
of relevance. The latent failure: at production scale with 50,000 chunks,
TOP_K=50 would retrieve ~25,000 tokens and overflow the model context window.
Failure is invisible on small datasets and catastrophic at scale.

**Break 3 — Missing environment variable**
Removing API key validation caused a crash deep inside the google.genai library
internals with no actionable error message. Fix: fail-fast validation in
config.py raises a clear error with the exact export command before any network
calls are made.

## How to scale

This single-machine architecture breaks in three places under load:

1. **ChromaDB** — local vector store has no replication. At scale, replace with
   Pinecone, Weaviate, or pgvector behind a load balancer.
2. **Gemini API free tier** — single-threaded, rate-limited. At scale, replace
   with a vLLM-served open-source model or a paid API endpoint with concurrency.
3. **Ingest pipeline** — sequential embedding calls. At scale, parallelize with
   asyncio or a Kafka-backed ingestion worker pool.

## Running locally

```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-key-here"

# Ingest documents
python ingest.py

# Query the pipeline
python query.py

# Run live stream simulation
python stream_simulator.py --events 5 --interval 2
```