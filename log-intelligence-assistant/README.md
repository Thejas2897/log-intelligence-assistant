# Log Intelligence Assistant

Natural language querying over system logs and runbooks using RAG — extended into an agentic orchestration system with cloud deployment.

Given a question like "why did the payment service fail at 2 AM?", the system retrieves relevant log entries and runbook context, generates a structured incident summary, and — if the root cause cannot be determined from available logs — escalates to a human operator automatically.

---

## What this project covers

Three distinct systems, sharing the same data layer:

### 1. Core RAG pipeline
Vector retrieval over system logs and runbooks using ChromaDB, LangChain LCEL, and the Gemini API.

- Chunking strategy (chunk_size=500, overlap=50, top_k=3) validated against empirical retrieval quality
- Anti-hallucination prompting — answers only from retrieved context, states so when absent
- Three deliberately documented failure modes in FAILURES.md (resource exhaustion, context overflow, missing credentials)
- Fixed-batching vLLM simulator demonstrating the distinction from true continuous batching

### 2. LangGraph Agent — agentic orchestration with conversational memory
An explicit StateGraph agent with two tools, thread-scoped conversational memory, and LangSmith tracing.

- **Two tools:** `search_log_store` (ChromaDB retrieval) and `escalate_to_human` (ticket creation stub)
- **Conditional routing:** `route_after_agent()` reads `tool_calls` from structured State — never raw text
- **Bounded iteration guard rail:** MAX_ITERATIONS=3 prevents runaway loops
- **Conversational memory:** MemorySaver + thread_id — follow-up questions resolve from prior turn context without re-retrieval
- **LangSmith tracing:** full span tree visible per run (agent_node → tool_node → agent_node), token counts per span

Verified two-turn exchange: Turn 1 retrieved and answered a payment service incident (7.33s, 400 → 1.6K tokens across two agent calls). Turn 2 follow-up resolved "What should the ops team do to resolve it?" from checkpointed context without retrieval (2.05s, 1.4K input tokens — the prior conversation injected automatically).

### 3. Serverless deployment — AWS Bedrock + Lambda + API Gateway
The same RAG pattern deployed to the cloud, separately documented in `experiments/aws-deploy/`.

---

## Structure

```
log-intelligence-assistant/
    ingest.py              — chunking, embedding, ChromaDB storage
    query.py               — retrieval, Gemini call, structured output
    config.py              — chunk size, top-k, model, environment config
    langchain_version.py   — LangChain LCEL variant (Runnable pipeline)
    langchain_agent.py     — LangChain ReAct agent with ChromaDB tool
    langgraph_agent.py     — Explicit LangGraph StateGraph agent (two tools,
                             MemorySaver, LangSmith tracing)
    stream_simulator.py    — live event stream simulation mode
    anomaly_detector.py    — z-score and rolling average anomaly detection
    vllm_simulator.py      — fixed-batching simulator (deliberately not
                             continuous batching — the distinction is the point)
    FAILURES.md            — documented failure modes and diagnosis paths
    data/
        sample_logs.txt
        runbook.txt
    experiments/
        aws-deploy/        — serverless deployment (see that folder's README)
```

---

## Stack

**Local pipeline:** Python 3 · ChromaDB · Gemini API (gemini-2.5-flash, gemini-embedding-001) · LangChain LCEL · LangGraph · LangSmith

**Cloud deployment:** AWS Bedrock · AWS Lambda · API Gateway · IAM

---

## Running the LangGraph agent

```bash
# Required environment variables
export GEMINI_API_KEY=your_key_here
export LANGCHAIN_TRACING_V2=true           # optional — enables LangSmith tracing
export LANGCHAIN_API_KEY=your_langsmith_key  # optional
export LANGCHAIN_PROJECT=log-intelligence-assistant  # optional

# Run ingest first if chroma_db does not exist
python3 ingest.py

# Run the agent
python3 langgraph_agent.py
```

Type a query. Use `new` to start a fresh conversation thread. Use `exit` to quit.

The agent will search the log store for operational queries and escalate to a human operator when the issue requires physical intervention or cannot be resolved from available logs.

---

## Design decisions

**Why ChromaDB over Pinecone or Weaviate**
ChromaDB runs locally with no infrastructure overhead and no egress of log data. For a single-machine deployment handling thousands of log chunks under data-sensitivity constraints, it is the correct choice. At scale — millions of chunks across multiple services — a managed vector store with hybrid search (BM25 + cosine) and a cross-encoder re-ranker would be the correct upgrade path.

**Why chunk_size=500, top_k=3**
Validated empirically: 100-token chunks cause answer fragmentation (the context for a single log event splits across non-adjacent retrieval ranks); 800-token chunks recruit more off-topic content into top-ranked results. 500 tokens is the sweet spot for this corpus — one complete log context unit per chunk. Top-k=3 keeps retrieved context to approximately 400 tokens, well within the Gemini context budget while providing enough evidence for a grounded answer. The arithmetic: top_k × chunk_size ≈ retrieved context. Budget this explicitly or it overflows silently.

**Why an explicit StateGraph instead of LangChain's AgentExecutor**
The explicit StateGraph makes the agent loop visible: each node's execution appears as a separate span in LangSmith, each conditional edge's routing decision is inspectable, and the max-iteration guard rail is a first-class check rather than a hidden configuration option. The legacy AgentExecutor is a black box — no per-iteration observability, no mid-loop interruptibility.

**Why MemorySaver over a database-backed checkpointer**
MemorySaver is in-process memory — correct for a single-machine demo, not for a production system where multiple processes serve different users. In production, replace with PostgresSaver (or a DynamoDB-backed equivalent) so any process can resume any thread_id by reading the checkpoint from a shared store. The interface is identical — one line change at `graph.compile(checkpointer=...)`.

**Anti-hallucination prompt design**
The generation prompt explicitly instructs the model to answer only from retrieved context and state so when context is insufficient. This is the core reliability mechanism — without it, the model fills gaps from training data, producing plausible-sounding but ungrounded answers that cannot be verified against the actual logs.

---

## What I learned breaking this

Three failure modes were deliberately introduced and diagnosed. Full diagnosis paths in FAILURES.md.

**Break 1 — Chunk size too small (CHUNK_SIZE = 50)**
Resource exhaustion: 50-character chunks produce ~96 embedding API calls from the same data that 500-character chunks handle in 12. At production scale this exhausts API quota before ingest completes. Detection: rate-limit error from the embedding API, not from the application logic.

**Break 2 — TOP_K too high (TOP_K = 50)**
Silent context overflow: on a small test set the query appears to work, returning all chunks regardless of relevance. At production scale — thousands of chunks, long conversation history — the total prompt exceeds the model context limit and either truncates silently or throws a context-length error. Detection: context-length error at generation time, or degraded answer quality with no error at all.

**Break 3 — Missing API key validation**
Unhelpful internal error: the SDK raises a cryptic authentication exception with no user-facing guidance. Detection: exception traceback with no actionable message. Fix: validate the env var at module load time with an explicit, human-readable error before making any API call.
