# langgraph_agent.py — Explicit LangGraph agent with two tools and conversational memory
#
# Architecture:
#   AgentState (TypedDict with add_messages reducer + iteration_count)
#     → agent_node (LLM with two tools bound via tool-calling)
#     → conditional_edge (tool_calls present? → tool_node; else → END)
#     → tool_node (ToolNode executes chosen tool deterministically)
#     → loops back to agent_node
#
# Two tools:
#   search_log_store  — queries ChromaDB (same retriever as langchain_agent.py)
#   escalate_to_human — stub action, simulates ticket creation
#
# Conversational memory:
#   MemorySaver checkpointer + thread_id in config
#   Follow-up questions resolve using prior turn context, not just single-query
#
# LangSmith tracing:
#   Set env vars before running — see instructions in main() below
#
# Interview answer this file earns:
#   "I built a LangGraph StateGraph agent with explicit nodes, conditional edges,
#    a bounded iteration guard rail, thread-scoped conversational memory, and
#    LangSmith tracing on the real multi-turn loop."

import os
from typing import Annotated

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from typing_extensions import TypedDict

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable not set.\n"
        "Run: export GEMINI_API_KEY=your_key_here"
    )

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "log_intelligence"
TOP_K = 3
MAX_ITERATIONS = 3   # guard rail — force END if agent loops beyond this

# ── EMBEDDINGS + VECTOR STORE ─────────────────────────────────────────────────
# Same embedding model and collection as ingest.py — must match or similarity breaks
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# ── TOOLS ─────────────────────────────────────────────────────────────────────
# Tool 1: retrieval — wraps the existing ChromaDB retriever
# The agent reads the docstring to decide when to call this tool
@tool
def search_log_store(query: str) -> str:
    """
    Search the system log store and runbook documentation for information
    relevant to the query. Use this tool when the question involves:
    - system incidents or errors
    - service failures or anomalies
    - operational runbook procedures
    - log analysis or incident timelines
    Returns the most relevant log entries and runbook sections.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant logs or runbook entries found for this query."
    results = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        results.append(f"[Result {i+1}] Source: {source}\n{doc.page_content}")
    return "\n\n".join(results)


# Tool 2: escalation — stub action simulating ticket creation
# In a real system this would call PagerDuty, JIRA, or ServiceNow
@tool
def escalate_to_human(reason: str) -> str:
    """
    Escalate this incident to a human operator when:
    - the root cause cannot be determined from available logs
    - the issue requires physical intervention
    - the severity is critical and automated resolution is not safe
    Creates an escalation ticket and notifies the on-call engineer.
    """
    print(f"\n[ESCALATION] Ticket created — reason: {reason}")
    return (
        f"Escalation ticket created. On-call engineer notified.\n"
        f"Reason: {reason}\n"
        f"Ticket ID: ESC-{abs(hash(reason)) % 10000:04d}"
    )

TOOLS = [search_log_store, escalate_to_human]

# ── STATE ──────────────────────────────────────────────────────────────────────
# AgentState is the single source of truth persisted across all nodes.
#
# messages field:
#   Annotated with add_messages reducer — deduplicates by message id instead of
#   naively appending. Without this, the loop would bloat with duplicate messages
#   each time the agent node re-runs. This is the core reason reducers exist.
#
# iteration_count field:
#   No reducer annotation → default overwrite semantics.
#   Each node returning {"iteration_count": n} replaces the field entirely.
#   Used by the conditional edge to enforce the MAX_ITERATIONS guard rail.

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration_count: int

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)

# Bind tools to the LLM using native tool-calling (not JSON-mode prompting).
# This registers the tool schemas so the model can emit structured tool_calls
# in its response — the ToolNode reads those and executes them deterministically.
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = (
    "You are an incident analysis assistant for an operations team. "
    "Use the search_log_store tool to retrieve relevant log entries and runbook "
    "procedures before answering. If the root cause cannot be determined from "
    "available logs or the issue requires human judgment, use escalate_to_human. "
    "Always ground your answer in retrieved evidence. "
    "Structure your final answer as:\n"
    "- Incident summary: what happened in one sentence\n"
    "- Affected service: which service is involved\n"
    "- Root cause: what caused it based on the retrieved logs\n"
    "- Recommended action: what the ops team should do next"
)

# ── NODES ─────────────────────────────────────────────────────────────────────
def agent_node(state: AgentState) -> dict:
    """
    The decision-maker node. Reads the full State (all messages so far),
    calls the LLM with tools bound, and returns the new AI message.

    The LLM decides whether to:
    (a) emit a tool_call → ToolNode will execute it next
    (b) emit a plain text response → conditional edge routes to END

    Also increments iteration_count so the guard rail can detect runaway loops.
    """
    # Prepend system message every time — the LLM has no memory of its own
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1
    }

# ToolNode is a built-in LangGraph node that:
# 1. Reads the last AIMessage's tool_calls field
# 2. Executes each named tool with its arguments
# 3. Returns ToolMessage results back into state
# It is deterministic — it does NOT call the LLM, it only executes what the LLM decided.
tool_node = ToolNode(TOOLS)

# ── CONDITIONAL EDGE ──────────────────────────────────────────────────────────
def route_after_agent(state: AgentState) -> str:
    """
    Router function for the conditional edge after agent_node.

    Reads STRUCTURED state fields — never raw text.
    Two routing conditions checked in order:

    1. Guard rail: if iteration_count >= MAX_ITERATIONS, force END regardless
       of whether tool_calls are present. Prevents infinite loops.

    2. Tool calls: if the last message has tool_calls, route to tool_node
       so the tools can execute.

    3. Otherwise: route to END — the agent has its answer.
    """
    # Guard rail check first — always
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        print(f"\n[Guard rail] Max iterations ({MAX_ITERATIONS}) reached — forcing END.")
        return END

    last_message = state["messages"][-1]

    # Check for tool calls in a way that's safe for both AIMessage and other types
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"

    return END

# ── GRAPH CONSTRUCTION ────────────────────────────────────────────────────────
def build_graph():
    """
    Build and compile the StateGraph with MemorySaver checkpointing.

    Graph structure:
        START → agent_node → (conditional) → tool_node → agent_node (loop)
                                           → END

    MemorySaver persists State after each node, keyed by thread_id.
    This is what makes multi-turn conversation possible:
    - Turn 1: State saved at checkpoint after final agent response
    - Turn 2: Same thread_id → State loaded, prior messages included automatically
    - Follow-up resolves correctly without re-stating context
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("agent_node", agent_node)
    graph.add_node("tool_node", tool_node)

    # Entry point
    graph.set_entry_point("agent_node")

    # Conditional edge from agent_node — reads tool_calls + iteration_count
    graph.add_conditional_edges(
        "agent_node",
        route_after_agent,
        {
            "tool_node": "tool_node",
            END: END
        }
    )

    # After tool execution, always return to agent_node
    # The agent reads the tool result and decides: answer or call another tool
    graph.add_edge("tool_node", "agent_node")

    # Compile with MemorySaver — enables thread_id-scoped conversational memory
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    """
    Interactive multi-turn loop demonstrating conversational memory.

    LangSmith tracing — export these before running:
        export LANGCHAIN_TRACING_V2=true
        export LANGCHAIN_API_KEY=your_langsmith_key
        export LANGCHAIN_PROJECT=log-intelligence-assistant

    Thread ID is fixed per session so follow-up questions resolve correctly.
    Change THREAD_ID between sessions to start a fresh conversation.
    """
    print("=" * 60)
    print("Log Intelligence Assistant — LangGraph Agent")
    print("=" * 60)
    print(f"Tools: search_log_store, escalate_to_human")
    print(f"Memory: MemorySaver (thread_id-scoped)")
    print(f"Guard rail: max {MAX_ITERATIONS} iterations per turn")
    print(f"LangSmith tracing: {'ON' if os.getenv('LANGCHAIN_TRACING_V2') == 'true' else 'OFF (set LANGCHAIN_TRACING_V2=true to enable)'}")
    print()
    print("Type your query and press Enter.")
    print("Type 'new' to start a fresh conversation thread.")
    print("Type 'exit' to quit.\n")

    graph = build_graph()

    # Fixed thread_id — same thread persists memory across turns in one session
    thread_id = "ops-session-001"

    while True:
        user_input = input("Query: ").strip()

        if user_input.lower() == "exit":
            print("Exiting.")
            break

        if user_input.lower() == "new":
            import uuid
            thread_id = f"ops-session-{uuid.uuid4().hex[:8]}"
            print(f"New conversation thread started: {thread_id}\n")
            continue

        if not user_input:
            continue

        # Config carries thread_id — this is how MemorySaver scopes checkpoints
        config = {"configurable": {"thread_id": thread_id}}

        # Input shape: messages list — new HumanMessage added each turn
        # Prior turns are loaded automatically from the checkpoint by thread_id
        inputs = {
            "messages": [HumanMessage(content=user_input)],
            "iteration_count": 0   # reset per turn — guard rail counts per-turn loops
        }

        try:
            print("\n--- Agent reasoning ---")
            final_answer = ""

            # stream() yields updates node-by-node — lets us log tool calls live
            for chunk in graph.stream(inputs, config=config, stream_mode="updates"):
                for node_name, update in chunk.items():
                    messages = update.get("messages", [])
                    for msg in messages:
                        # Tool call decision — show what the agent chose
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"[Tool call] {tc['name']}({tc['args']})")
                        # Tool result — show snippet of what came back
                        elif hasattr(msg, "name") and msg.name:
                            preview = str(msg.content)[:200]
                            print(f"[Tool result from {msg.name}] {preview}...")
                        # Final answer — AI message with no tool calls
                        elif hasattr(msg, "content") and msg.content:
                            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                                if isinstance(msg.content, list):
                                    # Gemini sometimes returns content as a list of blocks
                                    final_answer = " ".join(
                                        block.get("text", "") for block in msg.content
                                        if isinstance(block, dict) and block.get("type") == "text"
                                    )
                                else:
                                    final_answer = str(msg.content)

            print("\n--- Final Answer ---")
            print(final_answer if final_answer else "(no answer generated)")

        except Exception as e:
            print(f"\nAgent error: {str(e)}")
            print("Check: is chroma_db present? Run ingest.py first if not.")

        print(f"\n[Thread: {thread_id}]")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
