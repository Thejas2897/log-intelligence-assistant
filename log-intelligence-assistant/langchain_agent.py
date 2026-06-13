# langchain_agent.py — LangChain agent with one tool: ChromaDB log search
#
# LangChain 1.x uses create_agent() — a StateGraph-based agent that
# automatically handles tool calls in a loop until it has a final answer.
#
# Pipeline:
#   User query
#     → Agent (LLM with tool-calling loop)
#     → [if needed] search_log_store tool (queries ChromaDB)
#     → Agent observes result, decides to answer or call again
#     → Final answer returned
#
# Interview answer this file earns:
#   "I have built a LangChain agent with tool use" — not "I understand agents conceptually"

import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain.agents import create_agent

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

# ── EMBEDDINGS + VECTOR STORE ─────────────────────────────────────────────────

# Same embedding model as ingest.py — must match or similarity scores break
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)

# Connect to the existing ChromaDB collection built by ingest.py
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# ── THE TOOL ──────────────────────────────────────────────────────────────────

# @tool decorator turns a plain Python function into a LangChain tool.
# The agent reads the docstring to decide WHEN to call this tool.
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

# ── THE AGENT ─────────────────────────────────────────────────────────────────

# create_agent() in LangChain 1.x builds a StateGraph agent.
# It takes a model string, tools list, and system prompt.
# The agent automatically loops — call tool → observe → decide → answer.
agent = create_agent(
    model="google_genai:gemini-2.5-flash",
    tools=[search_log_store],
    system_prompt=(
        "You are an incident analysis assistant for an operations team. "
        "Use the search_log_store tool to retrieve relevant log entries and runbook "
        "procedures before answering. Always ground your answer in retrieved evidence. "
        "Structure your final answer as:\n"
        "- Incident summary: what happened in one sentence\n"
        "- Affected service: which service is involved\n"
        "- Root cause: what caused it based on the retrieved logs\n"
        "- Recommended action: what the ops team should do next"
    )
)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Log Intelligence Assistant — Agent Version")
    print("=" * 60)
    print("Agent tools: search_log_store (ChromaDB)")
    print("Type your query and press Enter.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Query: ").strip()

        if query.lower() == "exit":
            print("Exiting.")
            break

        if not query:
            continue

        try:
            # LangChain 1.x agent takes messages format
            inputs = {"messages": [{"role": "user", "content": query}]}

            # stream() yields updates at each step — tool calls and final answer
            print("\n--- Agent reasoning ---")
            final_answer = ""
            for chunk in agent.stream(inputs, stream_mode="updates"):
                # Each chunk is a dict of node_name -> state update
                for node, update in chunk.items():
                    messages = update.get("messages", [])
                    for msg in messages:
                        # Tool call messages show what the agent decided to do
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"[Tool call] {tc['name']}({tc['args']})")
                        # Tool result messages show what the tool returned
                        elif hasattr(msg, "name") and msg.name:
                            print(f"[Tool result] {msg.content[:200]}...")
                        # Final AI message with no tool calls is the answer
                        elif hasattr(msg, "content") and msg.content:
                            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                                 # Extract plain text from content — may be a list of blocks or a string
                                if isinstance(msg.content, list):
                                    final_answer = " ".join(
                                        block.get("text", "") for block in msg.content
                                        if isinstance(block, dict) and block.get("type") == "text"
                                    )
                            else:
                                final_answer = msg.content
            print("\n--- Final Answer ---")
            print(final_answer)

        except Exception as e:
            print(f"Agent error: {str(e)}")
            print("Check: is chroma_db present? Run ingest.py first if not.")

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()