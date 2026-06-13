# langchain_version.py — LangChain RAG variant of the Log Intelligence Assistant
#
# This file replicates the same RAG pipeline as query.py using LangChain components.
# Same logic. Same ChromaDB collection. Same Gemini model.
# The difference: every step is a swappable LangChain component.
#
# Pipeline:
#   User query
#     → Retriever (ChromaDB via LangChain wrapper)
#     → ChatPromptTemplate (structured prompt with context + question)
#     → ChatGoogleGenerativeAI (Gemini via LangChain wrapper)
#     → StrOutputParser (clean string output)
#
# Why this exists:
#   query.py hard-codes every step — ChromaDB calls, prompt formatting, Gemini calls.
#   This version declares the pipeline as composable components using LCEL (|).
#   Swapping the LLM, vector store, or prompt requires changing one line each.

import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# API key from environment — never hardcoded
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable not set.\n"
        "Run: export GEMINI_API_KEY=your_key_here"
    )

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "log_intelligence"
TOP_K = 3  # Number of chunks to retrieve per query

# ── COMPONENT 1: EMBEDDINGS ───────────────────────────────────────────────────

# LangChain wrapper around the same Gemini embedding model used in ingest.py
# Must match ingest.py exactly — query vectors and document vectors must come
# from the same model or cosine similarity scores are meaningless
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)

# ── COMPONENT 2: RETRIEVER ────────────────────────────────────────────────────

# LangChain wrapper around the existing ChromaDB collection built by ingest.py
# persist_directory must point to the same chroma_db folder
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

# as_retriever() turns the vector store into a standard LangChain retriever
# search_kwargs={"k": TOP_K} means: return top 3 most similar chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# ── COMPONENT 3: PROMPT TEMPLATE ──────────────────────────────────────────────

# ChatPromptTemplate replaces the build_prompt() function in query.py
# {context} will be filled with retrieved chunks
# {question} will be filled with the user's query
# Same anti-hallucination instruction as query.py — answers must come from context only
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an incident analysis assistant for an operations team.
Your job is to analyze system logs and runbook entries to produce structured incident summaries.
Answer ONLY using the context provided below.
If the context does not contain enough information to answer, say so explicitly.
Do not use your training data. Do not guess.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Produce your answer in this exact format:
- Incident summary: what happened in one sentence
- Affected service: which service is involved
- Timeline: when did it start
- Root cause: what caused it based on the logs
- Recommended action: what the ops team should do next"""
    ),
    (
        "human",
        "{question}"
    )
])

# ── COMPONENT 4: LLM ──────────────────────────────────────────────────────────

# LangChain wrapper around Gemini — identical model to query.py
# temperature=0.1 keeps answers factual and grounded in retrieved evidence
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.1
)

# ── COMPONENT 5: OUTPUT PARSER ────────────────────────────────────────────────

# StrOutputParser extracts clean text from the LLM response object
# Without this, chain.invoke() returns an AIMessage object — not a plain string
parser = StrOutputParser()

# ── HELPER: FORMAT RETRIEVED DOCUMENTS ───────────────────────────────────────

def format_docs(docs):
    """
    Takes a list of LangChain Document objects returned by the retriever
    and joins their text content into a single context block.
    This is what fills the {context} slot in the prompt template.
    """
    return "\n\n".join(
        f"[Chunk {i+1}] Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

# ── THE CHAIN — LCEL PIPELINE ─────────────────────────────────────────────────

# This is the entire RAG pipeline declared in one expression using LCEL (|).
#
# How it works step by step:
#   1. {"context": ..., "question": RunnablePassthrough()}
#      → Takes the user query, runs it through the retriever, formats the docs
#      → Passes the question through unchanged using RunnablePassthrough
#      → Produces a dict: {"context": "<formatted chunks>", "question": "<user query>"}
#
#   2. | prompt
#      → Fills the {context} and {question} slots in the ChatPromptTemplate
#      → Produces a formatted ChatPromptValue ready for the LLM
#
#   3. | llm
#      → Sends the formatted prompt to Gemini
#      → Returns an AIMessage object
#
#   4. | parser
#      → Extracts the plain text string from the AIMessage
#      → Returns a clean string — the final incident summary

chain = (
    {
        # retriever runs the query against ChromaDB, format_docs converts results to text
        "context": retriever | format_docs,
        # RunnablePassthrough passes the original query string through unchanged
        "question": RunnablePassthrough()
    }
    | prompt   # fills {context} and {question} in the template
    | llm      # calls Gemini with the formatted prompt
    | parser   # extracts clean string from the response
)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Log Intelligence Assistant — LangChain Version")
    print("=" * 60)
    print("RAG pipeline: ChromaDB retriever → Gemini → StrOutputParser")
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
            # Show what was retrieved before generating
            # This is useful for debugging — see what chunks informed the answer
            print("\n--- Retrieving chunks ---")
            retrieved_docs = retriever.invoke(query)
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get("source", "unknown")
                print(f"\n[Chunk {i+1}] Source: {source}")
                print(doc.page_content[:200])  # First 200 chars as preview

            # Run the full chain — retrieve + prompt + generate + parse
            print("\n--- Generated Answer ---")
            answer = chain.invoke(query)
            print(answer)

        except Exception as e:
            # Catch API errors, ChromaDB errors, or missing collection errors
            print(f"Error: {str(e)}")
            print("Check: is chroma_db present? Run ingest.py first if not.")

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()