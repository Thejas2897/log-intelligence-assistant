# query.py — Stages 4, 5, 6: Retrieve, Generate
# Uses google.genai (current library) for both embeddings and generation

import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types
from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
    TOP_K
)

# ── SETUP ──────────────────────────────────────────────────────────────────

client = genai.Client(api_key=GEMINI_API_KEY)


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Must be identical to ingest.py — query vectors and document vectors
    must come from the same model or similarity scores are meaningless.
    """
    def __init__(self):
        self._client = client

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            result = self._client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=text
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings


# Connect to ChromaDB with the same embedding function used at ingest
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=GeminiEmbeddingFunction()
)


# ── RETRIEVE ───────────────────────────────────────────────────────────────

def retrieve_chunks(query: str, top_k: int = TOP_K):
    """
    Embeds the query using Gemini and retrieves top_k similar chunks.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results["documents"][0], results["metadatas"][0]


# ── BUILD PROMPT ───────────────────────────────────────────────────────────

def build_prompt(query: str, chunks: list[str]) -> str:
    """
    Constructs RAG prompt — system instruction, retrieved context, user query.
    Anti-hallucination instruction constrains LLM to retrieved evidence only.
    """
    context_block = "\n\n".join(
        f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )

    return f"""You are an incident analysis assistant for an operations team.
Your job is to analyze system logs and runbook entries to produce structured incident summaries.
Answer ONLY using the context provided below.
If the context does not contain enough information to answer, say so explicitly.
Do not use your training data. Do not guess.

--- CONTEXT START ---
{context_block}
--- CONTEXT END ---

Question: {query}

Produce your answer in this exact format:
- Incident summary: what happened in one sentence
- Affected service: which service is involved
- Timeline: when did it start
- Root cause: what caused it based on the logs
- Recommended action: what the ops team should do next"""


# ── GENERATE ───────────────────────────────────────────────────────────────

def generate_answer(query: str):
    """
    Full RAG pipeline — retrieve then generate.
    """
    chunks, metadatas = retrieve_chunks(query)

    if not chunks:
        return "No chunks retrieved. Run ingest.py first.", [], []

    prompt = build_prompt(query, chunks)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text, chunks, metadatas
    except Exception as e:
        return f"Gemini API error: {str(e)}", chunks, metadatas


# ── MAIN ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Log Intelligence Assistant — Query Pipeline")
    print("=" * 60)
    print("Type your query and press Enter.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Query: ").strip()

        if query.lower() == "exit":
            print("Exiting.")
            break

        if not query:
            continue

        answer, chunks, metadatas = generate_answer(query)

        print("\n--- Retrieved Chunks ---")
        for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
            print(f"\n[Chunk {i+1}] Source: {meta['source']}")
            print(chunk[:200])

        print("\n--- Generated Answer ---")
        print(answer)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
