# ingest.py — Stage 1, 2, 3: Ingest, Chunk, Embed, Store
# Uses google.genai (current library) and gemini-embedding-001 for embeddings

import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from config import (
    GEMINI_API_KEY,
    CHROMA_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

# ── GEMINI EMBEDDING FUNCTION ──────────────────────────────────────────────

client = genai.Client(api_key=GEMINI_API_KEY)

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function using google.genai and gemini-embedding-001.
    Passed to ChromaDB so all embed operations go through Gemini API.
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
            # result.embeddings is a list of ContentEmbedding objects
            # each has a .values attribute containing the vector
            embeddings.append(result.embeddings[0].values)
        return embeddings


# ── STAGE 1: INGEST ────────────────────────────────────────────────────────

def load_documents(data_dir: str) -> list[dict]:
    """
    Reads all .txt files from the data/ directory.
    Returns a list of dicts with 'filename' and 'text' keys.
    """
    documents = []

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory '{data_dir}' not found.\n"
            "Make sure data/sample_logs.txt and data/runbook.txt exist."
        )

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append({"filename": filename, "text": text})
            print(f"Loaded: {filename} — {len(text)} characters")

    if not documents:
        raise ValueError(f"No .txt files found in '{data_dir}'")

    return documents


# ── STAGE 2: CHUNK ─────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits text into fixed-size chunks with overlap.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ── STAGE 3: EMBED AND STORE ───────────────────────────────────────────────

def get_collection():
    """
    Creates ChromaDB client with Gemini embedding function.
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=GeminiEmbeddingFunction()
    )
    return collection


def embed_and_store(chunks: list[str], filename: str, collection) -> None:
    """
    Stores chunks in ChromaDB using Gemini embeddings.
    upsert is safe to run multiple times — no duplicate chunks created.
    """
    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i}
                 for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=metadatas
    )
    print(f"Stored: {len(chunks)} chunks from {filename}")


# ── MAIN ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Log Intelligence Assistant — Ingest Pipeline")
    print("=" * 60)

    print("\n--- Stage 1: Loading documents ---")
    documents = load_documents("data")
    print(f"Total documents loaded: {len(documents)}")

    print("\n--- Stage 2 and 3: Chunking and storing ---")
    collection = get_collection()

    total_chunks = 0
    for doc in documents:
        chunks = chunk_text(doc["text"])
        embed_and_store(chunks, doc["filename"], collection)
        total_chunks += len(chunks)

    print(f"\nIngest complete — {total_chunks} total chunks stored in ChromaDB")
    print(f"ChromaDB location: {CHROMA_PATH}")
    print(f"Collection name: {COLLECTION_NAME}")
    print("\nRun query.py to start querying the pipeline.")


if __name__ == "__main__":
    main()
