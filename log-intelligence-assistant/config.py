# config.py — Central configuration for the Log Intelligence Assistant

import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable not set.\n"
        "Run: export GEMINI_API_KEY=your_key_here"
    )

# gemini-2.5-flash is the current free tier model
GEMINI_MODEL = "gemini-2.5-flash"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "log_intelligence"

# 500 characters fits roughly one log event plus surrounding context
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 3 chunks gives enough context without introducing noise
TOP_K = 3
