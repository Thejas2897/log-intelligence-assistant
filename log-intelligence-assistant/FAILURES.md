# Documented Failures — Log Intelligence Assistant

## Break 1 — Chunk size too small

**Change made:** CHUNK_SIZE reduced from 500 to 50 in config.py

**What happened:** ingest.py was killed by the OS mid-run. Process never completed.

**Diagnosis:** 50-character chunks produce ~96 chunks from the same data that 500-character
chunks handle in 12. Each chunk requires a separate Gemini embedding API call.
The combination of excessive API calls and memory pressure caused the OS to kill the process.

**Two failure modes exposed:**
1. Resource exhaustion — too many API calls, process killed before ingest completes
2. Retrieval degradation — if ingest had completed, 50-char chunks would lose all context
   (a log line split mid-sentence is not retrievable as a coherent unit)

**Fix:** Restored CHUNK_SIZE = 500. Rule: chunk size must be large enough to contain
one complete semantic unit — one log event, one runbook step.

**Interview answer:** "I set chunk size to 50 to test the lower bound. The ingest process
was killed by the OS — I had 96 chunks requiring 96 separate embedding API calls.
That taught me chunk size is not just a retrieval quality decision, it's a resource
and cost decision. At production scale with millions of log lines, this misconfiguration
would exhaust API quota and budget simultaneously."

## Break 2 — TOP_K set too high

**Change made:** TOP_K increased from 3 to 50 in config.py

**What happened:** With 12 chunks in the store, ChromaDB returned all 12 and the
query completed. No visible failure at this dataset size.

**Why this is still a real failure mode:**
At production scale — 50,000 log chunks — TOP_K=50 would retrieve ~25,000 tokens
of context. Gemini's context window would overflow and the API call would fail
with a token limit error. The failure is latent, not immediate.

**Diagnosis path at scale:** API returns 400 error — "context too long". Check TOP_K
in config.py. Calculate: TOP_K × avg_chunk_tokens must stay well under model limit.

**Fix:** Restored TOP_K = 3. Rule: TOP_K × average chunk size in tokens must fit
comfortably inside the model context window with room for the prompt and response.

**Interview answer:** "I set TOP_K to 50 to test the upper bound. On my 12-chunk
test set it appeared to work, but I recognized this is a latent failure — at
production scale with tens of thousands of chunks, this would overflow the context
window and crash the API call. That taught me TOP_K is a function of chunk size
and model context limit, not just retrieval quality."

## Break 3 — Missing environment variable, silent failure

**Change made:** Commented out the GEMINI_API_KEY validation block in config.py

**What happened:** query.py crashed with a ValueError from inside the google.genai
library internals — traceback pointed to _api_client.py, not to our code.

**Why this is dangerous:** Without the validation, the error message gives no
actionable guidance. A new team member deploying this on a fresh server would
spend time tracing library internals before realising the key was never set.

**Diagnosis path:** Any API failure with None as the key. Check os.getenv() return
value. Check that export was run in the current shell session, not a different one.

**Fix:** Restored the validation block. config.py now raises a clear ValueError
with the exact export command needed before the client is even created.

**Interview answer:** "I removed the API key validation to simulate a misconfigured
deployment. The error came from inside the library with no actionable message.
That confirmed why fail-fast validation at startup matters — catch configuration
errors before any network calls are made, and tell the operator exactly what to do."