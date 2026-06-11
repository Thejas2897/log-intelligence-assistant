"""
vllm_simulator.py — Simulates a vLLM-style request queue with batching.

NOT real vLLM. This is a simulation that makes the core concepts tangible:
- Multiple users send queries simultaneously
- A queue manager holds incoming requests
- Requests are processed in batches
- Response times are logged per request

Why build this on a 3.7GB RAM laptop?
Because the concepts — batching, concurrency, queue depth, throughput vs latency —
are what Vivek is evaluating. The simulation makes these concrete and demonstrable.

Interview answer this enables:
"I cannot run vLLM on my laptop, but I built a queue simulation that demonstrates
the batching and throughput concept from first principles."

Run this file directly:
    python vllm_simulator.py

Observe how batch processing changes throughput vs sequential processing.
"""

import time
import random
import threading
from queue import Queue
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Data structure representing one incoming request
# ---------------------------------------------------------------------------

@dataclass
class InferenceRequest:
    """
    Represents a single user query arriving at the vLLM serving system.

    In real vLLM, this would contain the tokenised prompt and generation params.
    Here we simulate it with a prompt string and a simulated token count.
    """
    request_id: str          # Unique identifier for this request
    user: str                # Which simulated user sent this
    prompt: str              # The query text
    token_count: int         # Simulated token count — determines processing time
    arrival_time: float = field(default_factory=time.time)  # When it arrived


# ---------------------------------------------------------------------------
# Result structure returned after processing
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """
    Represents the completed response for one request.
    Tracks timing so we can measure latency per request.
    """
    request_id: str
    user: str
    prompt: str
    simulated_response: str
    queue_wait_seconds: float   # Time spent waiting in queue before processing
    process_seconds: float      # Time spent actually being processed
    total_seconds: float        # End-to-end latency experienced by the user


# ---------------------------------------------------------------------------
# The simulated vLLM serving engine
# ---------------------------------------------------------------------------

class VLLMSimulator:
    """
    Simulates the core behaviour of a vLLM serving engine.

    Key concepts demonstrated:
    - Request queue: holds requests when GPU (simulated) is busy
    - Batch processing: groups multiple requests for one processing cycle
    - Continuous batching: processes a batch, immediately picks up the next
    - Throughput measurement: requests completed per second
    - Latency measurement: end-to-end time per individual request

    Parameters
    ----------
    batch_size : int
        Maximum number of requests processed together in one cycle.
        Simulates the GPU processing multiple sequences simultaneously.

    gpu_tokens_per_second : int
        Simulated GPU throughput — how many tokens the "GPU" processes per second.
        Higher = faster hardware. Used to compute realistic processing times.

    max_queue_depth : int
        Maximum requests that can wait in the queue.
        Requests arriving when the queue is full are rejected — simulates
        real serving systems that return 503 when overloaded.
    """

    def __init__(
        self,
        batch_size: int = 4,
        gpu_tokens_per_second: int = 200,
        max_queue_depth: int = 20
    ):
        self.batch_size = batch_size
        self.gpu_tokens_per_second = gpu_tokens_per_second
        self.max_queue_depth = max_queue_depth

        # Thread-safe queue — multiple user threads push requests here simultaneously
        self.request_queue: Queue = Queue(maxsize=max_queue_depth)

        self.results: List[InferenceResult] = []
        self.results_lock = threading.Lock()  # Prevents race conditions when storing results

        self.rejected_count = 0   # Requests dropped because queue was full
        self.start_time = None

    def submit_request(self, request: InferenceRequest) -> bool:
        """
        Called by each simulated user thread to submit a request.
        Returns True if accepted into the queue, False if queue was full.
        """
        try:
            # block=False means: if queue is full, raise exception immediately
            # rather than waiting — simulates real serving rejecting overloaded requests
            self.request_queue.put(request, block=False)
            return True
        except Exception:
            # Queue is full — request rejected
            self.rejected_count += 1
            print(f"  [REJECTED] {request.request_id} from {request.user} — queue full")
            return False

    def _process_batch(self, batch: List[InferenceRequest]):
        """
        Simulates processing one batch of requests on the GPU.

        In real vLLM, the GPU processes tokens from all requests in the batch
        simultaneously via continuous batching. Here we simulate this by:
        - Taking the maximum token count in the batch as the processing time
          (the batch finishes when the longest request finishes — others complete earlier)
        - Sleeping for the equivalent simulated processing time
        """
        if not batch:
            return

        # The batch takes as long as the longest request in it
        # (shorter requests finish first but GPU stays busy until the longest completes)
        max_tokens = max(req.token_count for req in batch)
        process_time = max_tokens / self.gpu_tokens_per_second

        print(f"\n  [BATCH] Processing {len(batch)} request(s) — "
              f"max tokens: {max_tokens} — estimated time: {process_time:.2f}s")

        # Simulate GPU processing time
        time.sleep(process_time)

        # Record results for each request in the batch
        batch_end_time = time.time()

        with self.results_lock:
            for req in batch:
                queue_wait = batch_end_time - process_time - req.arrival_time
                total_time = batch_end_time - req.arrival_time

                result = InferenceResult(
                    request_id=req.request_id,
                    user=req.user,
                    prompt=req.prompt,
                    simulated_response=f"[Simulated response to: '{req.prompt[:40]}...']",
                    queue_wait_seconds=max(0.0, queue_wait),
                    process_seconds=process_time,
                    total_seconds=total_time
                )
                self.results.append(result)
                print(f"    ✓ {req.request_id} ({req.user}) — "
                      f"wait: {result.queue_wait_seconds:.2f}s | "
                      f"process: {result.process_seconds:.2f}s | "
                      f"total: {result.total_seconds:.2f}s")

    def run(self, total_requests_expected: int):
        """
        Main processing loop — runs until all expected requests are processed.

        Continuously pulls batches from the queue and processes them.
        This is the 'continuous batching' behaviour: as soon as one batch
        finishes, the next batch is assembled and started immediately.
        """
        self.start_time = time.time()
        processed = 0

        print(f"\n[ENGINE] vLLM Simulator started")
        print(f"         Batch size        : {self.batch_size}")
        print(f"         GPU speed         : {self.gpu_tokens_per_second} tokens/sec")
        print(f"         Max queue depth   : {self.max_queue_depth}")
        print(f"         Requests expected : {total_requests_expected}\n")

        while processed < total_requests_expected:
            batch = []

            # Collect up to batch_size requests from the queue
            # Wait up to 0.1s for each slot — if queue is temporarily empty, keep waiting
            while len(batch) < self.batch_size:
                try:
                    # timeout=0.1: don't wait more than 100ms for the next request
                    req = self.request_queue.get(timeout=0.1)
                    batch.append(req)
                except Exception:
                    # Queue temporarily empty — process whatever we have so far
                    break

            if batch:
                self._process_batch(batch)
                processed += len(batch)

        total_time = time.time() - self.start_time
        return total_time


# ---------------------------------------------------------------------------
# Simulated users — each runs in its own thread
# ---------------------------------------------------------------------------

# Sample prompts representing realistic AIOps queries
SAMPLE_PROMPTS = [
    "Summarise all ERROR level events in the last hour",
    "What is the recommended action for a database connection timeout?",
    "How many times did the payment service restart today?",
    "What does runbook entry RB-004 say about memory pressure incidents?",
    "Correlate the disk I/O spike at 14:32 with downstream service failures",
    "Which services were affected during the 503 cascade at 09:15?",
    "What is the standard escalation path for P1 incidents?",
    "List all CRITICAL alerts that fired between midnight and 6 AM",
]

def simulate_user(
    user_id: str,
    engine: VLLMSimulator,
    num_requests: int,
    arrival_delay_range: tuple = (0.1, 0.5)
):
    """
    Simulates one user sending multiple requests with random delays between them.
    Each user runs in its own thread — concurrent with all other users.

    arrival_delay_range: simulates realistic think-time between queries.
    Shorter range = more aggressive user = higher load on the system.
    """
    for i in range(num_requests):
        # Random delay between requests — simulates real user think-time
        time.sleep(random.uniform(*arrival_delay_range))

        request = InferenceRequest(
            request_id=f"{user_id}-req{i+1:02d}",
            user=user_id,
            prompt=random.choice(SAMPLE_PROMPTS),
            # Token count between 50 and 300 — simulates variable response lengths
            token_count=random.randint(50, 300)
        )

        accepted = engine.submit_request(request)
        if accepted:
            print(f"  [QUEUED] {request.request_id} — {request.token_count} tokens")


# ---------------------------------------------------------------------------
# Run two scenarios and compare results
# ---------------------------------------------------------------------------

def run_scenario(
    scenario_name: str,
    num_users: int,
    requests_per_user: int,
    batch_size: int,
    gpu_tokens_per_second: int
):
    """
    Runs one complete simulation scenario and prints a summary.
    Call this twice with different batch sizes to see the throughput difference.
    """
    print("\n" + "="*65)
    print(f"SCENARIO: {scenario_name}")
    print("="*65)

    total_requests = num_users * requests_per_user
    engine = VLLMSimulator(
        batch_size=batch_size,
        gpu_tokens_per_second=gpu_tokens_per_second
    )

    # Launch one thread per simulated user
    user_threads = []
    for i in range(num_users):
        thread = threading.Thread(
            target=simulate_user,
            args=(f"User{i+1:02d}", engine, requests_per_user),
            daemon=True
        )
        user_threads.append(thread)

    # Start all user threads simultaneously — they all arrive at roughly the same time
    for thread in user_threads:
        thread.start()

    # Run the engine until all requests are processed
    total_time = engine.run(total_requests_expected=total_requests)

    # Wait for all user threads to finish submitting
    for thread in user_threads:
        thread.join()

    # ---------------------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------------------
    results = engine.results
    if not results:
        print("No results recorded.")
        return

    avg_latency = sum(r.total_seconds for r in results) / len(results)
    avg_wait = sum(r.queue_wait_seconds for r in results) / len(results)
    throughput = len(results) / total_time

    print(f"\n{'─'*65}")
    print(f"RESULTS — {scenario_name}")
    print(f"{'─'*65}")
    print(f"  Total requests processed : {len(results)}")
    print(f"  Requests rejected        : {engine.rejected_count}")
    print(f"  Total wall-clock time    : {total_time:.2f}s")
    print(f"  Throughput               : {throughput:.2f} requests/sec")
    print(f"  Avg end-to-end latency   : {avg_latency:.2f}s per request")
    print(f"  Avg queue wait time      : {avg_wait:.2f}s per request")
    print(f"  Batch size used          : {batch_size}")
    print(f"{'─'*65}")

    return {
        "throughput": throughput,
        "avg_latency": avg_latency,
        "total_time": total_time
    }


# ---------------------------------------------------------------------------
# Main — run both scenarios and compare
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*65)
    print("  vLLM Request Queue Simulator")
    print("  Demonstrates: batching, concurrency, throughput vs latency")
    print("="*65)

    # Scenario 1: No batching — each request processed one at a time
    # This is how a naive LLM server works
    result_sequential = run_scenario(
        scenario_name="Sequential (batch_size=1) — naive server",
        num_users=5,
        requests_per_user=3,
        batch_size=1,
        gpu_tokens_per_second=200
    )

    # Small pause between scenarios for readability
    time.sleep(1)

    # Scenario 2: Batching enabled — 4 requests processed together
    # This is closer to how vLLM's continuous batching works
    result_batched = run_scenario(
        scenario_name="Batched (batch_size=4) — vLLM-style",
        num_users=5,
        requests_per_user=3,
        batch_size=4,
        gpu_tokens_per_second=200
    )

    # ---------------------------------------------------------------------------
    # Final comparison — this is what you explain to Vivek
    # ---------------------------------------------------------------------------
    if result_sequential and result_batched:
        print("\n" + "="*65)
        print("COMPARISON — What batching changes")
        print("="*65)
        throughput_gain = result_batched["throughput"] / result_sequential["throughput"]
        time_reduction = (
            (result_sequential["total_time"] - result_batched["total_time"])
            / result_sequential["total_time"] * 100
        )
        print(f"  Throughput gain from batching : {throughput_gain:.1f}x")
        print(f"  Total time reduction         : {time_reduction:.0f}%")
        print(f"\n  Sequential: {result_sequential['throughput']:.2f} req/sec")
        print(f"  Batched   : {result_batched['throughput']:.2f} req/sec")
        print(f"\n  Observation: batching increases throughput at the cost of")
        print(f"  slightly higher latency for individual requests.")
        print(f"  This is the core throughput-vs-latency tradeoff in vLLM.")
        print("="*65)
