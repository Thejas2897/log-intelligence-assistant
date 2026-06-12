# stream_simulator.py — Live Event Stream Simulation
#
# Simulates a real-time log event stream feeding into the RAG pipeline.
# Generates synthetic log events continuously, runs each through query.py's
# RAG pipeline, and prints a structured incident summary per event.
#
# This demonstrates two modes the Log Intelligence Assistant supports:
#   1. Batch query (query.py)      — analyst asks a question manually
#   2. Live stream (this file)     — events arrive continuously, pipeline responds automatically
#
# Usage:
#   python stream_simulator.py              # runs until Ctrl+C
#   python stream_simulator.py --events 5  # runs for exactly 5 events then exits

import time
import random
import argparse
from datetime import datetime

# Import the RAG pipeline's generate_answer function directly
# This means stream_simulator reuses all of query.py — no duplication
from query import generate_answer

# ── SYNTHETIC EVENT TEMPLATES ──────────────────────────────────────────────
#
# These represent the kinds of events that would arrive from a real log stream
# (Kafka topic, Fluentd pipeline, CloudWatch stream, etc.)
# Each entry is a natural language description of a log event — the same
# format a human analyst would type into query.py manually.

EVENT_TEMPLATES = [
    "auth-service is returning authentication token expired errors for multiple users",
    "payment-service database connection pool exhausted, 100/100 connections used",
    "circuit breaker opened on payment-service, stopping requests to db-service",
    "payment-service response time 8200ms exceeds SLA threshold of 2000ms",
    "auth-db-primary unreachable, token refresh failing",
    "api-gateway returning 503 Service Unavailable to clients",
    "db-service query timeout after 30000ms on transactions table",
    "payment-service retry attempt 3 of 3 failed, auth-service unavailable",
]

# ── STREAM SIMULATOR ───────────────────────────────────────────────────────

def simulate_stream(max_events: int = None, interval_seconds: float = 3.0):
    """
    Continuously generates synthetic log events and runs each through the RAG pipeline.

    Args:
        max_events: Stop after this many events. None means run until Ctrl+C.
        interval_seconds: Pause between events to simulate real stream pacing.
    """
    print("=" * 60)
    print("Log Intelligence Assistant — Live Stream Simulation")
    print("=" * 60)
    print(f"Stream started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Interval: {interval_seconds}s between events")
    if max_events:
        print(f"Will process {max_events} events then exit")
    else:
        print("Running until Ctrl+C")
    print("=" * 60)

    event_count = 0

    try:
        while True:
            # Check if we've hit the event limit
            if max_events and event_count >= max_events:
                print(f"\nReached {max_events} events. Simulation complete.")
                break

            event_count += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Pick a random event from the templates
            # In production this would be replaced by a Kafka consumer or log tail
            event = random.choice(EVENT_TEMPLATES)

            print(f"\n[{timestamp}] EVENT #{event_count} RECEIVED")
            print(f"Raw event: {event}")
            print("Running through RAG pipeline...")

            # Feed the event directly into the same RAG pipeline used by query.py
            answer, chunks, metadatas = generate_answer(event)

            # Show which sources were retrieved
            sources = list({m["source"] for m in metadatas})
            print(f"Retrieved from: {', '.join(sources)}")

            # Print the structured incident summary
            print("\n--- Incident Summary ---")
            print(answer)
            print("-" * 40)

            # Wait before next event — simulates stream pacing
            # In production this is controlled by Kafka consumer poll interval
            print(f"Next event in {interval_seconds}s... (Ctrl+C to stop)")
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        # Clean exit on Ctrl+C — expected behaviour for a stream process
        print(f"\n\nStream stopped by user after {event_count} events.")
        print(f"Stopped at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ── MAIN ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Simulate a live log event stream feeding into the RAG pipeline"
    )
    parser.add_argument(
        "--events",
        type=int,
        default=None,
        help="Number of events to process before exiting (default: run until Ctrl+C)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Seconds between events (default: 3.0)"
    )
    args = parser.parse_args()

    simulate_stream(max_events=args.events, interval_seconds=args.interval)


if __name__ == "__main__":
    main()