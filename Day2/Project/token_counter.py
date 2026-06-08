import os
from dotenv import load_dotenv
from google import genai

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

# Initialise the new SDK client
client = genai.Client(api_key=api_key)

# Model to use for token counting
MODEL_NAME = "gemini-2.5-flash"

# Gemini 2.0 Flash context window — 1 million tokens
CONTEXT_WINDOW = 1_000_000

# Warning thresholds
WARN_YELLOW = 0.50  # 50% — worth knowing
WARN_ORANGE = 0.75  # 75% — start trimming
WARN_RED    = 0.90  # 90% — act now


def count_and_display(text):
    """
    Counts tokens in the given text, shows percentage of context
    window used, and warns based on threshold.
    """
    try:
        # Count tokens without making a full generate_content call
        result = client.models.count_tokens(
            model=MODEL_NAME,
            contents=text
        )
        token_count = result.total_tokens

        # Calculate percentage of context window consumed
        percentage = (token_count / CONTEXT_WINDOW) * 100

        # Build a simple visual bar — 40 chars wide
        filled = int((token_count / CONTEXT_WINDOW) * 40)
        bar = "█" * filled + "░" * (40 - filled)

        print("\n" + "=" * 50)
        print(f"Token count     : {token_count:,}")
        print(f"Context window  : {CONTEXT_WINDOW:,}")
        print(f"Usage           : {percentage:.4f}%")
        print(f"[{bar}]")

        # Warn based on threshold
        if percentage >= WARN_RED * 100:
            print("STATUS: CRITICAL — approaching context limit. Trim input now.")
        elif percentage >= WARN_ORANGE * 100:
            print("STATUS: WARNING  — context is getting large. Consider trimming.")
        elif percentage >= WARN_YELLOW * 100:
            print("STATUS: NOTICE   — halfway through context window.")
        else:
            print("STATUS: OK       — well within context window.")

        print("=" * 50 + "\n")

        return token_count

    except Exception as e:
        print(f"Token count failed: {e}")
        return None


def run_token_counter():
    """
    Main loop — accepts text input and counts tokens until user exits.
    Supports multiline input — type END on its own line to submit.
    """
    print("=" * 50)
    print("Token Counter — Gemini 2.0 Flash")
    print("Type or paste text, then type END on a new line to count.")
    print("Type 'exit' to quit.")
    print("=" * 50 + "\n")

    while True:
        print("Enter text (END to submit, exit to quit):")
        lines = []

        while True:
            line = input()

            # Exit condition — check before appending
            if line.strip().lower() == "exit":
                print("Session ended.")
                return

            # END signals submission of multiline input
            if line.strip() == "END":
                break

            lines.append(line)

        # Join all lines into a single text block
        text = "\n".join(lines).strip()

        if not text:
            print("No text entered. Try again.\n")
            continue

        count_and_display(text)


if __name__ == "__main__":
    run_token_counter()