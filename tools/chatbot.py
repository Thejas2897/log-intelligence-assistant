import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=api_key)

SYSTEM_PROMPT = """You are an AIOps incident analyst assistant.
You help engineers diagnose system incidents, analyse log errors,
and recommend remediation steps. Be concise and structured.
If you don't have enough context to answer confidently, say so."""

# Store conversation history manually as a list of dicts
conversation_history = []


def call_gemini_with_retry(user_input, max_retries=3):
    """
    Sends a message with full conversation history and retry logic.
    Returns response text or raises after max_retries exhausted.
    """
    # Append the new user message to history
    conversation_history.append({
        "role": "user",
        "parts": [{"text": user_input}]
    })

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=conversation_history,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "temperature": 0.1
                }
            )

            if not response.text:
                raise ValueError("Empty response received from API")

            # Append assistant response to history for next turn
            conversation_history.append({
                "role": "model",
                "parts": [{"text": response.text}]
            })

            return response.text

        except errors.APIError as e:
            if "429" in str(e):
                wait_time = 2 ** attempt
                print(f"\nRate limit hit. Waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            elif "400" in str(e):
                raise ValueError(f"Invalid request: {e}")
            else:
                print(f"\nAPI error on attempt {attempt + 1}: {e}")

        except Exception as e:
            print(f"\nUnexpected error on attempt {attempt + 1}: {e}")

    # Remove the user message we appended since all retries failed
    conversation_history.pop()
    raise RuntimeError(f"Gemini API failed after {max_retries} attempts")


def count_history_tokens():
    """
    Counts total tokens in the current conversation history.
    """
    history_text = ""
    for message in conversation_history:
        history_text += message["parts"][0]["text"] + "\n"

    # Rough estimate: 1 token ≈ 4 characters
    estimated_tokens = len(history_text) // 4
    return estimated_tokens


def run_chatbot():
    print("=" * 50)
    print("AIOps Incident Analyst — Terminal Chatbot")
    print("Type 'exit' to quit | Type 'history' to see token count")
    print("=" * 50)
    print()

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Session ended.")
            break

        if user_input.lower() == "history":
            tokens = count_history_tokens()
            print(f"Current conversation: ~{tokens} tokens used")
            continue

        try:
            response_text = call_gemini_with_retry(user_input)
            print(f"\nAssistant: {response_text}\n")

            tokens = count_history_tokens()
            if tokens > 50000:
                print(f"[Warning: ~{tokens} tokens in history — consider starting a new session]\n")

        except Exception as e:
            print(f"\nFailed to get response: {e}\n")


if __name__ == "__main__":
    run_chatbot()