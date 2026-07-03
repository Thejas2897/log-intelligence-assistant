"""
authorizer.py — Lambda authorizer for the bedrock-chatbot-api HTTP API.

Runs BEFORE the main bedrock-chatbot-handler Lambda. Checks for a shared
secret in the 'x-api-key' header. Returns a simple isAuthorized true/false
response, which API Gateway uses to either proceed to the main Lambda or
reject the request with a 403 — the main handler (and therefore Bedrock)
never runs on a rejected request.

The secret itself is read from an environment variable, never hardcoded.
"""

import os

EXPECTED_API_KEY = os.environ.get("EXPECTED_API_KEY", "")


def lambda_handler(event, context):
    # For HTTP API Lambda authorizers, headers arrive lowercased inside
    # event["headers"] — a dict, not a list, regardless of how the client
    # capitalized them.
    headers = event.get("headers", {})
    provided_key = headers.get("x-api-key", "")

    is_authorized = bool(EXPECTED_API_KEY) and provided_key == EXPECTED_API_KEY

    return {
        "isAuthorized": is_authorized
    }
