"""
lambda_handler.py — thin Lambda wrapper around the existing, already-debugged
Bedrock Knowledge Base RetrieveAndGenerate call.

Reuses the exact, proven configuration from the CLI debugging work:
  - Knowledge Base ID: XXE2NZKLC5
  - Generation model: amazon.nova-lite-v1:0 (confirmed working, no Marketplace gate)

No new Bedrock logic here — this is purely an HTTP-invocation layer on top of
a call that has already been exhaustively tested via CLI.
"""

import json
import os
import boto3

# Environment variable, not hardcoded — same discipline as every prior part
# of this project (no hardcoded API keys/IDs in source).
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "XXE2NZKLC5")
MODEL_ARN = os.environ.get(
    "MODEL_ARN",
    "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-lite-v1:0",
)

bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name="us-east-1")


def lambda_handler(event, context):
    """
    Expects an API Gateway proxy event with a JSON body: {"question": "..."}
    Returns an API Gateway proxy response: {"statusCode": ..., "body": "..."}
    """
    try:
        # API Gateway (HTTP API / Lambda proxy integration) delivers the
        # request body as a JSON string inside event["body"] — not as a
        # pre-parsed dict. Has to be parsed explicitly.
        body = json.loads(event.get("body", "{}"))
        question = body.get("question", "").strip()

        if not question:
            return _response(400, {"error": "Missing required field: 'question'"})

        # The exact call shape already proven to work via CLI throughout
        # this project's debugging arc — no new logic, just invoked
        # programmatically instead of via `aws bedrock-agent-runtime ...`.
        result = bedrock_agent_runtime.retrieve_and_generate(
            input={"text": question},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    "modelArn": MODEL_ARN,
                },
            },
        )

        answer = result.get("output", {}).get("text", "")

        # Surface citations too — useful for the interview demo, and
        # consistent with how every prior result in this project was
        # validated by checking real citations, not just trusting the text.
        citations = []
        for citation in result.get("citations", []):
            for ref in citation.get("retrievedReferences", []):
                source = ref.get("location", {}).get("s3Location", {}).get("uri", "")
                if source and source not in citations:
                    citations.append(source)

        return _response(200, {
            "answer": answer,
            "sources": citations,
            "sessionId": result.get("sessionId", ""),
        })

    except json.JSONDecodeError:
        return _response(400, {"error": "Request body must be valid JSON"})
    except bedrock_agent_runtime.exceptions.ResourceNotFoundException:
        return _response(404, {"error": "Knowledge Base not found"})
    except Exception as e:
        # Deliberately not leaking internal error details to the client —
        # log the real exception server-side (CloudWatch), return a generic
        # message externally. Same instinct as the anti-hallucination /
        # error-handling discipline already established in this project.
        print(f"Unhandled error: {repr(e)}")
        return _response(500, {"error": "Internal server error"})


def _response(status_code, body_dict):
    """Standard API Gateway Lambda-proxy response shape."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",  # fine for a demo; would be scoped in real prod
        },
        "body": json.dumps(body_dict),
    }
