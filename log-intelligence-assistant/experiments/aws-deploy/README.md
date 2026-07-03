# AWS Serverless RAG Deployment

Deploys a Bedrock Knowledge Base as a public, authenticated HTTPS chatbot endpoint.

This experiment takes the same RAG pattern as the local `log-intelligence-assistant` pipeline and runs it on AWS infrastructure — managed retrieval (Bedrock Knowledge Base), managed generation (Amazon Nova Lite), and a serverless API layer (Lambda + API Gateway). The purpose is to demonstrate the managed-cloud path alongside the local/explicit path in the main project.

---

## Architecture

```
curl (POST /ask + x-api-key header)
  → API Gateway (HTTP API)
  → Lambda authorizer (bedrock-chatbot-authorizer)  — 403 if key invalid
  → Lambda handler  (bedrock-chatbot-handler)
  → Bedrock RetrieveAndGenerate
  → Knowledge Base (multi-format: PDF, HTML, CSV)
  → Amazon Nova Lite
  → JSON response: { answer, sources[], sessionId }
```

---

## Files

| File | Purpose |
|---|---|
| `lambda_handler.py` | Lambda function wrapping Bedrock RetrieveAndGenerate |
| `authorizer.py` | Lambda authorizer — checks x-api-key header against env-var secret |
| `lambda_handler.py` | Local test script (invoke handler directly, no API Gateway) |
| `trust-policy.json` | IAM trust policy for Lambda execution roles |
| `permissions-policy.json` | Scoped IAM permissions (Bedrock KB ARN + model ARN only) |
| `test-event.json` | Sample API Gateway proxy event for local testing |

---

## What is deployed (honest status)

| Component | Status |
|---|---|
| Bedrock Knowledge Base (PDF/HTML/CSV) | ✅ Live — KB ID: XXE2NZKLC5 |
| Lambda handler | ✅ Deployed and tested — bedrock-chatbot-handler |
| Lambda authorizer | ✅ Deployed and wired — bedrock-chatbot-authorizer |
| API Gateway HTTP API | ✅ Live — POST /ask, authenticated |
| CloudWatch Model Invocation Logging | ✅ Configured and verified |
| CloudTrail data-event Trail | ✅ Created, verified via real captured event |
| End-to-end curl test | ✅ Confirmed: no-key → 403; with-key → real cited answer |

---

## IAM design

Two separate execution roles, each scoped to exactly what its function needs:

**Lambda handler role** (`lambda-bedrock-chatbot-role`):
- `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents` on the account's log groups
- `bedrock:RetrieveAndGenerate`, `bedrock:Retrieve` on the KB ARN specifically
- `bedrock:InvokeModel` on the Nova Lite model ARN specifically

**Lambda authorizer role** (`lambda-authorizer-role`):
- `AWSLambdaBasicExecutionRole` (CloudWatch logging only) — the authorizer reads an env var and returns a boolean, nothing more

**Why two separate roles:** the authorizer has no business touching Bedrock, and the handler has no business creating authorizers. Least-privilege scoped per-resource, not per-service.

---

## Reproducing from scratch

The zip files (`authorizer.zip`, `lambda_function.zip`) are not committed — regenerate from source:

```bash
zip lambda_function.zip lambda_handler.py
zip authorizer.zip authorizer.py
```

The IAM roles, API Gateway, and KB are not scripted (no CDK/Terraform) — this is a deliberately manual build to demonstrate understanding of each component. Infrastructure-as-code would be the correct next step for a production version.

---

## Key debugging finding

On the first Lambda deploy, the function returned HTTP 500. CloudWatch logs showed:

```
bedrock:InvokeModel ... not authorized ... no identity-based policy allows the action
```

`RetrieveAndGenerate` internally makes two separate AWS API calls: `Retrieve` against the Knowledge Base, and `InvokeModel` against the generation model. The IAM policy only granted the first. Fix: add a second statement granting `bedrock:InvokeModel` scoped to the specific model ARN. This is the most common first-deploy error for Bedrock Lambda integrations.

---

## HTTP API vs REST API — why the Lambda authorizer

AWS API Gateway HTTP APIs (v2) do not support native API key / usage plan features — those belong to the older REST API (v1). The correct HTTP API pattern for access control is a Lambda authorizer: a separate small function that runs before the main handler, checks a custom header, and returns `{"isAuthorized": true/false}`. This is not a workaround — it is the documented, supported approach for HTTP APIs.

---

## What this is not

This is a demo build on a personal AWS account, not a production system. Specific differences from production:
- The API key is a shared secret in a Lambda env var — production would use AWS Secrets Manager with rotation
- The Knowledge Base contains public domain documents — production would need data classification and access controls
- No rate limiting is configured on the API Gateway route — production would add a usage plan or WAF
- MemorySaver-equivalent for sessions is not implemented — each call is stateless
