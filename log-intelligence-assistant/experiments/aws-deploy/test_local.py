from lambda_handler import lambda_handler
import json

fake_event = {
    "body": json.dumps({"question": "How does the Ford Model T cooling system work?"})
}

result = lambda_handler(fake_event, None)
print(json.dumps(result, indent=2))
