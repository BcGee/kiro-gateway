# -*- coding: utf-8 -*-
"""
Empirically probe each model's effort schema by sending an INVALID effort value
inside the output_config shape and classifying the 400 response:

  - "does not have a value in the enumeration [...]"  -> model uses output_config schema
  - "property 'output_config' is not defined ..."     -> model uses reasoning schema (or other)
  - "Invalid additionalModelRequestFields ... reasoning is not defined" (when we send reasoning)
  - HTTP 200                                            -> field ignored / no effort schema

We probe both shapes to disambiguate.
"""
import asyncio, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv; load_dotenv()
import httpx
from kiro.auth import KiroAuthManager
from kiro.utils import get_kiro_headers
from kiro.config import FALLBACK_MODELS

REGION = os.getenv("KIRO_API_REGION","us-east-1")
URL = f"https://runtime.{REGION}.kiro.dev/generateAssistantResponse"
PROFILE_ARN = os.getenv("PROFILE_ARN")
CREDS = os.path.expanduser(os.getenv("KIRO_CREDS_FILE","").strip('"'))

_auth = None
def auth():
    global _auth
    if _auth is None:
        _auth = KiroAuthManager(profile_arn=PROFILE_ARN, region=os.getenv("KIRO_REGION","us-east-1"),
                                creds_file=CREDS, api_region=REGION)
    return _auth

async def probe(client, model, amrf):
    token = await auth().get_access_token()
    headers = get_kiro_headers(auth(), token)
    payload = {
        "conversationState": {
            "chatTriggerType": "MANUAL", "conversationId": "probe-schema",
            "currentMessage": {"userInputMessage": {"content": "hi", "modelId": model, "origin": "AI_EDITOR"}},
        },
        "profileArn": PROFILE_ARN,
        "additionalModelRequestFields": amrf,
    }
    try:
        r = await client.post(URL, headers=headers, content=json.dumps(payload).encode())
    except Exception as e:
        return f"ERR {type(e).__name__}"
    if r.status_code == 200:
        return "200 (accepted/ignored)"
    msg = ""
    try: msg = r.json().get("message","")
    except: msg = r.content[:160].decode("utf-8","ignore")
    return f"{r.status_code}: {msg[:150]}"

async def main():
    models = [m["modelId"] for m in FALLBACK_MODELS]
    bad_oc = {"thinking":{"type":"adaptive","display":"summarized"},"output_config":{"effort":"__BAD__"}}
    bad_re = {"reasoning":{"effort":"__BAD__"}}
    async with httpx.AsyncClient(timeout=30) as client:
        results = {}
        for m in models:
            a = await probe(client, m, bad_oc)
            b = await probe(client, m, bad_re)
            if "enumeration" in a:
                schema = "output_config"
            elif "enumeration" in b:
                schema = "reasoning"
            elif "not supported" in a.lower() or "not supported" in b.lower():
                schema = "UNSUPPORTED"
            elif "200" in a or "200" in b:
                schema = "NONE(ignored)"
            else:
                schema = "?"
            results[m] = schema
            print(f"\n=== {m} -> {schema} ===")
            print(f"  output_config probe: {a}")
            print(f"  reasoning probe:     {b}")
        print("\n\n==================== SUMMARY ====================")
        for m, s in results.items():
            print(f"{m:24} {s}")

asyncio.run(main())
