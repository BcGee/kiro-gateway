# -*- coding: utf-8 -*-
"""Fetch ListAvailableModels and inspect additionalModelRequestFieldsSchema per model."""
import asyncio, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv; load_dotenv()
import httpx
from kiro.auth import KiroAuthManager
from kiro.utils import get_kiro_headers

REGION = os.getenv("KIRO_API_REGION","us-east-1")
PROFILE_ARN = os.getenv("PROFILE_ARN")
CREDS = os.path.expanduser(os.getenv("KIRO_CREDS_FILE","").strip('"'))

async def main():
    auth = KiroAuthManager(profile_arn=PROFILE_ARN, region=os.getenv("KIRO_REGION","us-east-1"),
                           creds_file=CREDS, api_region=REGION)
    token = await auth.get_access_token()
    headers = get_kiro_headers(auth, token)
    headers["x-amz-target"] = "AmazonCodeWhispererStreamingService.ListAvailableModels"
    url = f"https://runtime.{REGION}.kiro.dev/listAvailableModels"
    body = {"origin":"AI_EDITOR","profileArn":PROFILE_ARN}
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(url, headers=headers, content=json.dumps(body).encode())
    print("HTTP", r.status_code)
    if r.status_code != 200:
        print(r.content[:400].decode("utf-8","ignore")); return
    data = r.json()
    models = data.get("models", [])
    print(f"models: {len(models)}\n")
    for m in models:
        sid = m["modelId"]
        has = "additionalModelRequestFieldsSchema" in m
        marker = "SCHEMA" if has else "no-schema"
        print(f"{sid:24} [{marker}] keys={list(m.keys())}")
        if has:
            sch = m["additionalModelRequestFieldsSchema"]
            print("   ", json.dumps(sch)[:500])

asyncio.run(main())
