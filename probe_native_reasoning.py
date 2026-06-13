# -*- coding: utf-8 -*-
"""
Probe whether runtime.kiro.dev backend supports NATIVE reasoning,
or whether kg's fake-reasoning prompt injection is still required.

Reuses kg's KiroAuthManager for auth. Sends raw GenerateAssistantResponse
requests directly and dumps the raw AWS event-stream so we can see exactly
what event types the backend emits.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import httpx
from kiro.auth import KiroAuthManager
from kiro.utils import get_kiro_headers

REGION = os.getenv("KIRO_API_REGION", "us-east-1")
API_HOST = f"https://runtime.{REGION}.kiro.dev"
URL = f"{API_HOST}/generateAssistantResponse"
PROFILE_ARN = os.getenv("PROFILE_ARN")
CREDS_FILE = os.path.expanduser(os.getenv("KIRO_CREDS_FILE", "").strip('"'))
MODEL = "claude-opus-4.8"  # runtime expects dot form (kg normalizes dash->dot)


def extract_event_types(raw: bytes):
    """Pull every distinct top-level JSON key-ish event marker from raw stream."""
    text = raw.decode("utf-8", errors="ignore")
    import re
    # AWS event stream embeds :event-type headers and JSON payloads.
    headers = sorted(set(re.findall(r":event-type[^a-zA-Z]*([a-zA-Z]+)", text)))
    # JSON object first-keys
    json_keys = sorted(set(re.findall(r'\{"([a-zA-Z][a-zA-Z0-9_]*)":', text)))
    return headers, json_keys


async def send(payload: dict, label: str):
    auth = KiroAuthManager(
        profile_arn=PROFILE_ARN,
        region=os.getenv("KIRO_REGION", "us-east-1"),
        creds_file=CREDS_FILE,
        api_region=REGION,
    )
    token = await auth.get_access_token()
    headers = get_kiro_headers(auth, token)
    print(f"\n{'='*70}\n[{label}]\n{'='*70}")
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(URL, headers=headers, content=json.dumps(payload).encode())
        except Exception as e:
            print(f"REQUEST ERROR: {type(e).__name__}: {e}")
            return
        print(f"HTTP {resp.status_code}")
        raw = resp.content
        if resp.status_code != 200:
            print("BODY:", raw.decode("utf-8", errors="ignore")[:600])
            return
        ev_headers, json_keys = extract_event_types(raw)
        print(f"raw bytes: {len(raw)}")
        print(f":event-type headers: {ev_headers}")
        print(f"JSON first-keys seen: {json_keys}")
        # Show any reasoning-ish content
        low = raw.decode("utf-8", errors="ignore").lower()
        for kw in ("reasoning", "thinking", "thought", "redacted", "signature"):
            if kw in low:
                idx = low.find(kw)
                print(f"  >> '{kw}' FOUND near: ...{raw.decode('utf-8',errors='ignore')[max(0,idx-40):idx+80]}...")


def base_payload(extra_user_fields=None, extra_conv=None):
    user_msg = {
        "content": "What is 17 * 23? Think step by step.",
        "modelId": MODEL,
        "origin": "AI_EDITOR",
    }
    if extra_user_fields:
        user_msg.update(extra_user_fields)
    conv = {
        "chatTriggerType": "MANUAL",
        "conversationId": "probe-0001",
        "currentMessage": {"userInputMessage": user_msg},
    }
    if extra_conv:
        conv.update(extra_conv)
    p = {"conversationState": conv}
    if PROFILE_ARN:
        p["profileArn"] = PROFILE_ARN
    return p


async def main():
    # 1) Baseline — plain request, no thinking injection. See what backend emits.
    await send(base_payload(), "BASELINE (no reasoning hint)")

    # 2) Try native reasoning field on userInputMessage (Bedrock/Anthropic style)
    await send(
        base_payload(extra_user_fields={
            "reasoningConfig": {"type": "enabled", "budgetTokens": 4000}
        }),
        "userInputMessage.reasoningConfig",
    )

    # 3) Try modelConfig / inferenceConfig style
    await send(
        base_payload(extra_conv={
            "inferenceConfig": {"reasoning": {"effort": "high"}}
        }),
        "conversationState.inferenceConfig.reasoning",
    )

    # 4) Try thinking field (Anthropic messages API style)
    await send(
        base_payload(extra_user_fields={
            "thinking": {"type": "enabled", "budget_tokens": 4000}
        }),
        "userInputMessage.thinking",
    )


if __name__ == "__main__":
    asyncio.run(main())
