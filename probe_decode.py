# -*- coding: utf-8 -*-
"""Decode the raw AWS event-stream and print each event's type + payload,
so we can SEE the native reasoningContentEvent text vs assistantResponseEvent."""
import asyncio, json, os, sys, struct
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv; load_dotenv()
import httpx
from kiro.auth import KiroAuthManager
from kiro.utils import get_kiro_headers

REGION = os.getenv("KIRO_API_REGION", "us-east-1")
URL = f"https://runtime.{REGION}.kiro.dev/generateAssistantResponse"
PROFILE_ARN = os.getenv("PROFILE_ARN")
CREDS_FILE = os.path.expanduser(os.getenv("KIRO_CREDS_FILE", "").strip('"'))


def decode_eventstream(raw: bytes):
    """Minimal AWS vnd.amazon.eventstream decoder."""
    events = []
    off = 0
    n = len(raw)
    while off + 12 <= n:
        total_len, headers_len = struct.unpack(">II", raw[off:off+8])
        if total_len <= 0 or off + total_len > n:
            break
        prelude_crc = raw[off+8:off+12]
        hdr_start = off + 12
        hdr_end = hdr_start + headers_len
        headers = {}
        p = hdr_start
        while p < hdr_end:
            name_len = raw[p]; p += 1
            name = raw[p:p+name_len].decode("utf-8", "ignore"); p += name_len
            htype = raw[p]; p += 1
            # htype 7 = string
            if htype == 7:
                vlen = struct.unpack(">H", raw[p:p+2])[0]; p += 2
                val = raw[p:p+vlen].decode("utf-8", "ignore"); p += vlen
            else:
                val = f"<htype {htype}>"; 
                # skip unknown header value lengths conservatively
                break
            headers[name] = val
        payload = raw[hdr_end: off+total_len-4]
        events.append((headers.get(":event-type", "?"), payload))
        off += total_len
    return events


async def main():
    auth = KiroAuthManager(profile_arn=PROFILE_ARN, region=os.getenv("KIRO_REGION","us-east-1"),
                           creds_file=CREDS_FILE, api_region=REGION)
    token = await auth.get_access_token()
    headers = get_kiro_headers(auth, token)
    payload = {
        "conversationState": {
            "chatTriggerType": "MANUAL",
            "conversationId": "probe-decode-1",
            "currentMessage": {"userInputMessage": {
                "content": "What is 17 * 23? Show your reasoning briefly.",
                "modelId": "claude-opus-4.8", "origin": "AI_EDITOR"}},
        },
        "profileArn": PROFILE_ARN,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(URL, headers=headers, content=json.dumps(payload).encode())
    print("HTTP", resp.status_code)
    raw = resp.content
    reasoning_text, answer_text, sig = "", "", None
    for etype, pl in decode_eventstream(raw):
        try:
            obj = json.loads(pl.decode("utf-8", "ignore"))
        except Exception:
            obj = {"_raw": pl[:60].decode("utf-8","ignore")}
        if etype == "reasoningContentEvent":
            reasoning_text += obj.get("text", "")
            if "signature" in obj: sig = obj["signature"][:24] + "..."
        elif etype == "assistantResponseEvent":
            answer_text += obj.get("content", "")
        print(f"[{etype}] {json.dumps(obj, ensure_ascii=False)[:120]}")
    print("\n--- NATIVE REASONING (assembled) ---")
    print(reasoning_text or "(none)")
    print("--- signature ---", sig)
    print("--- ANSWER (assembled) ---")
    print(answer_text or "(none)")

asyncio.run(main())
