# -*- coding: utf-8 -*-
"""
Verify the CORRECT effort mechanism decoded from Kiro client (extension.js):

  userInputMessage.modelId, content, origin  (as before)
  + the effort goes via Bedrock Converse 'additionalModelRequestFields':
      schema "reasoning":     {"reasoning": {"effort": "<Level>"}}
      schema "output_config": {"thinking":{"type":"adaptive","display":"summarized"},
                               "output_config":{"effort":"<Level>"}}
  Effort level value: capitalize first letter; xhigh->xHigh  (Yh2())

Question: where does Kiro put additionalModelRequestFields in the GenerateAssistantResponse
payload, and does the backend honor it (different reasoning length per effort)?

We test by placing it at a few plausible locations and measuring reasoning token volume.
"""
import asyncio, json, os, struct, sys
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv; load_dotenv()
import httpx
from kiro.auth import KiroAuthManager
from kiro.utils import get_kiro_headers

REGION = os.getenv("KIRO_API_REGION", "us-east-1")
URL = f"https://runtime.{REGION}.kiro.dev/generateAssistantResponse"
PROFILE_ARN = os.getenv("PROFILE_ARN")
CREDS = os.path.expanduser(os.getenv("KIRO_CREDS_FILE", "").strip('"'))
MODEL = "claude-opus-4.8"
PROMPT = "What is 1234 * 5678? Show your reasoning."


def decode(raw):
    ev = {"reasoning": "", "answer": ""}
    off, n = 0, len(raw)
    while off + 12 <= n:
        tot, hl = struct.unpack(">II", raw[off:off+8])
        if tot <= 0 or off+tot > n: break
        hs, he = off+12, off+12+hl
        headers, p = {}, hs
        while p < he:
            nl = raw[p]; p += 1
            name = raw[p:p+nl].decode("utf-8","ignore"); p += nl
            ht = raw[p]; p += 1
            if ht == 7:
                vl = struct.unpack(">H", raw[p:p+2])[0]; p += 2
                val = raw[p:p+vl].decode("utf-8","ignore"); p += vl
            else: break
            headers[name] = val
        payload = raw[he:off+tot-4]
        et = headers.get(":event-type","?")
        try: obj = json.loads(payload.decode("utf-8","ignore"))
        except: obj = {}
        if et == "reasoningContentEvent": ev["reasoning"] += obj.get("text","")
        elif et == "assistantResponseEvent": ev["answer"] += obj.get("content","")
        off += tot
    return ev


async def send(label, extra_user=None, extra_conv=None, extra_top=None):
    auth = KiroAuthManager(profile_arn=PROFILE_ARN, region=os.getenv("KIRO_REGION","us-east-1"),
                           creds_file=CREDS, api_region=REGION)
    token = await auth.get_access_token()
    headers = get_kiro_headers(auth, token)
    uim = {"content": PROMPT, "modelId": MODEL, "origin": "AI_EDITOR"}
    if extra_user: uim.update(extra_user)
    conv = {"chatTriggerType":"MANUAL","conversationId":"probe-eff","currentMessage":{"userInputMessage":uim}}
    if extra_conv: conv.update(extra_conv)
    payload = {"conversationState": conv, "profileArn": PROFILE_ARN}
    if extra_top: payload.update(extra_top)
    async with httpx.AsyncClient(timeout=90) as c:
        r = await c.post(URL, headers=headers, content=json.dumps(payload).encode())
    if r.status_code != 200:
        print(f"[{label}] HTTP {r.status_code}: {r.content[:200].decode('utf-8','ignore')}")
        return None
    ev = decode(r.content)
    print(f"[{label}] HTTP 200 | reasoning={len(ev['reasoning'])} chars | answer={len(ev['answer'])} chars")
    return len(ev["reasoning"])


async def main():
    # DECODED CORRECT SHAPE: top-level additionalModelRequestFields,
    # opus-4.8 uses the output_config schema, effort value is LOWERCASE.
    def oc(level):
        return {"thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": level}}

    print("=== baseline (no effort field) — run 2x to gauge noise ===")
    await send("baseline-1")
    await send("baseline-2")
    print("\n=== top-level additionalModelRequestFields, output_config schema, LOWERCASE ===")
    for lvl in ["low", "medium", "high", "xhigh", "max"]:
        await send(f"effort={lvl}", extra_top={"additionalModelRequestFields": oc(lvl)})
    print("\n=== confirm low vs max twice each (check effort actually scales reasoning) ===")
    await send("low(b)",  extra_top={"additionalModelRequestFields": oc("low")})
    await send("max(b)",  extra_top={"additionalModelRequestFields": oc("max")})

asyncio.run(main())
