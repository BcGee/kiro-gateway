# -*- coding: utf-8 -*-
"""
E2E: send reasoning_effort via the OpenAI-compatible endpoint to the patched kg
(port 9400, NATIVE_REASONING=true) and confirm:
  1. native effort actually scales reasoning_content (low vs max)
  2. the gateway emits no <thinking_mode> prompt injection (native path)
"""
import json, os, sys, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv; load_dotenv()

KEY = os.getenv("PROXY_API_KEY", "").strip('"')
URL = "http://127.0.0.1:9400/v1/chat/completions"
MODEL = "claude-opus-4.8"
PROMPT = "Estimate how many piano tuners work in Chicago. Reason it out."


def call(effort):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}], "stream": True}
    if effort:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    reasoning, content = "", ""
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            try: d = json.loads(data)["choices"][0]["delta"]
            except: continue
            if d.get("reasoning_content"): reasoning += d["reasoning_content"]
            if d.get("content"): content += d["content"]
    leaked = "<thinking_mode>" in content or "<max_thinking_length>" in content
    return reasoning, content, leaked


if __name__ == "__main__":
    print(f"{'effort':10} | reasoning chars | answer chars | <thinking_mode> leaked")
    print("-"*70)
    for eff in ["low", "low", "max", "max", "xhigh"]:
        r, c, leaked = call(eff)
        print(f"{eff:10} | {len(r):15} | {len(c):12} | {leaked}")
    print("\nSample reasoning @ max:")
    r, c, _ = call("max")
    print(r[:300])
