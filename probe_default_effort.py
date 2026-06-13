# -*- coding: utf-8 -*-
"""Verify DEFAULT_EFFORT_LEVEL=max is live on port 8000: compare default (expect max)
vs explicit low. If default >> low in reasoning volume, the global default is active."""
import json, os, urllib.request
os.chdir("/mnt/c/workspace/kiro-gateway")
from dotenv import load_dotenv; load_dotenv()
KEY = os.getenv("PROXY_API_KEY", "").strip('"')
URL = "http://localhost:8000/v1/chat/completions"
PROMPT = "Estimate the number of golf balls that fit in a school bus. Reason step by step."

def call(effort):
    body = {"model": "claude-opus-4.8", "messages": [{"role": "user", "content": PROMPT}], "stream": False}
    if effort:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    rc = m.get("reasoning_content") or ""
    c = m.get("content") or ""
    return len(rc), len(c)

print("A) default (no reasoning_effort -> expect max):")
for i in range(2):
    rc, c = call(None)
    print(f"   run{i+1}: reasoning={rc} chars, answer={c} chars")
print("B) explicit low (control):")
for i in range(2):
    rc, c = call("low")
    print(f"   run{i+1}: reasoning={rc} chars, answer={c} chars")
print("C) explicit max (upper bound):")
rc, c = call("max")
print(f"   run1: reasoning={rc} chars, answer={c} chars")
