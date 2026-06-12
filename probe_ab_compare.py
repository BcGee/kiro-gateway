# -*- coding: utf-8 -*-
"""
A/B comparison: native reasoning (port 9100) vs legacy fake reasoning (port 9200).

Same code, only NATIVE_REASONING differs. Sends an identical complex prompt to
both via the OpenAI-compatible streaming endpoint and reports:
  - whether reasoning_content arrived (and how much)
  - whether the prompt was polluted with <thinking_mode> tags (fake path)
  - the final answer content
  - rough timing / token feel
"""
import json
import os
import sys
import time
import urllib.request

from dotenv import load_dotenv
load_dotenv()

KEY = os.getenv("PROXY_API_KEY", "").strip('"')
MODEL = "claude-opus-4.8"

COMPLEX_PROMPT = (
    "A train leaves City A at 9:00 AM traveling at 60 km/h toward City B, "
    "300 km away. Another train leaves City B at 9:30 AM traveling at 90 km/h "
    "toward City A on a parallel track. At what clock time do they meet, and "
    "how far from City A? Then briefly explain whether the 30-minute head start "
    "matters to the meeting point. Keep the final answer concise."
)


def call(port: int):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": COMPLEX_PROMPT}],
        "reasoning_effort": "high",
        "stream": True,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
    })
    reasoning, content = "", ""
    t0 = time.time()
    ttf_reasoning = ttf_content = None
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            delta = obj.get("choices", [{}])[0].get("delta", {})
            if "reasoning_content" in delta and delta["reasoning_content"]:
                if ttf_reasoning is None:
                    ttf_reasoning = time.time() - t0
                reasoning += delta["reasoning_content"]
            if "content" in delta and delta["content"]:
                if ttf_content is None:
                    ttf_content = time.time() - t0
                content += delta["content"]
    elapsed = time.time() - t0
    return {
        "reasoning": reasoning, "content": content, "elapsed": elapsed,
        "ttf_reasoning": ttf_reasoning, "ttf_content": ttf_content,
    }


def report(label, r):
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    print(f"elapsed: {r['elapsed']:.2f}s | ttf_reasoning: {r['ttf_reasoning']} | ttf_content: {r['ttf_content']}")
    print(f"reasoning_content chars: {len(r['reasoning'])}")
    print(f"content chars: {len(r['content'])}")
    # Detect prompt-injection leakage into the answer (fake-path failure mode)
    leaked = "<thinking_mode>" in r["content"] or "<max_thinking_length>" in r["content"]
    print(f"<thinking_mode> leaked into content: {leaked}")
    print("\n--- reasoning_content (first 500) ---")
    print(r["reasoning"][:500] or "(none)")
    print("\n--- content (first 500) ---")
    print(r["content"][:500] or "(none)")


if __name__ == "__main__":
    print("Sending identical complex prompt to all instances...")
    native = call(9100)
    fake = call(9200)
    oldmain = call(9300)
    report("NATIVE REASONING (port 9100, new code, NATIVE_REASONING=true)", native)
    report("LEGACY FAKE (port 9200, new code, NATIVE_REASONING=false)", fake)
    report("OLD MAIN (port 9300, pre-change code on main branch)", oldmain)

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    print(f"native (new, native ON) : reasoning={len(native['reasoning'])} chars, answer={len(native['content'])} chars, {native['elapsed']:.1f}s")
    print(f"fake   (new, native OFF) : reasoning={len(fake['reasoning'])} chars, answer={len(fake['content'])} chars, {fake['elapsed']:.1f}s")
    print(f"oldmain(pre-change)      : reasoning={len(oldmain['reasoning'])} chars, answer={len(oldmain['content'])} chars, {oldmain['elapsed']:.1f}s")
    print()
    print("KEY: old main should show reasoning=0 (native reasoningContentEvent dropped),")
    print("     proving the fix recovers reasoning the old parser silently discarded.")
