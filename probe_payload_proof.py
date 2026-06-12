# -*- coding: utf-8 -*-
"""Deterministic proof: build the actual kg payload from an OpenAI request with
reasoning_effort and show additionalModelRequestFields is emitted (native effort),
with NO <thinking_mode> injection in content."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv; load_dotenv()
from kiro.models_openai import ChatCompletionRequest
from kiro.converters_openai import build_kiro_payload

PROFILE = os.getenv("PROFILE_ARN", "arn:test")

def build(model, effort):
    req = ChatCompletionRequest(
        model=model,
        messages=[{"role": "user", "content": "What is 2+2?"}],
        reasoning_effort=effort,
        stream=True,
    )
    return build_kiro_payload(req, "conv-test", PROFILE)

for model, effort in [("claude-opus-4.8", "max"),
                      ("claude-opus-4.8", "minimal"),
                      ("claude-opus-4.8", None),
                      ("claude-sonnet-4", "high")]:
    p = build(model, effort)
    amrf = p.get("additionalModelRequestFields")
    content = p["conversationState"]["currentMessage"]["userInputMessage"]["content"]
    leaked = "<thinking_mode>" in content
    print(f"\n=== {model} effort={effort} ===")
    print(f"  additionalModelRequestFields: {json.dumps(amrf) if amrf else None}")
    print(f"  <thinking_mode> in content:   {leaked}")
    print(f"  content preview: {content[:60]!r}")
