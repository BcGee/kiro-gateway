# Native Reasoning Investigation & Fix

## TL;DR

As of 2026-06, the Kiro backend (`runtime.{region}.kiro.dev`) emits the model's
**native extended thinking** as a dedicated `reasoningContentEvent` channel —
distinct from `assistantResponseEvent`. The old kg parser had no pattern for it,
so it **silently discarded all native reasoning** while *still* injecting the
legacy "fake reasoning" `<thinking_mode>` prompt tags. This branch captures the
native channel and gates off the now-redundant (and harmful) prompt injection.

## How we proved it

Raw AWS event-stream dump from a plain request (`probe_decode.py`, no injection,
prompt "What is 17 * 23?") on `claude-opus-4.8`:

```
[reasoningContentEvent] {"text":" I'm working through the multiplication..."}
[reasoningContentEvent] {"signature":"Eu0BCmMIDhAB..."}   <- Anthropic ext-thinking signature
[assistantResponseEvent] {"content":"17 * 23 = 391."}
```

Event types emitted by the backend:
`assistantResponseEvent`, `reasoningContentEvent`, `contextUsageEvent`, `meteringEvent`.

The model reasons natively by default on opus-4.8 — no `<thinking_mode>` tag was
sent. The `signature` field is the cryptographic marker of Anthropic native
extended thinking.

### Probing native control fields (negative result)

`probe_native_reasoning.py` sent these extra fields; all were **ignored** by the
backend (byte-identical responses), so native reasoning is currently
model-default with no exposed effort knob through this endpoint:
- `userInputMessage.reasoningConfig = {type: enabled, budgetTokens}`
- `conversationState.inferenceConfig.reasoning = {effort}`
- `userInputMessage.thinking = {type: enabled, budget_tokens}`

(Open question: how to tune native effort / disable it. The kiro-cli "Max"
selector is a client feature; these payload fields don't move the backend.)

## 3-way live A/B (`probe_ab_compare.py`)

Identical complex prompt (two-train meeting-time problem, `reasoning_effort=high`,
streaming) sent to three instances of kg:

| Instance | code | NATIVE_REASONING | reasoning_content | answer | TTFC |
|---|---|---|---|---|---|
| port 9300 | **old main (pre-change)** | n/a | **0 chars** | 559 | 10.4s |
| port 9100 | this branch | true | 546 chars | 591 | 8.2s |
| port 9200 | this branch | false | 652 chars | 753 | 6.9s |

**Interpretation**
- Old main returns **zero** reasoning_content — the native channel is dropped on
  the floor by the pattern-less parser. The fix recovers it.
- Both new-code instances surface reasoning, because the parser change captures
  `reasoningContentEvent` regardless of the `NATIVE_REASONING` flag. The flag
  controls only (a) whether `<thinking_mode>` tags are injected into the prompt
  and (b) whether the legacy `<thinking>`-tag ThinkingParser runs.
- Old main is also the slowest to first content (10.4s) — consistent with it
  wasting the prompt on fake-reasoning injection that the natively-thinking model
  no longer answers with parseable `<thinking>` tags.

## What changed

- `kiro/parsers.py` — `EVENT_PATTERNS` recognizes `reasoningContentEvent`
  (`{"text":...}` → `reasoning` event; `{"signature":...}` → swallowed metadata).
- `kiro/streaming_core.py` — `reasoning` events become `thinking` `KiroEvent`s
  (→ `reasoning_content` for OpenAI, native thinking blocks for Anthropic).
  Native deltas bypass the tag-detecting ThinkingParser. ThinkingParser is not
  created at all when native reasoning is active.
- `kiro/converters_core.py` — `inject_thinking_tags()` returns content unchanged
  when native reasoning is active (no prompt pollution).
- `kiro/config.py` — `NATIVE_REASONING` env flag, default **true**. Set
  `NATIVE_REASONING=false` to restore the old fake-injection behavior.

## Recommendation

Keep `NATIVE_REASONING=true` (default). Native reasoning is the real model
thinking, costs no prompt-budget hack, and is faster to first token. The fake
path remains available as a fallback for any model that does *not* emit
`reasoningContentEvent`.

## Repro scripts (kept in repo root, not packaged)
- `probe_decode.py` — decode raw event-stream, show native reasoning vs answer.
- `probe_native_reasoning.py` — probe whether native control fields are honored.
- `probe_ab_compare.py` — 3-way live comparison (needs instances on 9100/9200/9300).
