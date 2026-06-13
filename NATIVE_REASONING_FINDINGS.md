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

## Native effort control (decoded from the Kiro client + verified live)

The Kiro "Max"/effort selector is **NOT prompt injection** — it sends a
top-level Bedrock Converse `additionalModelRequestFields` object. Decoded from
the client bundle (`extension.js`, function `Er`):

```js
function Er(level, schema) {
  switch (schema) {
    case "output_config":
      return { thinking: { type: "adaptive", display: "summarized" },
               output_config: { effort: level } };
    case "reasoning":
      return { reasoning: { effort: level } };
  }
}
```

Verified live against `runtime.kiro.dev/generateAssistantResponse`:
- Location: **top-level** `additionalModelRequestFields` (sibling of
  `conversationState`/`profileArn`). Placing it on `userInputMessage` or
  `conversationState` is silently ignored; placing the wrong schema returns a
  400 that names the expected enum.
- Effort value is **lowercase**: `low|medium|high|xhigh|max` (the client's
  `Yh2()` PascalCase is display-only; the backend enum is lowercase).
- Effort actually scales reasoning: e.g. opus-4.8 baseline ≈131 reasoning chars
  vs `max` ≈316–449 chars (probe_effort_field.py).

### Per-model effort schema (empirically probed — probe_all_schemas.py)

`runtime.kiro.dev` does not expose `ListAvailableModels`, so the schema cannot
be read dynamically. Probing every model with a deliberately-invalid effort and
reading the 400 message gives:

| schema | models |
|---|---|
| `output_config` | auto, claude-opus-4.6, claude-opus-4.7, claude-opus-4.8, claude-sonnet-4.6 |
| unsupported (400 "not supported for this model") | claude-sonnet-4, claude-sonnet-4.5, claude-haiku-4.5, claude-opus-4.5, deepseek-3.2, glm-5, minimax-m2.1, minimax-m2.5, qwen3-coder-next |

Only the 4.6+ Anthropic models accept native effort; older/non-Anthropic models
reject `additionalModelRequestFields` outright.

### What changed for effort

- `config.py` — `NATIVE_EFFORT_SCHEMA_BY_MODEL` (empirical map),
  `VALID_EFFORT_LEVELS`, `EFFORT_LEVEL_ALIASES` (minimal→low).
- `converters_core.py` — `build_native_effort_fields()` builds the correct
  `additionalModelRequestFields` per model/schema; payload assembly emits it at
  top level when native reasoning is on, the request asked for effort, and the
  model supports it. `ThinkingConfig` gained `effort_level`.
- `converters_openai.py` — `reasoning_effort` (incl. new `max`) flows into
  `effort_level`; `reasoning_effort_to_budget` gained `max`.
- `converters_anthropic.py` — Anthropic `thinking.budget_tokens` is bucketed
  into an effort level (`_budget_to_effort_level`).
- `models_openai.py` — `reasoning_effort` literal now accepts `max`.
- `get_thinking_system_prompt_addition()` is also gated off under native
  reasoning (it previously polluted the system prompt with `<thinking_mode>`
  documentation on every request — the second injection path).

### Global default effort (DEFAULT_EFFORT_LEVEL)

`DEFAULT_EFFORT_LEVEL` (env, default unset) applies an effort level to every
request that does NOT specify `reasoning_effort`. Set `DEFAULT_EFFORT_LEVEL=max`
to force maximum native reasoning on all supported models by default; an
explicit per-request `reasoning_effort` still overrides it. Models that don't
support effort (see schema table) are skipped safely. Currently set to `max`.

## Repro scripts (kept in repo root, not packaged)
- `probe_decode.py` — decode raw event-stream, show native reasoning vs answer.
- `probe_native_reasoning.py` — probe whether native control fields are honored.
- `probe_ab_compare.py` — 3-way live comparison (needs instances on 9100/9200/9300).
- `probe_effort_field.py` — locate where effort goes + confirm it scales reasoning.
- `probe_all_schemas.py` — empirically map each model's effort schema.
- `probe_payload_proof.py` — deterministic: show built payload has correct
  additionalModelRequestFields and no `<thinking_mode>` leakage.
- `probe_e2e_effort.py` — end-to-end through the OpenAI endpoint.
