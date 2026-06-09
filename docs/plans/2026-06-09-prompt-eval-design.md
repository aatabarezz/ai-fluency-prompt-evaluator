# Prompt Evaluator — Design Doc
**Date:** 2026-06-09

## Overview

A standalone web app that ingests a user prompt, runs it against a selected Claude model, scores its effectiveness, and compares the original response side-by-side with an AI-optimized version. Sessions are archived with Excel export.

---

## Architecture

**Location:** `/Users/altanatabarut/Claude Code/prompt-eval/`

**Files:**
- `server.py` — FastAPI backend
- `index.html` — single-file frontend (HTML + CSS + vanilla JS)
- `sessions.db` — SQLite, auto-created on first run
- `requirements.txt`

**Stack:** Python 3.12, FastAPI, Anthropic SDK, openpyxl, uvicorn

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/evaluate` | SSE stream — runs all 4 Claude calls, saves session |
| GET | `/archive` | Paginated session list |
| GET | `/archive/{id}` | Full session detail |
| GET | `/archive/export` | `.xlsx` download (all or selected IDs via query param) |

---

## Evaluation Flow (4 Claude calls)

All calls stream via SSE. Token usage summed across all calls.

1. **Score call** — Claude evaluates the original prompt, returns JSON:
   ```json
   { "score": 7, "good": "...", "bad": "...", "fix": "..." }
   ```

2. **Left response call** — Claude answers the original prompt as-is.
   - If multimodal model: attachments (image/audio/video) included in messages array.
   - Tool use enabled: Claude may call `python_exec` or `web_search` mid-response.

3. **Optimize call** — Claude rewrites the prompt, names the technique(s) applied, returns JSON:
   ```json
   { "technique": "Chain-of-Thought + Role", "optimized_prompt": "..." }
   ```

4. **Right response call** — Claude answers the optimized prompt.
   - Same tool use and multimodal support as left call.

---

## Tool Use (Option C — Agentic)

Tools registered with Anthropic API:

- `python_exec(code: str) → str` — server runs code in a subprocess with timeout, returns stdout/stderr
- `web_search(query: str) → str` — server fetches search results (via DuckDuckGo or similar free API), returns top snippets

Both calls 2 and 4 support tool loops until Claude emits `stop_reason: end_turn`.

Tool calls and their results are displayed inline in the split panel.

---

## Multimodal Support

When a multimodal-capable model is selected (claude-opus-4-8, claude-sonnet-4-6, claude-haiku-4-5):
- File attach button appears in the input area
- Accepted types: image (png/jpg/gif/webp), audio (mp3/wav), video (mp4/webm)
- Files base64-encoded, sent as content blocks in the messages array
- Attachment filenames stored in `sessions.attachments` as JSON array

---

## UI Layout

### Tab 1 — Evaluate

```
┌────────────────────────────────────────────────────┐
│  [Model ▼]  [Prompt textarea .....................]  │
│             [📎 Attach (multimodal only)]  [Send]   │
│             [file preview strip]                    │
├────────────────────────────────────────────────────┤
│  SCORE: 7/10  ████████░░                           │
│  ✓ Good: ...  ✗ Bad: ...  → Fix: ...               │
├──────────────────────┬─────────────────────────────┤
│  Original Response   │  Optimized                   │
│  ─────────────────   │  Technique: CoT + Role       │
│  [tool calls inline] │  Prompt: "..."               │
│  [response text]     │  [tool calls inline]         │
│                      │  [response text]             │
├──────────────────────┴─────────────────────────────┤
│  [ Prefer Left ]  [ Prefer Both ]  [ Prefer Right ] │
├────────────────────────────────────────────────────┤
│  Tokens — Input: 1,240  Output: 892  Total: 2,132   │
└────────────────────────────────────────────────────┘
```

### Tab 2 — Archive

- Table columns: timestamp | model | prompt (truncated) | score | technique | preference | tokens
- Click row → expand full detail
- Checkboxes for selection
- Buttons: "Download Selected" / "Download All" → `.xlsx`

---

## SQLite Schema

```sql
CREATE TABLE sessions (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at          TEXT NOT NULL,
  model               TEXT NOT NULL,
  original_prompt     TEXT NOT NULL,
  score               INTEGER,
  score_good          TEXT,
  score_bad           TEXT,
  score_fix           TEXT,
  left_response       TEXT,
  left_tools_log      TEXT,   -- JSON array of {tool, input, output}
  optimized_prompt    TEXT,
  optimized_technique TEXT,
  right_response      TEXT,
  right_tools_log     TEXT,   -- JSON array of {tool, input, output}
  preference          TEXT,   -- 'left' | 'both' | 'right' | null
  attachments         TEXT,   -- JSON array of filenames
  tokens_input        INTEGER DEFAULT 0,
  tokens_output       INTEGER DEFAULT 0,
  tokens_total        INTEGER DEFAULT 0
);
```

---

## Models Supported (Claude only)

- claude-opus-4-8
- claude-sonnet-4-6
- claude-haiku-4-5-20251001

All support multimodal input.

---

## Key Constraints

- No auth — local dev tool
- Python sandbox: 10s timeout, no network access inside exec
- Web search: DuckDuckGo Instant Answer API (free, no key)
- Excel export: openpyxl, all columns included
