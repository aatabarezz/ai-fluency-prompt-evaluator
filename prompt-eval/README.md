# ✳ AI Fluency — Prompt Evaluator

> **The quality of your AI output is only as good as your prompt.**

Paste any prompt, get an instant score, see a side-by-side comparison with an expert-rewritten version, and learn exactly which techniques close the gap — so every conversation with AI works harder for you.

Built with FastAPI + vanilla JS. Powered by Claude.

---

## Why this exists

Most people use AI like a search engine — terse, vague, hopeful. They get generic answers and assume that's what AI does. It isn't. The difference between a 3/10 prompt and a 9/10 prompt is usually one or two missing ingredients: a role, a constraint, an example, a reasoning chain.

This tool makes that gap visible, measurable, and fixable — in seconds.

---

## What it does

### 1. Score your prompt (1–10)
Every prompt is evaluated across four axes: **Clarity**, **Context**, **Specificity**, and **Technique**. You get three verdicts:
- ✓ What works
- ✗ What doesn't  
- → Exactly how to fix it

| Score | Meaning |
|-------|---------|
| 1–3 | Vague single sentence. Model must guess intent, audience, format, scope. |
| 4–6 | Clear goal but missing constraints. You know what you want but haven't said who it's for or in what format. |
| 7–9 | Specific, contextual, well-structured. Role, audience, format, scope all defined. |
| 10 | Exemplary. Every best practice applied — role, audience, format, constraints, reasoning chain, examples. |

### 2. Side-by-side comparison
Your original prompt runs on the left. An AI-rewritten, expert-optimized version runs on the right. See the difference in response quality, depth, and structure instantly.

### 3. Optimized prompt score
The rewritten prompt also gets scored — so you can see the jump: *3/10 → 9/10*.

### 4. Token transparency
See exactly how many tokens your original prompt used vs. the optimized version — and the total cost across all 5 calls.

| Column | What it counts |
|--------|---------------|
| Prompt Tokens | Tokens in your original prompt (call 2 input only — no system overhead) |
| Response Tokens | Tokens Claude generated answering your prompt |
| Opt Prompt Tokens | Tokens in the rewritten prompt |
| Opt Response Tokens | Tokens Claude generated answering the optimized prompt |
| Total Tokens | Every token across all 5 calls |

### 5. Preference selector
Mark which response you preferred. Tracked in the archive.

### 6. Archive + Excel export
Every session saved to SQLite. Export any selection to `.xlsx` for review, training, or reporting.

---

## The 6 prompt engineering techniques

The optimizer applies the most impactful combination of these per prompt:

1. **Define a role** — "You are a senior software engineer…"
2. **Provide context** — audience, background, purpose, constraints
3. **Specify output constraints** — format, length, tone, sections
4. **Show an example** — one concrete before/after removes more ambiguity than a paragraph of instructions
5. **Break into steps** — for complex tasks, number sub-tasks explicitly
6. **Ask it to think first** — "Before answering, think through the key factors…"

---

## How it works (5-call pipeline)

```
Your prompt
    │
    ├─ CALL 1: Score original prompt → score JSON (1–10, good/bad/fix)
    ├─ CALL 2: Generate original response ← your actual prompt
    ├─ CALL 3: Rewrite/optimize prompt → optimized_prompt + technique names
    ├─ CALL 4: Generate optimized response ← rewritten prompt
    └─ CALL 5: Score optimized prompt → score JSON for comparison
```

All 5 calls stream via SSE — results appear in real time as each call completes.

---

## Setup

### Requirements
- Python 3.12+
- An [Anthropic API key](https://console.anthropic.com/settings/keys)

### Install & run

```bash
git clone https://github.com/aatabarezz/ai-fluency-prompt-evaluator
cd ai-fluency-prompt-evaluator/prompt-eval

pip install -r requirements.txt
uvicorn server:app --port 8001
```

Open http://localhost:8001 — on first load you'll be prompted to enter your API key (stored locally in `.api_key`, never committed).

---

## Models

| Model | Speed | Best for |
|-------|-------|----------|
| Haiku 4.5 | Fast & lightweight | Quick iteration, low cost |
| Sonnet 4.5 | Balanced | Most use cases |
| Opus 4 | Most capable | Complex reasoning, high-stakes prompts |

All models support multimodal input (images). Attach files with the 📎 button.

---

## Tool use

Claude can call two tools during response generation:

- **`python_exec`** — runs Python code in a sandboxed subprocess (10s timeout). Useful for math, data analysis, algorithms.
- **`web_search`** — queries DuckDuckGo Instant Answers. Useful for current events, factual lookup.

---

## Project structure

```
prompt-eval/
├── server.py        # FastAPI backend — 5-call SSE stream, archive, export
├── index.html       # Single-file frontend — all HTML/CSS/JS
├── requirements.txt
└── sessions.db      # SQLite — auto-created, gitignored
```

---

## Security

- `.api_key` is in `.gitignore` and has never been committed
- `sessions.db` is gitignored
- No auth — this is a local dev tool; do not expose port 8001 publicly

---

## License

MIT
