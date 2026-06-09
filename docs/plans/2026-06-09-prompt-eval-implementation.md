# Prompt Evaluator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone prompt evaluation web app that scores prompts, compares original vs optimized responses side-by-side, supports tool use and multimodal input, and archives all sessions with Excel export.

**Architecture:** FastAPI backend streams evaluation results via SSE across 4 sequential Claude API calls (score → left response → optimize → right response). A single-file HTML+JS frontend renders the split view, score panel, and archive tab. SQLite persists all sessions.

**Tech Stack:** Python 3.12, FastAPI, Anthropic SDK (`anthropic`), openpyxl, uvicorn, sqlite3 (stdlib), vanilla JS (no frameworks)

---

### Task 1: Scaffold project directory and dependencies

**Files:**
- Create: `prompt-eval/requirements.txt`
- Create: `prompt-eval/server.py` (skeleton only)
- Create: `prompt-eval/index.html` (skeleton only)

**Step 1: Create directory and requirements.txt**

```bash
mkdir -p "/Users/altanatabarut/Claude Code/prompt-eval"
```

`prompt-eval/requirements.txt`:
```
fastapi==0.115.0
uvicorn==0.30.6
anthropic>=0.40.0
openpyxl>=3.1.0
httpx>=0.27.0
```

**Step 2: Create server.py skeleton**

```python
import os, json, sqlite3, subprocess, textwrap, base64, io
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
import anthropic, openpyxl, httpx

app = FastAPI()
DB = os.path.join(os.path.dirname(__file__), "sessions.db")

# DB init — run at startup
def init_db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        model TEXT NOT NULL,
        original_prompt TEXT NOT NULL,
        score INTEGER,
        score_good TEXT, score_bad TEXT, score_fix TEXT,
        left_response TEXT, left_tools_log TEXT,
        optimized_prompt TEXT, optimized_technique TEXT,
        right_response TEXT, right_tools_log TEXT,
        preference TEXT,
        attachments TEXT,
        tokens_input INTEGER DEFAULT 0,
        tokens_output INTEGER DEFAULT 0,
        tokens_total INTEGER DEFAULT 0
    )""")
    con.commit(); con.close()

init_db()

@app.get("/", response_class=HTMLResponse)
async def root():
    return open(os.path.join(os.path.dirname(__file__), "index.html")).read()
```

**Step 3: Create index.html skeleton**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Prompt Evaluator</title>
<style>
  /* styles go here in Task 5 */
</style>
</head>
<body>
  <div id="app">
    <nav>
      <button class="tab active" data-tab="evaluate">Evaluate</button>
      <button class="tab" data-tab="archive">Archive</button>
    </nav>
    <div id="evaluate" class="panel active"><!-- Task 2 --></div>
    <div id="archive" class="panel"><!-- Task 4 --></div>
  </div>
<script>
  // JS goes here in Tasks 3 & 4
</script>
</body>
</html>
```

**Step 4: Install dependencies**

```bash
cd "/Users/altanatabarut/Claude Code/prompt-eval"
pip install -r requirements.txt
```

Expected: all packages install cleanly.

**Step 5: Smoke-test server starts**

```bash
cd "/Users/altanatabarut/Claude Code/prompt-eval"
uvicorn server:app --port 8001 --reload &
curl -s http://localhost:8001/ | head -5
```

Expected: HTML skeleton returned.

**Step 6: Commit**

```bash
git add prompt-eval/
git commit -m "feat: scaffold prompt-eval app"
```

---

### Task 2: Backend — tool definitions and helper functions

**Files:**
- Modify: `prompt-eval/server.py`

**Step 1: Add tool definitions**

Append to `server.py` after `init_db()`:

```python
TOOLS = [
    {
        "name": "python_exec",
        "description": "Execute Python code and return stdout/stderr. Use for math, data analysis, scientific computation.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to execute"}},
            "required": ["code"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the web for current information. Returns top result snippets.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"]
        }
    }
]

MODELS = ["claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5-20251001"]
```

**Step 2: Add tool executor functions**

```python
def run_python(code: str) -> str:
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "PYTHONPATH": ""}  # no network inside sandbox
        )
        out = result.stdout[-3000:] if result.stdout else ""
        err = result.stderr[-1000:] if result.stderr else ""
        return (out + ("\nSTDERR: " + err if err else "")).strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out (10s limit)"
    except Exception as e:
        return f"Error: {e}"

def web_search(query: str) -> str:
    try:
        url = f"https://api.duckduckgo.com/?q={httpx.URL(query)}&format=json&no_redirect=1"
        r = httpx.get(f"https://api.duckduckgo.com/", params={"q": query, "format": "json", "no_redirect": "1"}, timeout=8)
        data = r.json()
        results = []
        if data.get("AbstractText"):
            results.append(data["AbstractText"])
        for topic in data.get("RelatedTopics", [])[:4]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(topic["Text"])
        return "\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search error: {e}"

def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "python_exec":
        return run_python(inputs["code"])
    elif name == "web_search":
        return web_search(inputs["query"])
    return "Unknown tool"
```

**Step 3: Commit**

```bash
git add prompt-eval/server.py
git commit -m "feat: add tool definitions and executors"
```

---

### Task 3: Backend — 4-call evaluation SSE endpoint

**Files:**
- Modify: `prompt-eval/server.py`

**Step 1: Add Claude call helper with tool loop**

```python
client = anthropic.Anthropic()

def claude_call_with_tools(model: str, messages: list, system: str = "", json_mode: bool = False) -> tuple[str, list, dict]:
    """Returns (full_text, tools_log, usage_totals)"""
    tools_log = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    
    kwargs = {"model": model, "max_tokens": 4096, "messages": messages, "tools": TOOLS}
    if system:
        kwargs["system"] = system
    if json_mode:
        # Guide toward JSON by prefilling assistant turn
        pass

    while True:
        response = client.messages.create(**kwargs)
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens

        # Collect text content
        text_parts = [b.text for b in response.content if hasattr(b, "text")]
        full_text = "".join(text_parts)

        if response.stop_reason == "end_turn":
            return full_text, tools_log, usage

        if response.stop_reason == "tool_use":
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            tool_results = []
            for tu in tool_uses:
                result = dispatch_tool(tu.name, tu.input)
                tools_log.append({"tool": tu.name, "input": tu.input, "output": result})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result
                })
            # Append assistant turn and tool results, loop
            messages = messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results}
            ]
            kwargs["messages"] = messages
        else:
            return full_text, tools_log, usage
```

**Step 2: Add SSE helper**

```python
def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
```

**Step 3: Add /evaluate endpoint**

```python
@app.post("/evaluate")
async def evaluate(request: Request):
    body = await request.json()
    prompt: str = body["prompt"]
    model: str = body["model"]
    attachments: list = body.get("attachments", [])  # [{name, type, b64}]

    total_input = total_output = 0

    async def stream():
        nonlocal total_input, total_output
        session = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "original_prompt": prompt,
            "attachments": json.dumps([a["name"] for a in attachments]),
        }

        # ── CALL 1: Score ────────────────────────────────────────────────
        yield sse("status", {"phase": "scoring"})
        score_system = textwrap.dedent("""
            You are an expert prompt engineer. Evaluate the user's prompt and return ONLY valid JSON:
            {"score": <1-10>, "good": "<what works>", "bad": "<what doesn't>", "fix": "<how to improve>"}
            No markdown, no explanation outside the JSON.
        """).strip()
        score_text, _, score_usage = claude_call_with_tools(model, [{"role":"user","content":prompt}], system=score_system)
        total_input += score_usage["input_tokens"]
        total_output += score_usage["output_tokens"]
        try:
            score_data = json.loads(score_text.strip())
        except Exception:
            import re
            m = re.search(r'\{.*\}', score_text, re.DOTALL)
            score_data = json.loads(m.group()) if m else {"score":5,"good":"","bad":"","fix":""}
        session.update({
            "score": score_data.get("score"),
            "score_good": score_data.get("good",""),
            "score_bad": score_data.get("bad",""),
            "score_fix": score_data.get("fix",""),
        })
        yield sse("score", score_data)

        # ── CALL 2: Left (original prompt) ───────────────────────────────
        yield sse("status", {"phase": "left"})
        content_blocks = []
        for att in attachments:
            media = att["type"]
            content_blocks.append({"type":"image","source":{"type":"base64","media_type":media,"data":att["b64"]}})
        content_blocks.append({"type":"text","text":prompt})
        left_msgs = [{"role":"user","content":content_blocks if attachments else prompt}]
        left_text, left_tools, left_usage = claude_call_with_tools(model, left_msgs)
        total_input += left_usage["input_tokens"]
        total_output += left_usage["output_tokens"]
        session["left_response"] = left_text
        session["left_tools_log"] = json.dumps(left_tools)
        yield sse("left", {"response": left_text, "tools": left_tools})

        # ── CALL 3: Optimize prompt ──────────────────────────────────────
        yield sse("status", {"phase": "optimizing"})
        opt_system = textwrap.dedent("""
            You are an expert prompt engineer. Analyze the given prompt and rewrite it using the most
            impactful technique(s) from: Zero-shot, Few-shot, Chain-of-Thought, Role, Output Constraints,
            Step Decomposition, Generate Knowledge, Directional Stimulus.
            Return ONLY valid JSON:
            {"technique": "<technique name(s)>", "optimized_prompt": "<full rewritten prompt>"}
            No markdown, no explanation outside the JSON.
        """).strip()
        opt_text, _, opt_usage = claude_call_with_tools(model, [{"role":"user","content":prompt}], system=opt_system)
        total_input += opt_usage["input_tokens"]
        total_output += opt_usage["output_tokens"]
        try:
            opt_data = json.loads(opt_text.strip())
        except Exception:
            import re
            m = re.search(r'\{.*\}', opt_text, re.DOTALL)
            opt_data = json.loads(m.group()) if m else {"technique":"","optimized_prompt":prompt}
        session["optimized_prompt"] = opt_data.get("optimized_prompt","")
        session["optimized_technique"] = opt_data.get("technique","")
        yield sse("optimized_prompt", opt_data)

        # ── CALL 4: Right (optimized prompt) ────────────────────────────
        yield sse("status", {"phase": "right"})
        opt_prompt = opt_data.get("optimized_prompt", prompt)
        right_content = []
        for att in attachments:
            right_content.append({"type":"image","source":{"type":"base64","media_type":att["type"],"data":att["b64"]}})
        right_content.append({"type":"text","text":opt_prompt})
        right_msgs = [{"role":"user","content":right_content if attachments else opt_prompt}]
        right_text, right_tools, right_usage = claude_call_with_tools(model, right_msgs)
        total_input += right_usage["input_tokens"]
        total_output += right_usage["output_tokens"]
        session["right_response"] = right_text
        session["right_tools_log"] = json.dumps(right_tools)
        yield sse("right", {"response": right_text, "tools": right_tools})

        # ── Save to DB ───────────────────────────────────────────────────
        session["tokens_input"] = total_input
        session["tokens_output"] = total_output
        session["tokens_total"] = total_input + total_output
        cols = ",".join(session.keys())
        placeholders = ",".join(["?"] * len(session))
        con = sqlite3.connect(DB)
        cur = con.execute(f"INSERT INTO sessions ({cols}) VALUES ({placeholders})", list(session.values()))
        session_id = cur.lastrowid
        con.commit(); con.close()

        yield sse("done", {
            "session_id": session_id,
            "tokens_input": total_input,
            "tokens_output": total_output,
            "tokens_total": total_input + total_output
        })

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```

**Step 4: Add preference save endpoint**

```python
@app.post("/sessions/{session_id}/preference")
async def save_preference(session_id: int, request: Request):
    body = await request.json()
    con = sqlite3.connect(DB)
    con.execute("UPDATE sessions SET preference=? WHERE id=?", (body["preference"], session_id))
    con.commit(); con.close()
    return {"ok": True}
```

**Step 5: Commit**

```bash
git add prompt-eval/server.py
git commit -m "feat: add /evaluate SSE endpoint with 4-call flow"
```

---

### Task 4: Backend — archive and export endpoints

**Files:**
- Modify: `prompt-eval/server.py`

**Step 1: Add /archive list endpoint**

```python
@app.get("/archive")
async def archive(page: int = 1, per_page: int = 20):
    offset = (page - 1) * per_page
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT id, created_at, model,
               substr(original_prompt, 1, 80) as prompt_preview,
               score, optimized_technique, preference,
               tokens_total
        FROM sessions ORDER BY id DESC LIMIT ? OFFSET ?
    """, (per_page, offset)).fetchall()
    total = con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    con.close()
    return {"total": total, "page": page, "per_page": per_page,
            "rows": [dict(r) for r in rows]}

@app.get("/archive/{session_id}")
async def archive_detail(session_id: int):
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    con.close()
    if not row: return {"error": "not found"}, 404
    return dict(row)
```

**Step 2: Add /archive/export endpoint**

```python
@app.get("/archive/export")
async def export(ids: str = ""):
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    if ids:
        id_list = [int(i) for i in ids.split(",") if i.strip().isdigit()]
        placeholders = ",".join("?" * len(id_list))
        rows = con.execute(f"SELECT * FROM sessions WHERE id IN ({placeholders})", id_list).fetchall()
    else:
        rows = con.execute("SELECT * FROM sessions ORDER BY id DESC").fetchall()
    con.close()

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sessions"
    if rows:
        ws.append(list(rows[0].keys()))
        for row in rows:
            ws.append(list(row))

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return StreamingResponse(buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=prompt_sessions.xlsx"})
```

**Step 3: Commit**

```bash
git add prompt-eval/server.py
git commit -m "feat: add archive list, detail, and Excel export endpoints"
```

---

### Task 5: Frontend — full index.html

**Files:**
- Modify: `prompt-eval/index.html`

This is the largest task. Replace the skeleton with the full single-file app.

**Step 1: Write the complete index.html**

The file has four logical sections: CSS, HTML structure, JS evaluate logic, JS archive logic.

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prompt Evaluator</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #f5f5f5; color: #1a1a1a; min-height: 100vh; }
#app { max-width: 1400px; margin: 0 auto; padding: 16px; }

/* Nav */
nav { display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 2px solid #e0e0e0; padding-bottom: 0; }
.tab { padding: 10px 20px; border: none; background: none; cursor: pointer; font-size: 15px;
       border-bottom: 3px solid transparent; margin-bottom: -2px; color: #666; }
.tab.active { border-bottom-color: #4a7c59; color: #4a7c59; font-weight: 600; }
.panel { display: none; } .panel.active { display: block; }

/* Input area */
.input-area { background: white; border-radius: 12px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 16px; }
.input-row { display: flex; gap: 10px; align-items: flex-start; }
select#model { padding: 8px 12px; border: 1px solid #ddd; border-radius: 8px; font-size: 14px; flex-shrink: 0; }
textarea#prompt { flex: 1; padding: 10px 12px; border: 1px solid #ddd; border-radius: 8px;
                  font-size: 14px; resize: vertical; min-height: 80px; font-family: inherit; }
.input-actions { display: flex; gap: 8px; flex-shrink: 0; align-items: flex-start; }
button#send { background: #4a7c59; color: white; border: none; border-radius: 8px;
              padding: 10px 20px; font-size: 14px; cursor: pointer; font-weight: 600; }
button#send:disabled { opacity: .5; cursor: default; }
button#attach-btn { background: #f0f0f0; border: 1px solid #ddd; border-radius: 8px;
                    padding: 10px 14px; cursor: pointer; font-size: 16px; display: none; }
#file-preview { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
.file-chip { background: #e8f0ea; border-radius: 6px; padding: 4px 10px; font-size: 12px;
             display: flex; align-items: center; gap: 6px; }
.file-chip button { background: none; border: none; cursor: pointer; color: #666; font-size: 14px; }

/* Score panel */
#score-panel { background: white; border-radius: 12px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08);
               margin-bottom: 16px; display: none; }
.score-header { display: flex; align-items: center; gap: 16px; margin-bottom: 12px; }
.score-badge { font-size: 28px; font-weight: 700; color: #4a7c59; }
.score-bar-bg { flex: 1; height: 10px; background: #e0e0e0; border-radius: 5px; }
.score-bar-fill { height: 100%; background: #4a7c59; border-radius: 5px; transition: width .5s; }
.score-details { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
.score-item { padding: 10px 14px; border-radius: 8px; font-size: 13px; line-height: 1.5; }
.score-item.good { background: #e8f5e9; border-left: 4px solid #43a047; }
.score-item.bad  { background: #fce4ec; border-left: 4px solid #e53935; }
.score-item.fix  { background: #fff8e1; border-left: 4px solid #ffa000; }
.score-item strong { display: block; margin-bottom: 4px; }

/* Status bar */
#status-bar { text-align: center; color: #666; font-size: 13px; padding: 8px; display: none; }
.pulse { animation: pulse 1.2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

/* Split panel */
#split-panel { display: none; margin-bottom: 16px; }
.split-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.response-card { background: white; border-radius: 12px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
.response-card h3 { font-size: 14px; font-weight: 600; color: #555; margin-bottom: 8px; }
.technique-badge { display: inline-block; background: #e8f0ea; color: #4a7c59; border-radius: 6px;
                   padding: 3px 10px; font-size: 12px; font-weight: 600; margin-bottom: 10px; }
.optimized-prompt-box { background: #f9f9f9; border: 1px solid #e0e0e0; border-radius: 8px;
                         padding: 10px; font-size: 12px; color: #555; margin-bottom: 12px;
                         white-space: pre-wrap; max-height: 120px; overflow-y: auto; }
.tool-call { background: #1e1e2e; color: #cdd6f4; border-radius: 6px; padding: 8px 12px;
             font-size: 12px; font-family: monospace; margin-bottom: 6px; }
.tool-call .tool-name { color: #89dceb; font-weight: bold; }
.tool-call .tool-result { color: #a6e3a1; margin-top: 4px; white-space: pre-wrap; max-height: 100px; overflow-y: auto; }
.response-text { font-size: 14px; line-height: 1.7; white-space: pre-wrap; }

/* Preference bar */
#pref-bar { background: white; border-radius: 12px; padding: 16px;
            box-shadow: 0 1px 4px rgba(0,0,0,.08); display: none;
            text-align: center; margin-bottom: 16px; }
#pref-bar p { font-size: 13px; color: #666; margin-bottom: 12px; }
.pref-btns { display: flex; gap: 10px; justify-content: center; }
.pref-btn { padding: 10px 24px; border-radius: 8px; border: 2px solid #ddd;
            background: white; cursor: pointer; font-size: 14px; font-weight: 600; transition: all .2s; }
.pref-btn:hover { border-color: #4a7c59; color: #4a7c59; }
.pref-btn.selected { background: #4a7c59; color: white; border-color: #4a7c59; }

/* Token footer */
#token-footer { background: white; border-radius: 12px; padding: 12px 16px;
                box-shadow: 0 1px 4px rgba(0,0,0,.08); display: none;
                font-size: 13px; color: #555; text-align: center; }
#token-footer span { margin: 0 16px; }
#token-footer strong { color: #1a1a1a; }

/* Archive tab */
.archive-toolbar { display: flex; gap: 8px; margin-bottom: 12px; align-items: center; }
.archive-toolbar button { padding: 8px 16px; border-radius: 8px; border: 1px solid #ddd;
                           background: white; cursor: pointer; font-size: 13px; }
.archive-toolbar button:hover { background: #f0f0f0; }
#archive-table { width: 100%; border-collapse: collapse; background: white;
                 border-radius: 12px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
#archive-table th { background: #f5f5f5; padding: 10px 14px; text-align: left; font-size: 13px; color: #555; }
#archive-table td { padding: 10px 14px; border-top: 1px solid #f0f0f0; font-size: 13px; vertical-align: top; }
#archive-table tr:hover td { background: #fafafa; cursor: pointer; }
.detail-row td { background: #fafafa; padding: 16px; }
.detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 13px; }
.detail-grid pre { white-space: pre-wrap; background: #f5f5f5; padding: 8px; border-radius: 6px; max-height: 200px; overflow-y: auto; }
.score-pill { display: inline-block; background: #e8f0ea; color: #4a7c59; border-radius: 12px;
              padding: 2px 10px; font-weight: 600; font-size: 12px; }
.pref-pill { display: inline-block; background: #e3f2fd; color: #1976d2; border-radius: 12px;
             padding: 2px 10px; font-size: 12px; }
</style>
</head>
<body>
<div id="app">
  <nav>
    <button class="tab active" data-tab="evaluate">Evaluate</button>
    <button class="tab" data-tab="archive">Archive</button>
  </nav>

  <!-- EVALUATE TAB -->
  <div id="evaluate" class="panel active">
    <div class="input-area">
      <div class="input-row">
        <select id="model">
          <option value="claude-sonnet-4-6">Sonnet 4.6</option>
          <option value="claude-opus-4-8">Opus 4.8</option>
          <option value="claude-haiku-4-5-20251001">Haiku 4.5</option>
        </select>
        <textarea id="prompt" placeholder="Enter your prompt here…"></textarea>
        <div class="input-actions">
          <button id="attach-btn" title="Attach file">📎</button>
          <button id="send">Send</button>
        </div>
      </div>
      <div id="file-preview"></div>
      <input type="file" id="file-input" multiple accept="image/*,audio/*,video/*" style="display:none">
    </div>

    <div id="status-bar"></div>
    <div id="score-panel"></div>
    <div id="split-panel">
      <div class="split-grid">
        <div class="response-card" id="left-card">
          <h3>Original Prompt Response</h3>
          <div id="left-tools"></div>
          <div id="left-response" class="response-text"></div>
        </div>
        <div class="response-card" id="right-card">
          <h3>Optimized Prompt Response</h3>
          <div id="technique-badge" class="technique-badge" style="display:none"></div>
          <div id="optimized-prompt-box" class="optimized-prompt-box" style="display:none"></div>
          <div id="right-tools"></div>
          <div id="right-response" class="response-text"></div>
        </div>
      </div>
    </div>
    <div id="pref-bar">
      <p>Which response did you prefer?</p>
      <div class="pref-btns">
        <button class="pref-btn" data-pref="left">Prefer Left</button>
        <button class="pref-btn" data-pref="both">Prefer Both</button>
        <button class="pref-btn" data-pref="right">Prefer Right</button>
      </div>
    </div>
    <div id="token-footer"></div>
  </div>

  <!-- ARCHIVE TAB -->
  <div id="archive" class="panel">
    <div class="archive-toolbar">
      <button id="dl-selected">Download Selected</button>
      <button id="dl-all">Download All</button>
      <span id="archive-count" style="margin-left:auto;color:#999;font-size:13px"></span>
    </div>
    <table id="archive-table">
      <thead>
        <tr>
          <th><input type="checkbox" id="check-all"></th>
          <th>Time</th><th>Model</th><th>Prompt</th>
          <th>Score</th><th>Technique</th><th>Preference</th><th>Tokens</th>
        </tr>
      </thead>
      <tbody id="archive-body"></tbody>
    </table>
  </div>
</div>

<script>
// ── Tab switching ────────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab,.panel').forEach(el => el.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'archive') loadArchive();
  });
});

// ── Multimodal file attach ───────────────────────────────────────────────────
const MULTIMODAL_MODELS = ["claude-sonnet-4-6","claude-opus-4-8","claude-haiku-4-5-20251001"];
const modelSel = document.getElementById('model');
const attachBtn = document.getElementById('attach-btn');
const fileInput = document.getElementById('file-input');
const filePreview = document.getElementById('file-preview');
let attachedFiles = []; // [{name, type, b64}]

modelSel.addEventListener('change', () => {
  attachBtn.style.display = MULTIMODAL_MODELS.includes(modelSel.value) ? '' : 'none';
});
attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', async () => {
  for (const f of fileInput.files) {
    const b64 = await toB64(f);
    attachedFiles.push({name: f.name, type: f.type, b64});
  }
  renderPreviews();
  fileInput.value = '';
});

function toB64(file) {
  return new Promise(res => {
    const r = new FileReader();
    r.onload = e => res(e.target.result.split(',')[1]);
    r.readAsDataURL(file);
  });
}

function renderPreviews() {
  filePreview.innerHTML = attachedFiles.map((f,i) =>
    `<div class="file-chip">📄 ${f.name} <button onclick="removeFile(${i})">✕</button></div>`
  ).join('');
}

window.removeFile = (i) => { attachedFiles.splice(i,1); renderPreviews(); };

// ── Evaluate ─────────────────────────────────────────────────────────────────
let currentSessionId = null;

document.getElementById('send').addEventListener('click', runEval);

function setStatus(msg, pulse=true) {
  const bar = document.getElementById('status-bar');
  bar.style.display = msg ? 'block' : 'none';
  bar.innerHTML = msg ? `<span class="${pulse?'pulse':''}">${msg}</span>` : '';
}

function renderTools(container, tools) {
  container.innerHTML = tools.map(t => `
    <div class="tool-call">
      <div><span class="tool-name">${t.tool}</span> — ${JSON.stringify(t.input)}</div>
      <div class="tool-result">${t.output}</div>
    </div>`).join('');
}

async function runEval() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return;
  const model = modelSel.value;
  const btn = document.getElementById('send');

  // Reset UI
  btn.disabled = true;
  currentSessionId = null;
  document.getElementById('score-panel').style.display = 'none';
  document.getElementById('split-panel').style.display = 'none';
  document.getElementById('pref-bar').style.display = 'none';
  document.getElementById('token-footer').style.display = 'none';
  document.getElementById('left-tools').innerHTML = '';
  document.getElementById('left-response').textContent = '';
  document.getElementById('right-tools').innerHTML = '';
  document.getElementById('right-response').textContent = '';
  document.getElementById('technique-badge').style.display = 'none';
  document.getElementById('optimized-prompt-box').style.display = 'none';
  document.querySelectorAll('.pref-btn').forEach(b => b.classList.remove('selected'));

  setStatus('⏳ Scoring your prompt…');

  const res = await fetch('/evaluate', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({prompt, model, attachments: attachedFiles})
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buf += decoder.decode(value, {stream:true});
    const lines = buf.split('\n');
    buf = lines.pop();
    let event = null;
    for (const line of lines) {
      if (line.startsWith('event: ')) event = line.slice(7).trim();
      if (line.startsWith('data: ') && event) {
        const data = JSON.parse(line.slice(6));
        handleSSE(event, data);
        event = null;
      }
    }
  }
  btn.disabled = false;
  setStatus('');
}

function handleSSE(event, data) {
  if (event === 'status') {
    const msgs = {scoring:'⏳ Scoring your prompt…', left:'🤖 Generating original response…',
                  optimizing:'✨ Optimizing your prompt…', right:'🚀 Running optimized prompt…'};
    setStatus(msgs[data.phase] || data.phase);
  }
  else if (event === 'score') {
    const pct = (data.score / 10) * 100;
    const color = data.score >= 7 ? '#43a047' : data.score >= 4 ? '#ffa000' : '#e53935';
    document.getElementById('score-panel').style.display = 'block';
    document.getElementById('score-panel').innerHTML = `
      <div class="score-header">
        <div class="score-badge" style="color:${color}">${data.score}/10</div>
        <div class="score-bar-bg"><div class="score-bar-fill" style="width:${pct}%;background:${color}"></div></div>
      </div>
      <div class="score-details">
        <div class="score-item good"><strong>✓ What works</strong>${data.good}</div>
        <div class="score-item bad"><strong>✗ What doesn't</strong>${data.bad}</div>
        <div class="score-item fix"><strong>→ How to improve</strong>${data.fix}</div>
      </div>`;
  }
  else if (event === 'left') {
    document.getElementById('split-panel').style.display = 'block';
    renderTools(document.getElementById('left-tools'), data.tools || []);
    document.getElementById('left-response').textContent = data.response;
  }
  else if (event === 'optimized_prompt') {
    const badge = document.getElementById('technique-badge');
    badge.textContent = '🎯 ' + data.technique;
    badge.style.display = 'inline-block';
    const box = document.getElementById('optimized-prompt-box');
    box.textContent = data.optimized_prompt;
    box.style.display = 'block';
  }
  else if (event === 'right') {
    renderTools(document.getElementById('right-tools'), data.tools || []);
    document.getElementById('right-response').textContent = data.response;
  }
  else if (event === 'done') {
    currentSessionId = data.session_id;
    document.getElementById('pref-bar').style.display = 'block';
    const tf = document.getElementById('token-footer');
    tf.style.display = 'block';
    tf.innerHTML = `<span>Input <strong>${data.tokens_input.toLocaleString()}</strong></span>` +
                   `<span>Output <strong>${data.tokens_output.toLocaleString()}</strong></span>` +
                   `<span>Total <strong>${data.tokens_total.toLocaleString()}</strong></span>`;
  }
}

// Preference buttons
document.querySelectorAll('.pref-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    if (!currentSessionId) return;
    document.querySelectorAll('.pref-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    await fetch(`/sessions/${currentSessionId}/preference`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({preference: btn.dataset.pref})
    });
  });
});

// ── Archive ───────────────────────────────────────────────────────────────────
let expandedRow = null;

async function loadArchive() {
  const res = await fetch('/archive');
  const data = await res.json();
  const tbody = document.getElementById('archive-body');
  document.getElementById('archive-count').textContent = `${data.total} sessions`;
  tbody.innerHTML = data.rows.map(r => `
    <tr data-id="${r.id}" onclick="toggleDetail(${r.id}, this)">
      <td onclick="event.stopPropagation()"><input type="checkbox" class="row-check" value="${r.id}"></td>
      <td>${new Date(r.created_at).toLocaleString()}</td>
      <td>${r.model.replace('claude-','')}</td>
      <td>${escHtml(r.prompt_preview)}${r.prompt_preview.length>=80?'…':''}</td>
      <td><span class="score-pill">${r.score ?? '-'}/10</span></td>
      <td>${r.optimized_technique || '-'}</td>
      <td>${r.preference ? `<span class="pref-pill">${r.preference}</span>` : '-'}</td>
      <td>${(r.tokens_total||0).toLocaleString()}</td>
    </tr>`).join('');
}

window.toggleDetail = async (id, tr) => {
  // Remove existing detail row if any
  const existing = document.getElementById('detail-'+id);
  if (existing) { existing.remove(); expandedRow = null; return; }
  if (expandedRow) { document.getElementById('detail-'+expandedRow)?.remove(); }
  expandedRow = id;

  const d = await (await fetch(`/archive/${id}`)).json();
  const detail = document.createElement('tr');
  detail.id = 'detail-'+id;
  detail.className = 'detail-row';
  detail.innerHTML = `<td colspan="8">
    <div class="detail-grid">
      <div><strong>Original Prompt</strong><pre>${escHtml(d.original_prompt)}</pre></div>
      <div><strong>Optimized Prompt (${escHtml(d.optimized_technique||'')})</strong><pre>${escHtml(d.optimized_prompt||'')}</pre></div>
      <div><strong>Left Response</strong><pre>${escHtml(d.left_response||'')}</pre></div>
      <div><strong>Right Response</strong><pre>${escHtml(d.right_response||'')}</pre></div>
    </div>
  </td>`;
  tr.after(detail);
};

function escHtml(s) {
  return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

document.getElementById('check-all').addEventListener('change', e => {
  document.querySelectorAll('.row-check').forEach(c => c.checked = e.target.checked);
});

document.getElementById('dl-selected').addEventListener('click', () => {
  const ids = [...document.querySelectorAll('.row-check:checked')].map(c=>c.value).join(',');
  if (!ids) return alert('Select at least one row.');
  window.location = '/archive/export?ids=' + ids;
});

document.getElementById('dl-all').addEventListener('click', () => {
  window.location = '/archive/export';
});
</script>
</body>
</html>
```

**Step 2: Commit**

```bash
git add prompt-eval/index.html
git commit -m "feat: complete single-file frontend"
```

---

### Task 6: End-to-end smoke test

**Step 1: Set ANTHROPIC_API_KEY and start server**

```bash
export ANTHROPIC_API_KEY=<your key>
cd "/Users/altanatabarut/Claude Code/prompt-eval"
uvicorn server:app --port 8001 --reload
```

**Step 2: Open browser and test**

- Navigate to `http://localhost:8001`
- Type a simple prompt: `Tell me about climate change.`
- Select Haiku 4.5 (cheapest)
- Hit Send
- Expected: score panel appears → left response streams → optimized prompt shown → right response streams → token footer appears

**Step 3: Test tool use**

- Prompt: `What is 2^100?`
- Expected: right panel shows a python_exec tool call with result `1267650600228229401496703205376`

**Step 4: Test archive**

- Click Archive tab
- Expected: row appears with score, technique, tokens
- Click row → detail expands showing all 4 text fields

**Step 5: Test Excel export**

- Click "Download All"
- Expected: `.xlsx` file downloads with all session columns

**Step 6: Final commit**

```bash
git add prompt-eval/
git commit -m "feat: prompt-eval app complete"
```

---

## Environment Notes

- **ANTHROPIC_API_KEY** must be set in the shell before starting uvicorn
- **Port**: 8001 (avoids conflict with microgpt on 8000)
- **Python**: 3.12
- **No database migrations needed** — `init_db()` creates the table on first run
```
