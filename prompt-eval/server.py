import os, json, re, sqlite3, subprocess, textwrap, base64, io
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import anthropic, openpyxl, httpx

app = FastAPI()
BASE = Path(__file__).parent
DB = str(BASE / "sessions.db")
KEY_FILE = BASE / ".api_key"

def load_api_key() -> str | None:
    """Load key from env, then fallback to .api_key file."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    if KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
            return key
    return None

load_api_key()

def init_db():
    with sqlite3.connect(DB) as con:
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

init_db()

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


def run_python(code: str) -> str:
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=10,
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
        r = httpx.get("https://api.duckduckgo.com/", params={"q": query, "format": "json", "no_redirect": "1"}, timeout=8)
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


def get_client() -> anthropic.Anthropic:
    key = load_api_key()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=key)


def claude_call_with_tools(model: str, messages: list, system: str = "") -> tuple[str, list, dict]:
    """Returns (full_text, tools_log, usage_totals)"""
    tools_log = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    kwargs = {"model": model, "max_tokens": 4096, "messages": messages, "tools": TOOLS}
    if system:
        kwargs["system"] = system

    for _ in range(10):
        response = get_client().messages.create(**kwargs)
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens
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
            messages = messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results}
            ]
            kwargs["messages"] = messages
        else:
            return full_text, tools_log, usage
    return full_text, tools_log, usage


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/evaluate")
async def evaluate(request: Request):
    body = await request.json()
    prompt: str = body["prompt"]
    model: str = body["model"]
    attachments: list = body.get("attachments", [])

    total_input = total_output = 0

    async def stream():
        nonlocal total_input, total_output
        try:
            session = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "original_prompt": prompt,
                "attachments": json.dumps([a["name"] for a in attachments]),
            }

            # CALL 1: Score
            yield sse("status", {"phase": "scoring"})
            score_system = textwrap.dedent("""
                You are an expert prompt engineer. Evaluate the user's prompt and return ONLY valid JSON:
                {"score": <1-10>, "good": "<what works>", "bad": "<what doesn't>", "fix": "<how to improve>"}
                No markdown, no explanation outside the JSON.
            """).strip()
            score_text, _, score_usage = claude_call_with_tools(
                model, [{"role": "user", "content": prompt}], system=score_system
            )
            total_input += score_usage["input_tokens"]
            total_output += score_usage["output_tokens"]
            try:
                score_data = json.loads(score_text.strip())
            except Exception:
                m = re.search(r'\{.*\}', score_text, re.DOTALL)
                score_data = json.loads(m.group()) if m else {"score": 5, "good": "", "bad": "", "fix": ""}
            session.update({
                "score": score_data.get("score"),
                "score_good": score_data.get("good", ""),
                "score_bad": score_data.get("bad", ""),
                "score_fix": score_data.get("fix", ""),
            })
            yield sse("score", score_data)

            # CALL 2: Left (original prompt)
            yield sse("status", {"phase": "left"})
            content_blocks = []
            for att in attachments:
                content_blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": att["type"], "data": att["b64"]}
                })
            content_blocks.append({"type": "text", "text": prompt})
            left_msgs = [{"role": "user", "content": content_blocks if attachments else prompt}]
            left_text, left_tools, left_usage = claude_call_with_tools(model, left_msgs)
            total_input += left_usage["input_tokens"]
            total_output += left_usage["output_tokens"]
            session["left_response"] = left_text
            session["left_tools_log"] = json.dumps(left_tools)
            yield sse("left", {"response": left_text, "tools": left_tools})

            # CALL 3: Optimize prompt
            yield sse("status", {"phase": "optimizing"})
            opt_system = textwrap.dedent("""
                You are an expert prompt engineer. Analyze the given prompt and rewrite it using the most
                impactful technique(s) from: Zero-shot, Few-shot, Chain-of-Thought, Role, Output Constraints,
                Step Decomposition, Generate Knowledge, Directional Stimulus.
                Return ONLY valid JSON:
                {"technique": "<technique name(s)>", "optimized_prompt": "<full rewritten prompt>"}
                No markdown, no explanation outside the JSON.
            """).strip()
            opt_text, _, opt_usage = claude_call_with_tools(
                model, [{"role": "user", "content": prompt}], system=opt_system
            )
            total_input += opt_usage["input_tokens"]
            total_output += opt_usage["output_tokens"]
            try:
                opt_data = json.loads(opt_text.strip())
            except Exception:
                m = re.search(r'\{.*\}', opt_text, re.DOTALL)
                opt_data = json.loads(m.group()) if m else {"technique": "", "optimized_prompt": prompt}
            session["optimized_prompt"] = opt_data.get("optimized_prompt", "")
            session["optimized_technique"] = opt_data.get("technique", "")
            yield sse("optimized_prompt", opt_data)

            # CALL 4: Right (optimized prompt)
            yield sse("status", {"phase": "right"})
            opt_prompt = opt_data.get("optimized_prompt", prompt)
            right_content = []
            for att in attachments:
                right_content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": att["type"], "data": att["b64"]}
                })
            right_content.append({"type": "text", "text": opt_prompt})
            right_msgs = [{"role": "user", "content": right_content if attachments else opt_prompt}]
            right_text, right_tools, right_usage = claude_call_with_tools(model, right_msgs)
            total_input += right_usage["input_tokens"]
            total_output += right_usage["output_tokens"]
            session["right_response"] = right_text
            session["right_tools_log"] = json.dumps(right_tools)
            yield sse("right", {"response": right_text, "tools": right_tools})

            # Save to DB
            session["tokens_input"] = total_input
            session["tokens_output"] = total_output
            session["tokens_total"] = total_input + total_output
            cols = ",".join(session.keys())
            placeholders = ",".join(["?"] * len(session))
            with sqlite3.connect(DB) as con:
                cur = con.execute(
                    f"INSERT INTO sessions ({cols}) VALUES ({placeholders})",
                    list(session.values())
                )
                session_id = cur.lastrowid

            yield sse("done", {
                "session_id": session_id,
                "tokens_input": total_input,
                "tokens_output": total_output,
                "tokens_total": total_input + total_output
            })
        except Exception as e:
            yield sse("error", {"message": str(e)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/sessions/{session_id}/preference")
async def save_preference(session_id: int, request: Request):
    body = await request.json()
    with sqlite3.connect(DB) as con:
        con.execute("UPDATE sessions SET preference=? WHERE id=?", (body["preference"], session_id))
    return {"ok": True}


@app.get("/archive")
async def archive_list(page: int = 1, per_page: int = 20):
    offset = (page - 1) * per_page
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT id, created_at, model,
                   substr(original_prompt, 1, 80) as prompt_preview,
                   score, optimized_technique, preference,
                   tokens_total
            FROM sessions ORDER BY id DESC LIMIT ? OFFSET ?
        """, (per_page, offset)).fetchall()
        total = con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    return {"total": total, "page": page, "per_page": per_page,
            "rows": [dict(r) for r in rows]}


@app.get("/archive/export")
async def archive_export(ids: str = ""):
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        if ids:
            id_list = [int(i) for i in ids.split(",") if i.strip().isdigit()]
            placeholders = ",".join("?" * len(id_list))
            rows = con.execute(
                f"SELECT * FROM sessions WHERE id IN ({placeholders})", id_list
            ).fetchall()
        else:
            rows = con.execute("SELECT * FROM sessions ORDER BY id DESC").fetchall()

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
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=prompt_sessions.xlsx"}
    )


@app.get("/archive/{session_id}")
async def archive_detail(session_id: int):
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Session not found")
    return dict(row)


@app.get("/config/status")
async def config_status():
    key = load_api_key()
    return {"has_key": bool(key), "masked": ("sk-ant-..." + key[-4:]) if key else None}

@app.post("/config/key")
async def config_set_key(request: Request):
    body = await request.json()
    key = (body.get("key") or "").strip()
    if not key.startswith("sk-ant-"):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid key format — must start with sk-ant-")
    KEY_FILE.write_text(key)
    os.environ["ANTHROPIC_API_KEY"] = key
    return {"ok": True}

@app.delete("/config/key")
async def config_delete_key():
    if KEY_FILE.exists():
        KEY_FILE.unlink()
    os.environ.pop("ANTHROPIC_API_KEY", None)
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
async def root():
    return Path(__file__).parent.joinpath("index.html").read_text()
