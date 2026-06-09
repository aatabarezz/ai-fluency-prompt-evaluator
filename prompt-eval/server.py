import os, json, sqlite3, subprocess, textwrap, base64, io
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import anthropic, openpyxl, httpx

app = FastAPI()
DB = os.path.join(os.path.dirname(__file__), "sessions.db")

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


@app.get("/", response_class=HTMLResponse)
async def root():
    return Path(__file__).parent.joinpath("index.html").read_text()
