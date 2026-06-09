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

@app.get("/", response_class=HTMLResponse)
async def root():
    return Path(__file__).parent.joinpath("index.html").read_text()
