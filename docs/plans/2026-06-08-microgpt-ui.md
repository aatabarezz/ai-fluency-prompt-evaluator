# MicroGPT Web UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a local web app that trains Karpathy's microgpt and shows live progress via a computation graph, loss chart, mid-training samples, and an inference panel.

**Architecture:** FastAPI backend embeds the full microgpt algorithm, runs training in a background thread, and streams progress via SSE. A single `index.html` (vanilla JS + Chart.js) provides four tabs: Load, Train, Report, Infer.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, vanilla JS, Chart.js (CDN)

---

### Task 1: Project scaffold + dependencies

**Files:**
- Create: `microgpt/requirements.txt`
- Create: `microgpt/server.py` (skeleton only)

**Step 1: Create requirements.txt**

```
fastapi==0.115.0
uvicorn==0.30.6
```

**Step 2: Install**

```bash
cd /Users/altanatabarut/Claude\ Code/microgpt && pip install -r requirements.txt
```

Expected: installs fastapi and uvicorn cleanly.

**Step 3: Create server.py skeleton**

```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pathlib

app = FastAPI()

@app.get("/")
def root():
    return HTMLResponse(pathlib.Path("index.html").read_text())

@app.get("/health")
def health():
    return {"ok": True}
```

**Step 4: Verify it starts**

```bash
cd /Users/altanatabarut/Claude\ Code/microgpt && uvicorn server:app --port 7860 &
sleep 2 && curl -s http://localhost:7860/health
```

Expected: `{"ok":true}`

```bash
kill %1
```

**Step 5: Commit**

```bash
cd /Users/altanatabarut/Claude\ Code && git add microgpt/ && git commit -m "feat: microgpt project scaffold"
```

---

### Task 2: Embed microgpt core in server.py

**Files:**
- Modify: `microgpt/server.py`

The full microgpt algorithm (Value class, GPT forward, tokenizer, training loop, inference) lives inside `server.py`. No external microgpt file — keep it self-contained.

**Step 1: Add all microgpt internals to server.py**

Paste the following block after the imports, before the FastAPI app definition:

```python
import math, random, os, threading, time
from dataclasses import dataclass, field
from typing import Optional

# ── microgpt core ──────────────────────────────────────────────────────────────

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._children: build(c)
                topo.append(v)
        build(self)
        self.grad = 1
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad


@dataclass
class TrainState:
    # config
    docs: list = field(default_factory=list)
    uchars: list = field(default_factory=list)
    BOS: int = 0
    vocab_size: int = 0
    n_embd: int = 16
    n_head: int = 4
    n_layer: int = 1
    block_size: int = 16
    num_steps: int = 1000
    # runtime
    step: int = 0
    loss: float = 0.0
    best_loss: float = float('inf')
    losses: list = field(default_factory=list)
    samples_log: list = field(default_factory=list)  # list of {step, samples}
    stage: str = "idle"   # idle|dataset|tokenizer|forward|loss|backward|adam|done
    running: bool = False
    state_dict: dict = field(default_factory=dict)
    params: list = field(default_factory=list)
    m: list = field(default_factory=list)
    v_buf: list = field(default_factory=list)
    num_params: int = 0
    start_time: float = 0.0
    events: list = field(default_factory=list)  # SSE event queue


_state = TrainState()
_lock = threading.Lock()


def _matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


def _init_model(state: TrainState):
    sd = {}
    sd['wte'] = _matrix(state.vocab_size, state.n_embd)
    sd['wpe'] = _matrix(state.block_size, state.n_embd)
    sd['lm_head'] = _matrix(state.vocab_size, state.n_embd)
    for i in range(state.n_layer):
        sd[f'layer{i}.attn_wq'] = _matrix(state.n_embd, state.n_embd)
        sd[f'layer{i}.attn_wk'] = _matrix(state.n_embd, state.n_embd)
        sd[f'layer{i}.attn_wv'] = _matrix(state.n_embd, state.n_embd)
        sd[f'layer{i}.attn_wo'] = _matrix(state.n_embd, state.n_embd)
        sd[f'layer{i}.mlp_fc1'] = _matrix(4 * state.n_embd, state.n_embd)
        sd[f'layer{i}.mlp_fc2'] = _matrix(state.n_embd, 4 * state.n_embd)
    state.state_dict = sd
    state.params = [p for mat in sd.values() for row in mat for p in row]
    state.num_params = len(state.params)
    state.m = [0.0] * state.num_params
    state.v_buf = [0.0] * state.num_params


def _linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def _softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def _gpt(token_id, pos_id, keys, values, state: TrainState):
    sd = state.state_dict
    n_layer, n_head = state.n_layer, state.n_head
    head_dim = state.n_embd // n_head
    tok_emb = sd['wte'][token_id]
    pos_emb = sd['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = _rmsnorm(x)
    for li in range(n_layer):
        x_residual = x
        x = _rmsnorm(x)
        q = _linear(x, sd[f'layer{li}.attn_wq'])
        k = _linear(x, sd[f'layer{li}.attn_wk'])
        v = _linear(x, sd[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = _softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = _linear(x_attn, sd[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = _rmsnorm(x)
        x = _linear(x, sd[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = _linear(x, sd[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    return _linear(x, sd['lm_head'])


def _infer(state: TrainState, n_samples=5, temperature=0.5):
    results = []
    for _ in range(n_samples):
        keys = [[] for _ in range(state.n_layer)]
        values = [[] for _ in range(state.n_layer)]
        token_id = state.BOS
        sample = []
        for pos_id in range(state.block_size):
            logits = _gpt(token_id, pos_id, keys, values, state)
            probs = _softmax([l / temperature for l in logits])
            token_id = random.choices(range(state.vocab_size), weights=[p.data for p in probs])[0]
            if token_id == state.BOS:
                break
            sample.append(state.uchars[token_id])
        results.append(''.join(sample))
    return results


def _push_event(state: TrainState, data: dict):
    import json
    state.events.append(f"data: {json.dumps(data)}\n\n")


def _train_thread(state: TrainState):
    lr = 0.01; beta1 = 0.85; beta2 = 0.99; eps = 1e-8
    state.start_time = time.time()

    for step in range(state.num_steps):
        if not state.running:
            break

        state.stage = "dataset"
        doc = state.docs[step % len(state.docs)]
        tokens = [state.BOS] + [state.uchars.index(ch) for ch in doc] + [state.BOS]
        n = min(state.block_size, len(tokens) - 1)

        state.stage = "forward"
        keys = [[] for _ in range(state.n_layer)]
        values = [[] for _ in range(state.n_layer)]
        losses_t = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = _gpt(token_id, pos_id, keys, values, state)
            probs = _softmax(logits)
            loss_t = -probs[target_id].log()
            losses_t.append(loss_t)

        state.stage = "loss"
        loss = (1 / n) * sum(losses_t)
        state.loss = loss.data
        if loss.data < state.best_loss:
            state.best_loss = loss.data
        state.losses.append({"step": step + 1, "loss": round(loss.data, 4)})

        state.stage = "backward"
        loss.backward()

        state.stage = "adam"
        lr_t = lr * (1 - step / state.num_steps)
        for i, p in enumerate(state.params):
            state.m[i] = beta1 * state.m[i] + (1 - beta1) * p.grad
            state.v_buf[i] = beta2 * state.v_buf[i] + (1 - beta2) * p.grad ** 2
            m_hat = state.m[i] / (1 - beta1 ** (step + 1))
            v_hat = state.v_buf[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps)
            p.grad = 0

        state.step = step + 1
        elapsed = time.time() - state.start_time
        eta = (elapsed / state.step) * (state.num_steps - state.step) if state.step > 0 else 0

        payload = {
            "type": "step",
            "step": state.step,
            "loss": round(state.loss, 4),
            "best_loss": round(state.best_loss, 4),
            "stage": state.stage,
            "eta": round(eta),
        }

        if step % 100 == 0 and step > 0:
            samples = _infer(state, n_samples=5, temperature=0.5)
            entry = {"step": state.step, "samples": samples}
            state.samples_log.append(entry)
            payload["samples"] = samples

        _push_event(state, payload)

    state.stage = "done"
    state.running = False
    _push_event(state, {"type": "done", "stage": "done"})
```

**Step 2: Verify Python parses cleanly**

```bash
cd /Users/altanatabarut/Claude\ Code/microgpt && python -c "import server; print('ok')"
```

Expected: `ok`

---

### Task 3: FastAPI endpoints

**Files:**
- Modify: `microgpt/server.py` (add endpoints after the core block)

**Step 1: Add all endpoints**

```python
import json, urllib.request
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
import pathlib

app = FastAPI()

@app.get("/")
def root():
    return HTMLResponse((pathlib.Path(__file__).parent / "index.html").read_text())

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/load")
def load(body: dict):
    """Load dataset from URL or raw text."""
    global _state
    with _lock:
        if body.get("url"):
            url = body["url"]
            tmp = "/tmp/microgpt_input.txt"
            urllib.request.urlretrieve(url, tmp)
            raw = open(tmp).read()
        else:
            raw = body.get("text", "")
        docs = [l.strip() for l in raw.strip().split('\n') if l.strip()]
        random.shuffle(docs)
        uchars = sorted(set(''.join(docs)))
        BOS = len(uchars)
        _state.docs = docs
        _state.uchars = uchars
        _state.BOS = BOS
        _state.vocab_size = len(uchars) + 1
        return {
            "num_docs": len(docs),
            "vocab_size": _state.vocab_size,
            "sample": docs[:5],
        }

@app.post("/train")
def start_train(body: dict):
    global _state
    with _lock:
        if _state.running:
            return {"error": "already running"}
        _state.num_steps = body.get("num_steps", 1000)
        _state.step = 0
        _state.losses = []
        _state.samples_log = []
        _state.best_loss = float('inf')
        _state.events = []
        _state.stage = "idle"
        _init_model(_state)
        _state.running = True
    t = threading.Thread(target=_train_thread, args=(_state,), daemon=True)
    t.start()
    return {"ok": True, "num_params": _state.num_params}

@app.post("/stop")
def stop_train():
    with _lock:
        _state.running = False
    return {"ok": True}

@app.get("/stream")
def stream():
    def generator():
        last = 0
        while True:
            with _lock:
                evs = _state.events[last:]
                last += len(evs)
                done = not _state.running and _state.stage == "done"
            for e in evs:
                yield e
            if done and not evs:
                break
            time.sleep(0.05)
    return StreamingResponse(generator(), media_type="text/event-stream")

@app.get("/state")
def get_state():
    with _lock:
        return {
            "step": _state.step,
            "num_steps": _state.num_steps,
            "loss": round(_state.loss, 4),
            "best_loss": round(_state.best_loss, 4),
            "stage": _state.stage,
            "running": _state.running,
            "losses": _state.losses[-200:],
            "samples_log": _state.samples_log,
            "num_params": _state.num_params,
            "vocab_size": _state.vocab_size,
        }

@app.post("/infer")
def infer(body: dict):
    with _lock:
        if not _state.state_dict:
            return {"error": "model not trained"}
        temperature = float(body.get("temperature", 0.5))
        n = int(body.get("n", 10))
        results = _infer(_state, n_samples=n, temperature=temperature)
    return {"samples": results}
```

**Step 2: Quick smoke test**

```bash
cd /Users/altanatabarut/Claude\ Code/microgpt && uvicorn server:app --port 7860 &
sleep 2
curl -s -X POST http://localhost:7860/load \
  -H "Content-Type: application/json" \
  -d '{"url":"https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"}' | python -m json.tool
kill %1
```

Expected: JSON with `num_docs`, `vocab_size`, `sample`.

**Step 3: Commit**

```bash
cd /Users/altanatabarut/Claude\ Code && git add microgpt/server.py && git commit -m "feat: microgpt FastAPI backend with SSE training stream"
```

---

### Task 4: index.html — shell + tab routing

**Files:**
- Create: `microgpt/index.html`

**Step 1: Write the HTML shell**

Four-tab layout, dark theme, Chart.js from CDN. Tab switching in vanilla JS.

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MicroGPT</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --accent: #6366f1; --accent2: #818cf8;
    --text: #e2e8f0; --muted: #64748b;
    --green: #22c55e; --red: #ef4444; --yellow: #f59e0b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; }
  header { padding: 16px 24px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 18px; font-weight: 700; letter-spacing: -0.5px; }
  header .badge { background: var(--accent); color: white; padding: 2px 8px; border-radius: 99px; font-size: 10px; }
  nav { display: flex; border-bottom: 1px solid var(--border); }
  nav button { flex: 1; padding: 12px; background: none; border: none; color: var(--muted); cursor: pointer; font-family: inherit; font-size: 12px; letter-spacing: 0.5px; text-transform: uppercase; border-bottom: 2px solid transparent; transition: all 0.15s; }
  nav button.active { color: var(--accent2); border-bottom-color: var(--accent); }
  nav button:hover:not(.active) { color: var(--text); }
  .tab { display: none; padding: 24px; }
  .tab.active { display: block; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 16px; }
  .card h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 14px; }
  input, textarea { width: 100%; background: var(--bg); border: 1px solid var(--border); color: var(--text); padding: 8px 12px; border-radius: 6px; font-family: inherit; font-size: 13px; outline: none; transition: border-color 0.15s; }
  input:focus, textarea:focus { border-color: var(--accent); }
  textarea { resize: vertical; min-height: 80px; }
  button.btn { display: inline-flex; align-items: center; gap: 6px; padding: 8px 16px; border-radius: 6px; border: none; cursor: pointer; font-family: inherit; font-size: 12px; font-weight: 600; letter-spacing: 0.3px; transition: all 0.15s; }
  button.btn-primary { background: var(--accent); color: white; }
  button.btn-primary:hover { background: var(--accent2); }
  button.btn-danger { background: #3f1818; color: var(--red); border: 1px solid #5f2020; }
  button.btn-danger:hover { background: #5f2020; }
  button.btn-ghost { background: var(--surface); color: var(--text); border: 1px solid var(--border); }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  .stat { text-align: center; }
  .stat .val { font-size: 28px; font-weight: 700; color: var(--accent2); }
  .stat .lbl { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; }
  .tag-idle { background: #1e2433; color: var(--muted); }
  .tag-active { background: #1e1e3a; color: var(--accent2); }
  .tag-done { background: #0f2a1a; color: var(--green); }
  .samples-log { max-height: 300px; overflow-y: auto; }
  .sample-entry { border-left: 2px solid var(--border); padding: 8px 12px; margin-bottom: 8px; }
  .sample-entry .step-label { font-size: 10px; color: var(--muted); margin-bottom: 4px; }
  .sample-entry .names { display: flex; flex-wrap: wrap; gap: 6px; }
  .sample-entry .name { background: var(--bg); border: 1px solid var(--border); padding: 2px 8px; border-radius: 4px; }
  .name-card { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px 16px; text-align: center; font-size: 15px; font-weight: 600; }
  .names-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 8px; }
  input[type=range] { -webkit-appearance: none; height: 4px; border-radius: 2px; background: var(--border); outline: none; }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%; background: var(--accent); cursor: pointer; }
  .flex-row { display: flex; align-items: center; gap: 12px; }
  .log { font-size: 11px; color: var(--muted); margin-top: 8px; }
</style>
</head>
<body>

<header>
  <h1>μGPT</h1>
  <span class="badge">microgpt</span>
  <span id="status-badge" class="tag tag-idle" style="margin-left:auto">idle</span>
</header>

<nav>
  <button class="active" onclick="switchTab('load')">① Load</button>
  <button onclick="switchTab('train')">② Train</button>
  <button onclick="switchTab('report')">③ Report</button>
  <button onclick="switchTab('infer')">④ Infer</button>
</nav>

<div id="tab-load" class="tab active"><!-- filled by JS --></div>
<div id="tab-train" class="tab"><!-- filled by JS --></div>
<div id="tab-report" class="tab"><!-- filled by JS --></div>
<div id="tab-infer" class="tab"><!-- filled by JS --></div>

<script>
// ── tab routing ────────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  const idx = ['load','train','report','infer'].indexOf(name);
  document.querySelectorAll('nav button')[idx].classList.add('active');
}
</script>

<!-- Tab content scripts follow in Task 5-8 -->
</body>
</html>
```

**Step 2: Open in browser to verify tabs switch**

```bash
cd /Users/altanatabarut/Claude\ Code/microgpt && uvicorn server:app --port 7860 &
sleep 1 && open http://localhost:7860
kill %1
```

---

### Task 5: Load tab JS

**Files:**
- Modify: `microgpt/index.html` (replace `<!-- Tab content scripts follow -->` comment)

**Step 1: Render load tab and wire up /load endpoint**

```html
<script>
// ── Load tab ───────────────────────────────────────────────────────────────────
document.getElementById('tab-load').innerHTML = `
<div class="card">
  <h2>Dataset Source</h2>
  <label style="color:var(--muted);font-size:11px">URL</label>
  <input id="ds-url" type="text" placeholder="https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
    value="https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt" style="margin-top:6px;margin-bottom:12px">
  <label style="color:var(--muted);font-size:11px">— or paste text —</label>
  <textarea id="ds-text" placeholder="One document per line..." style="margin-top:6px;margin-bottom:12px"></textarea>
  <button class="btn btn-primary" onclick="loadData()">Load Dataset</button>
  <div class="log" id="load-log"></div>
</div>
<div class="card" id="ds-preview" style="display:none">
  <h2>Preview</h2>
  <div class="grid3" id="ds-stats"></div>
  <div style="margin-top:16px;color:var(--muted);font-size:11px">Sample docs:</div>
  <div id="ds-sample" style="margin-top:8px"></div>
</div>
<div class="card">
  <h2>Training Config</h2>
  <div class="grid2">
    <div>
      <label style="color:var(--muted);font-size:11px">Steps</label>
      <input id="cfg-steps" type="number" value="1000" min="10" max="10000" style="margin-top:4px">
    </div>
    <div>
      <label style="color:var(--muted);font-size:11px">Temperature (inference)</label>
      <input id="cfg-temp" type="number" value="0.5" min="0.1" max="1.0" step="0.05" style="margin-top:4px">
    </div>
  </div>
</div>`;

async function loadData() {
  const url = document.getElementById('ds-url').value.trim();
  const text = document.getElementById('ds-text').value.trim();
  document.getElementById('load-log').textContent = 'Loading…';
  const res = await fetch('/load', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(url ? {url} : {text})
  });
  const data = await res.json();
  document.getElementById('load-log').textContent = '';
  document.getElementById('ds-preview').style.display = 'block';
  document.getElementById('ds-stats').innerHTML = `
    <div class="stat"><div class="val">${data.num_docs.toLocaleString()}</div><div class="lbl">Documents</div></div>
    <div class="stat"><div class="val">${data.vocab_size}</div><div class="lbl">Vocab Size</div></div>
    <div class="stat"><div class="val">ready</div><div class="lbl">Status</div></div>`;
  document.getElementById('ds-sample').innerHTML =
    data.sample.map(s => `<span class="tag tag-idle" style="margin:2px">${s}</span>`).join('');
}
</script>
```

---

### Task 6: Train tab — workflow graph

**Files:**
- Modify: `microgpt/index.html`

**Step 1: Add SVG workflow graph and start/stop controls**

```html
<script>
document.getElementById('tab-train').innerHTML = `
<div class="card">
  <h2>Pipeline</h2>
  <svg id="pipeline-svg" viewBox="0 0 760 80" style="width:100%;overflow:visible" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <filter id="glow"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    </defs>
  </svg>
</div>
<div class="card" id="gpt-subgraph-card" style="display:none">
  <h2>GPT Forward Pass</h2>
  <svg id="gpt-svg" viewBox="0 0 760 60" style="width:100%;overflow:visible" xmlns="http://www.w3.org/2000/svg"></svg>
</div>
<div class="card flex-row" style="flex-wrap:wrap;gap:12px">
  <button class="btn btn-primary" id="btn-start" onclick="startTraining()">▶ Start Training</button>
  <button class="btn btn-danger" id="btn-stop" onclick="stopTraining()" style="display:none">■ Stop</button>
  <span id="train-status" style="color:var(--muted);font-size:11px"></span>
</div>`;

const PIPELINE_NODES = [
  {id:'dataset', label:'Dataset'},
  {id:'tokenizer', label:'Tokenizer'},
  {id:'forward', label:'GPT Forward'},
  {id:'loss', label:'Loss'},
  {id:'backward', label:'Backward'},
  {id:'adam', label:'Adam'},
];

const GPT_NODES = [
  {id:'embed', label:'Embed'},
  {id:'rmsnorm', label:'RMSNorm'},
  {id:'attention', label:'Attention'},
  {id:'mlp', label:'MLP'},
  {id:'lmhead', label:'LM Head'},
];

function renderPipeline(svg, nodes, activeStageMap) {
  svg.innerHTML = '';
  const W = 760, H = 80;
  const nw = 90, nh = 36, gap = (W - nodes.length * nw) / (nodes.length + 1);
  nodes.forEach((node, i) => {
    const x = gap + i * (nw + gap);
    const y = (H - nh) / 2;
    const state = activeStageMap[node.id] || 'idle';
    const fill = state === 'active' ? '#1e1e3a' : state === 'done' ? '#0f2a1a' : '#1a1d27';
    const stroke = state === 'active' ? '#6366f1' : state === 'done' ? '#22c55e' : '#2a2d3a';
    const textColor = state === 'active' ? '#818cf8' : state === 'done' ? '#22c55e' : '#64748b';
    const filter = state === 'active' ? 'url(#glow)' : '';
    // connector line
    if (i > 0) {
      const prevX = gap + (i-1) * (nw + gap) + nw;
      const line = document.createElementNS('http://www.w3.org/2000/svg','line');
      line.setAttribute('x1', prevX); line.setAttribute('y1', H/2);
      line.setAttribute('x2', x); line.setAttribute('y2', H/2);
      line.setAttribute('stroke', '#2a2d3a'); line.setAttribute('stroke-width', '2');
      svg.appendChild(line);
    }
    const rect = document.createElementNS('http://www.w3.org/2000/svg','rect');
    rect.setAttribute('x', x); rect.setAttribute('y', y);
    rect.setAttribute('width', nw); rect.setAttribute('height', nh);
    rect.setAttribute('rx', 6); rect.setAttribute('fill', fill);
    rect.setAttribute('stroke', stroke); rect.setAttribute('stroke-width', '1.5');
    if (filter) rect.setAttribute('filter', filter);
    svg.appendChild(rect);
    const text = document.createElementNS('http://www.w3.org/2000/svg','text');
    text.setAttribute('x', x + nw/2); text.setAttribute('y', y + nh/2 + 5);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('fill', textColor); text.setAttribute('font-size', '11');
    text.setAttribute('font-family', 'monospace');
    text.textContent = node.label;
    svg.appendChild(text);
  });
}

function stageMap(activeStage, nodes) {
  const map = {};
  let passed = false;
  for (const n of nodes) {
    if (n.id === activeStage) { map[n.id] = 'active'; passed = true; }
    else if (!passed) map[n.id] = 'done';
    else map[n.id] = 'idle';
  }
  return map;
}

// initial idle render
setTimeout(() => {
  renderPipeline(document.getElementById('pipeline-svg'), PIPELINE_NODES, {});
}, 50);

async function startTraining() {
  const steps = parseInt(document.getElementById('cfg-steps')?.value || 1000);
  const res = await fetch('/train', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({num_steps: steps})
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  document.getElementById('btn-start').style.display = 'none';
  document.getElementById('btn-stop').style.display = 'inline-flex';
  document.getElementById('train-status').textContent = `${data.num_params.toLocaleString()} params`;
  startSSE();
}

async function stopTraining() {
  await fetch('/stop', {method:'POST'});
}
</script>
```

---

### Task 7: SSE listener + Report tab

**Files:**
- Modify: `microgpt/index.html`

**Step 1: Add SSE listener and report tab**

```html
<script>
// ── Report tab ────────────────────────────────────────────────────────────────
document.getElementById('tab-report').innerHTML = `
<div class="grid2" style="margin-bottom:16px">
  <div class="card">
    <h2>Training Stats</h2>
    <div class="grid3" id="stats-grid">
      <div class="stat"><div class="val" id="s-step">—</div><div class="lbl">Step</div></div>
      <div class="stat"><div class="val" id="s-loss">—</div><div class="lbl">Current Loss</div></div>
      <div class="stat"><div class="val" id="s-best">—</div><div class="lbl">Best Loss</div></div>
      <div class="stat"><div class="val" id="s-params">—</div><div class="lbl">Params</div></div>
      <div class="stat"><div class="val" id="s-vocab">—</div><div class="lbl">Vocab</div></div>
      <div class="stat"><div class="val" id="s-eta">—</div><div class="lbl">ETA (s)</div></div>
    </div>
  </div>
  <div class="card">
    <h2>Loss Curve</h2>
    <canvas id="loss-chart" height="120"></canvas>
  </div>
</div>
<div class="card">
  <h2>Mid-Training Samples</h2>
  <div class="samples-log" id="samples-log"><div style="color:var(--muted)">Samples appear every 100 steps…</div></div>
</div>`;

let lossChart = null;

function initChart() {
  const ctx = document.getElementById('loss-chart').getContext('2d');
  lossChart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'loss', data: [], borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,0.08)', borderWidth: 2, pointRadius: 0, tension: 0.3 }] },
    options: { animation: false, plugins: { legend: { display: false } }, scales: {
      x: { ticks: { color: '#64748b', maxTicksLimit: 8, font:{size:10} }, grid: { color: '#1e2433' } },
      y: { ticks: { color: '#64748b', font:{size:10} }, grid: { color: '#1e2433' } }
    }}
  });
}
setTimeout(initChart, 100);

// ── SSE ───────────────────────────────────────────────────────────────────────
let sse = null;
function startSSE() {
  if (sse) sse.close();
  sse = new EventSource('/stream');
  sse.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    updateStatusBadge(msg.stage);
    if (msg.type === 'step') {
      // update pipeline graph
      const pSvg = document.getElementById('pipeline-svg');
      if (pSvg) renderPipeline(pSvg, PIPELINE_NODES, stageMap(msg.stage, PIPELINE_NODES));
      // show GPT subgraph when in forward stage
      const subCard = document.getElementById('gpt-subgraph-card');
      if (subCard) {
        if (msg.stage === 'forward') {
          subCard.style.display = 'block';
          renderPipeline(document.getElementById('gpt-svg'), GPT_NODES, {attention:'active'});
        }
      }
      // stats
      const el = (id) => document.getElementById(id);
      if (el('s-step')) el('s-step').textContent = `${msg.step}/${_numSteps}`;
      if (el('s-loss')) el('s-loss').textContent = msg.loss;
      if (el('s-best')) el('s-best').textContent = msg.best_loss;
      if (el('s-eta')) el('s-eta').textContent = msg.eta;
      // chart
      if (lossChart && msg.step % 5 === 0) {
        lossChart.data.labels.push(msg.step);
        lossChart.data.datasets[0].data.push(msg.loss);
        if (lossChart.data.labels.length > 300) {
          lossChart.data.labels.shift();
          lossChart.data.datasets[0].data.shift();
        }
        lossChart.update('none');
      }
      // samples
      if (msg.samples) {
        const log = document.getElementById('samples-log');
        if (log) {
          const entry = document.createElement('div');
          entry.className = 'sample-entry';
          entry.innerHTML = `<div class="step-label">Step ${msg.step}</div><div class="names">${msg.samples.map(s=>`<span class="name">${s}</span>`).join('')}</div>`;
          log.prepend(entry);
        }
      }
    }
    if (msg.type === 'done') {
      sse.close();
      document.getElementById('btn-start').style.display = 'inline-flex';
      document.getElementById('btn-stop').style.display = 'none';
      document.getElementById('train-status').textContent = 'done';
      const pSvg = document.getElementById('pipeline-svg');
      if (pSvg) renderPipeline(pSvg, PIPELINE_NODES, Object.fromEntries(PIPELINE_NODES.map(n=>[n.id,'done'])));
    }
  };
}

let _numSteps = 1000;

function updateStatusBadge(stage) {
  const b = document.getElementById('status-badge');
  if (!b) return;
  b.textContent = stage;
  b.className = 'tag ' + (stage==='idle'||stage==='done' ? 'tag-idle' : stage==='done' ? 'tag-done' : 'tag-active');
}

// restore state on load
fetch('/state').then(r=>r.json()).then(s => {
  _numSteps = s.num_steps || 1000;
  if (s.num_params) {
    const el = (id) => document.getElementById(id);
    if (el('s-params')) el('s-params').textContent = s.num_params.toLocaleString();
    if (el('s-vocab')) el('s-vocab').textContent = s.vocab_size;
    if (el('s-step')) el('s-step').textContent = `${s.step}/${s.num_steps}`;
    if (el('s-loss')) el('s-loss').textContent = s.loss;
    if (el('s-best')) el('s-best').textContent = s.best_loss;
  }
  if (s.running) startSSE();
  if (s.losses && lossChart) {
    lossChart.data.labels = s.losses.map(l=>l.step);
    lossChart.data.datasets[0].data = s.losses.map(l=>l.loss);
    lossChart.update('none');
  }
});
</script>
```

---

### Task 8: Infer tab

**Files:**
- Modify: `microgpt/index.html`

**Step 1: Add inference tab**

```html
<script>
document.getElementById('tab-infer').innerHTML = `
<div class="card">
  <h2>Generate Names</h2>
  <div class="flex-row" style="margin-bottom:16px;flex-wrap:wrap">
    <div style="flex:1;min-width:200px">
      <label style="color:var(--muted);font-size:11px">Temperature: <span id="temp-val">0.5</span></label>
      <input type="range" id="infer-temp" min="0.1" max="1.0" step="0.05" value="0.5"
        oninput="document.getElementById('temp-val').textContent=this.value"
        style="width:100%;margin-top:6px">
    </div>
    <div style="flex:1;min-width:200px">
      <label style="color:var(--muted);font-size:11px">Count: <span id="count-val">10</span></label>
      <input type="range" id="infer-count" min="1" max="20" step="1" value="10"
        oninput="document.getElementById('count-val').textContent=this.value"
        style="width:100%;margin-top:6px">
    </div>
  </div>
  <button class="btn btn-primary" onclick="generate()">✦ Generate</button>
</div>
<div class="card" id="infer-output" style="display:none">
  <h2>Output</h2>
  <div class="names-grid" id="names-grid"></div>
</div>`;

async function generate() {
  const temperature = parseFloat(document.getElementById('infer-temp').value);
  const n = parseInt(document.getElementById('infer-count').value);
  const res = await fetch('/infer', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({temperature, n})
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  document.getElementById('infer-output').style.display = 'block';
  document.getElementById('names-grid').innerHTML =
    data.samples.map(s => `<div class="name-card">${s}</div>`).join('');
}
</script>
```

**Step 2: Full end-to-end test**

```bash
cd /Users/altanatabarut/Claude\ Code/microgpt && uvicorn server:app --port 7860 &
sleep 1 && open http://localhost:7860
```

Manual checklist:
- [ ] Load tab: paste a few names, click Load → preview appears
- [ ] Train tab: click Start → pipeline nodes animate
- [ ] Report tab: loss chart updates, stats fill in, samples appear at step 100
- [ ] Infer tab: click Generate → name cards appear

```bash
kill %1
```

**Step 3: Commit**

```bash
cd /Users/altanatabarut/Claude\ Code && git add microgpt/ docs/ && git commit -m "feat: microgpt web UI — load, train graph, report, infer tabs"
```

---

## Done

Run with:

```bash
cd microgpt && pip install -r requirements.txt && uvicorn server:app --port 7860 --reload
```

Open http://localhost:7860.
