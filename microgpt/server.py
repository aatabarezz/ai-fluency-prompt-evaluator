import math, random, os, threading, time, json, urllib.request, pathlib
from dataclasses import dataclass, field
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

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
    docs: list = field(default_factory=list)
    uchars: list = field(default_factory=list)
    BOS: int = 0
    vocab_size: int = 0
    n_embd: int = 16
    n_head: int = 4
    n_layer: int = 1
    block_size: int = 16
    num_steps: int = 1000
    step: int = 0
    loss: float = 0.0
    best_loss: float = float('inf')
    losses: list = field(default_factory=list)
    samples_log: list = field(default_factory=list)
    stage: str = "idle"
    running: bool = False
    state_dict: dict = field(default_factory=dict)
    params: list = field(default_factory=list)
    m: list = field(default_factory=list)
    v_buf: list = field(default_factory=list)
    num_params: int = 0
    start_time: float = 0.0
    events: list = field(default_factory=list)


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


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI()

@app.get("/")
def root():
    return HTMLResponse((pathlib.Path(__file__).parent / "index.html").read_text())

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/load")
def load(body: dict):
    global _state
    with _lock:
        if body.get("url"):
            tmp = "/tmp/microgpt_input.txt"
            urllib.request.urlretrieve(body["url"], tmp)
            raw = open(tmp).read()
        else:
            raw = body.get("text", "")
        docs = [l.strip() for l in raw.strip().split('\n') if l.strip()]
        random.shuffle(docs)
        uchars = sorted(set(''.join(docs)))
        _state.docs = docs
        _state.uchars = uchars
        _state.BOS = len(uchars)
        _state.vocab_size = len(uchars) + 1
        return {"num_docs": len(docs), "vocab_size": _state.vocab_size, "sample": docs[:5]}

@app.post("/train")
def start_train(body: dict):
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
    threading.Thread(target=_train_thread, args=(_state,), daemon=True).start()
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
        results = _infer(_state, n_samples=int(body.get("n", 10)), temperature=float(body.get("temperature", 0.5)))
    return {"samples": results}
