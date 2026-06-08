# MicroGPT Web UI Design
_2026-06-08_

## Architecture

Single `server.py` (FastAPI) + single `index.html`. No build step.

**Backend endpoints:**
- `POST /load` — validate URL or text, return doc count + sample
- `POST /train` — start background thread
- `GET /stream` — SSE: loss per step + periodic inference samples
- `POST /infer` — run inference, return names
- `GET /state` — current training state

**Frontend:** vanilla JS, Chart.js for loss curve, no framework.

## Tabs

### Tab 1: Load
URL field (default: names.txt) + paste area + doc preview (count, sample lines).

### Tab 2: Train (Workflow Graph)
SVG computation graph: `Dataset → Tokenizer → Autograd → GPT Forward → Loss → Backward → Adam`
- Node states: idle (grey) / active (blue pulse) / done (green)
- GPT Forward expands: `Embed → RMSNorm → Attention → MLP → LM Head`
- Start/Stop button

### Tab 3: Report
- Live loss chart (Chart.js via SSE)
- Stats sidebar: step/total, current loss, best loss, params, vocab size, ETA
- Every 100 steps: 5 inference samples appended to scrolling samples log

### Tab 4: Infer
Temperature slider (0.1–1.0), sample count (1–20), Generate button, card grid output.
