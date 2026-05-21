# Drawer Change Detection API

Pipeline-based backend for aligning drawer images (LightGlue), segmenting items (SAM3), matching detections (bbox or mask IoU), and rendering change overlays.

## Project structure

```text
sam_matching/
├── app/
│   ├── main.py
│   ├── api/routes/drawer_change.py
│   ├── core/config.py
│   ├── core/model_registry.py
│   ├── schemas/drawer_change.py
│   ├── pipelines/drawer_change_pipeline.py
│   ├── services/
│   └── utils/
├── requirements.txt
└── run_local.py
```

## Virtual environment

Create and use a Python virtual environment at the repo root. All commands below assume the venv is **activated** so `python`, `pip`, and `uvicorn` refer to that environment.

**Create venv** (once):

```bash
python -m venv venv
```

**Activate:**

| Platform | Command |
|----------|---------|
| Windows (PowerShell) | `.\venv\Scripts\Activate.ps1` |
| Windows (cmd) | `venv\Scripts\activate.bat` |
| Linux / macOS | `source venv/bin/activate` |

**Install dependencies** (with venv active):

```bash
pip install -r requirements.txt
```

LightGlue is listed in `requirements.txt` as a Git dependency; install PyTorch separately for your platform (CPU or CUDA) from [pytorch.org](https://pytorch.org/get-started/locally/) if needed.

Place `sam3.pt` where Ultralytics expects it, or in the directory you run commands from.

The `venv/` directory is gitignored — each machine creates its own environment.

## Run API

With the venv activated, start the server from the repo root. Use **one worker** so ML models load once per process:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Health check: `GET http://localhost:8000/health`

## Analyze endpoint

`POST /product_change`

| Field | Type | Default |
|-------|------|---------|
| `before_image` | file | required |
| `after_image` | file | required |
| `mode` | `bbox` \| `mask` | `mask` |
| `prompts` | comma-separated string | item,gold bolt,... |
| `bbox_iou_threshold` | float | 0.15 |
| `mask_iou_threshold` | float | 0.35 |
| `sam_conf` | float | 0.50 |

Example with curl:

```bash
curl -X POST "http://localhost:8000/product_change" \
  -F "before_image=@before.jpg" \
  -F "after_image=@after.jpg" \
  -F "mode=mask"
```

The JSON field `output_image` is a **full URL** (e.g. `http://localhost:8000/outputs/overlay_abc.jpg`), built from the request host or `PUBLIC_BASE_URL`.

Files are stored under `app/outputs/` and served at `/outputs/<filename>`.

For ngrok, set your tunnel URL before starting the API (recommended if auto-detection uses `http`):

```bash
export PUBLIC_BASE_URL=https://your-subdomain.ngrok-free.app   # Linux/macOS
$env:PUBLIC_BASE_URL="https://your-subdomain.ngrok-free.app"  # PowerShell
```

## Local CLI (no HTTP)

With the venv activated:

```bash
python run_local.py \
  --before path/to/before.jpg \
  --after path/to/after.jpg \
  --mode mask
```

Windows PowerShell (line continuation):

```powershell
python run_local.py `
  --before "path\to\before.jpg" `
  --after "path\to\after.jpg" `
  --mode mask
```

## Pipeline flow

```text
API → DrawerChangePipeline
        → AlignmentService (LightGlue + homography)
        → SegmentationService (SAM3, thread-safe lock)
        → BBoxMatcher (bbox mode) OR MaskChangeMatcher (mask mode):
            dedupe masks → IoU/containment match → DINOv2 rescue → local RANSAC verify
        → RenderingService
        → JSON + output image URL
```

Mask mode loads `facebook/dinov2-base` via Hugging Face `transformers` (cached on first run).

## Deployment note

Each Uvicorn worker loads its own copy of the models. For GPU/RAM limits, start with `--workers 1` and scale only after measuring memory use.
