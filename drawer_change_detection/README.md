# Drawer Change Detection API

Pipeline-based backend for aligning drawer images (LightGlue), segmenting items (SAM3), matching detections (bbox or mask IoU), and rendering change overlays.

## Project structure

```text
drawer_change_detection/
├── app/
│   ├── main.py
│   ├── api/routes/drawer_change.py
│   ├── core/config.py
│   ├── core/model_registry.py
│   ├── schemas/drawer_change.py
│   ├── pipelines/drawer_change_pipeline.py
│   ├── services/
│   │   ├── alignment_service.py
│   │   ├── segmentation_service.py
│   │   ├── matching_service.py
│   │   └── rendering_service.py
│   └── utils/
├── requirements.txt
└── run_local.py
```

## Setup

From the repo root (use your existing `.matching` venv if present):

```powershell
cd drawer_change_detection
..\.matching\Scripts\pip.exe install fastapi uvicorn python-multipart
```

Place `sam3.pt` where Ultralytics expects it (or in the working directory).

## Run API

Use a single worker so models are loaded once:

```powershell
cd drawer_change_detection
..\.matching\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Health check: `GET http://localhost:8000/health`

## Analyze endpoint

`POST /api/v1/drawer-change/analyze`

| Field | Type | Default |
|-------|------|---------|
| `before_image` | file | required |
| `after_image` | file | required |
| `mode` | `bbox` \| `mask` | `mask` |
| `prompts` | comma-separated string | item,gold bolt,... |
| `bbox_iou_threshold` | float | 0.15 |
| `mask_iou_threshold` | float | 0.50 |
| `sam_conf` | float | 0.50 |

Example with curl:

```bash
curl -X POST "http://localhost:8000/api/v1/drawer-change/analyze" \
  -F "before_image=@before.jpg" \
  -F "after_image=@after.jpg" \
  -F "mode=mask"
```

Output images are written under `app/outputs/` and exposed at `/outputs/<filename>`.

## Local CLI (no HTTP)

```powershell
cd drawer_change_detection
..\.matching\Scripts\python.exe run_local.py ^
  --before "E:\path\to\before.jpg" ^
  --after "E:\path\to\after.jpg" ^
  --mode mask
```

## Pipeline flow

```text
API → DrawerChangePipeline
        → AlignmentService (LightGlue + homography)
        → SegmentationService (SAM3, thread-safe lock)
        → BBoxMatcher / MaskMatcher
        → RenderingService
        → JSON + output image path
```
