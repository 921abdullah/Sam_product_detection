"""
Legacy monolithic script — logic moved to drawer_change_detection/.

Run the pipeline locally:

    cd drawer_change_detection
    ..\.matching\Scripts\python.exe run_local.py ^
        --before path/to/before.jpg ^
        --after path/to/after.jpg ^
        --mode mask

Start the API:

    cd drawer_change_detection
    ..\.matching\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

raise SystemExit(
    "This script was refactored into drawer_change_detection/. "
    "See drawer_change_detection/README.md"
)
