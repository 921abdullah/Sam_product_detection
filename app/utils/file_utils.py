import os
import uuid
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def unique_output_path(output_dir: str | Path, prefix: str, suffix: str = ".jpg") -> Path:
    output_dir = ensure_dir(output_dir)
    return output_dir / f"{prefix}_{uuid.uuid4().hex[:12]}{suffix}"


def save_upload_to_temp(upload_bytes: bytes, temp_dir: str | Path, prefix: str) -> Path:
    temp_dir = ensure_dir(temp_dir)
    path = temp_dir / f"{prefix}_{uuid.uuid4().hex[:12]}.jpg"
    path.write_bytes(upload_bytes)
    return path


def relative_output_url(output_path: Path, outputs_root: Path) -> str:
    try:
        rel = output_path.resolve().relative_to(outputs_root.resolve())
        return f"/outputs/{rel.as_posix()}"
    except ValueError:
        return str(output_path)
