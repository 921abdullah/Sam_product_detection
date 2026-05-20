import os
import uuid
from pathlib import Path

from starlette.requests import Request


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


def public_base_url(request: Request) -> str:
    """Base URL for absolute links (ngrok, reverse proxy, or PUBLIC_BASE_URL)."""
    configured = os.environ.get("PUBLIC_BASE_URL", "").strip()
    if configured:
        return configured.rstrip("/")

    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    if host:
        return f"{proto}://{host}".rstrip("/")
    return str(request.base_url).rstrip("/")


def absolute_output_url(
    output_path: Path,
    outputs_root: Path,
    base_url: str,
) -> str:
    path = relative_output_url(output_path, outputs_root)
    if path.startswith(("http://", "https://")):
        return path
    return f"{base_url.rstrip('/')}{path}"
