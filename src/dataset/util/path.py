from pathlib import Path


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        msg = f"{label} not found: {path}"
        raise FileNotFoundError(msg)
    return path
