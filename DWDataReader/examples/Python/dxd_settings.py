"""Persistent processing preferences; no threshold is assumed on first use."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile


@dataclass(frozen=True)
class ProcessingSettings:
    ct_min_peak_a: float
    minimum_event_ms: float = 3.0

    def __post_init__(self):
        if isinstance(self.ct_min_peak_a, bool) or not isinstance(self.ct_min_peak_a, (int, float)):
            raise ValueError("CT threshold must be a number in amps")
        if not math.isfinite(self.ct_min_peak_a) or self.ct_min_peak_a < 0:
            raise ValueError("CT threshold must be finite and non-negative")
        if isinstance(self.minimum_event_ms, bool) or not isinstance(self.minimum_event_ms, (int, float)):
            raise ValueError("Minimum event duration must be a number in milliseconds")
        if not math.isfinite(self.minimum_event_ms) or self.minimum_event_ms < 0:
            raise ValueError("Minimum event duration must be finite and non-negative")


def default_settings_path() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "DewesoftReader" / "settings.json"


def load_settings(path: Path) -> ProcessingSettings | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ProcessingSettings(ct_min_peak_a=data["ct_min_peak_a"],
                                  minimum_event_ms=data.get("minimum_event_ms", 3.0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid settings file {path}: {exc}") from exc


def save_settings(settings: ProcessingSettings, path: Path) -> None:
    """Replace atomically so an interrupted save cannot leave partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            json.dump(asdict(settings), f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
