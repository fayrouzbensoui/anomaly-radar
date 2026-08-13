"""Shared helpers: config loading, logging, robust HTTP."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests
import yaml

log = logging.getLogger("scanner")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # yfinance is chatty about individual ticker failures; keep it quiet.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG) as fh:
        return yaml.safe_load(fh)


def http_get(url: str, retries: int = 3, timeout: int = 30) -> str:
    """GET with retry/backoff and a browser UA (some hosts reject default UAs)."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers={"User-Agent": _UA}, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all w/ retry
            last_exc = exc
            wait = 2**attempt
            log.warning("GET %s failed (%s); retry in %ss", url, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"GET {url} failed after {retries} attempts") from last_exc
