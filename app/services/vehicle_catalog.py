"""Allowlisted vehicle artwork variants and car -> variant resolution.

Pure logic over an in-repo JSON catalog. Holds no VINs, plates, user ids or photos.
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

_CATALOG_PATH = Path(__file__).with_name("vehicle_catalog.json")


@dataclass(frozen=True)
class Resolution:
    kind: Literal["selected", "choose", "unknown"]
    variant_key: str | None
    options: list[tuple[str, str]]


def normalize(value: str) -> str:
    if not isinstance(value, str):
        return ""
    folded = unicodedata.normalize("NFKD", value.lower())
    ascii_only = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", ascii_only).strip()


@lru_cache(maxsize=1)
def _load() -> dict:
    return json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))


def _variants() -> dict[str, dict]:
    return {v["key"]: v for v in _load()["variants"]}


def _matches_name(v: dict, make: str, model: str) -> bool:
    return normalize(make) == v["make"] and normalize(model) in v["models"]


def candidates(make: str, model: str, year: int) -> list[str]:
    if not isinstance(year, int) or isinstance(year, bool):
        return []
    return sorted(
        k
        for k, v in _variants().items()
        if _matches_name(v, make, model) and v["years"][0] <= year <= v["years"][1]
    )


def resolve(make: str, model: str, year: int, choice: str | None = None) -> Resolution:
    found = candidates(make, model, year)
    if not found:
        return Resolution("unknown", None, [])
    if choice in found:
        return Resolution("selected", choice, [])
    variants = _variants()
    if len(found) == 1 and not variants[found[0]]["ambiguous"]:
        return Resolution("selected", found[0], [])
    return Resolution("choose", None, [(k, variants[k]["label"]) for k in found])


def is_allowed(variant_key: str) -> bool:
    return isinstance(variant_key, str) and variant_key in _variants()


def variant_matches_car(variant_key: str, make: str, model: str) -> bool:
    v = _variants().get(variant_key) if isinstance(variant_key, str) else None
    return v is not None and _matches_name(v, make, model)


def label(variant_key: str) -> str:
    return _variants()[variant_key]["label"]
