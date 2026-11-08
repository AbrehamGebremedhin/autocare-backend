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

# A car the catalog does not know still gets artwork: the owner picks a body style and that choice becomes
# its own variant, "auto-<make>-<model>-<year>-<body>". Keys are built from normalized text only (a-z, 0-9, -),
# so nothing the user typed reaches the provider prompt except plain words.
BODY_STYLES = ("sedan", "hatchback", "wagon", "suv", "coupe", "convertible", "pickup", "van")
_AUTO_TAIL = re.compile(r"(\d{4})-(" + "|".join(BODY_STYLES) + ")")


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


def _slug(value: str) -> str:
    return normalize(value).replace(" ", "-")


def usable(make: str, model: str, year: int) -> bool:
    """Enough to name a car: a four-digit year and a make and model with some letters or digits."""
    return (
        isinstance(year, int) and not isinstance(year, bool) and 1000 <= year <= 9999
        and bool(_slug(make)) and bool(_slug(model))
    )


def _auto_keys(make: str, model: str, year: int, styles) -> list[str]:
    if not usable(make, model, year):
        return []
    head = f"auto-{_slug(make)}-{_slug(model)}-{year}"
    return [f"{head}-{body}" for body in BODY_STYLES if body in styles]  # allowlist order, no duplicates


def candidates(make: str, model: str, year: int, styles=()) -> list[str]:
    """The catalog's variants for the car; failing that, one generated variant per body style in `styles`
    (the body styles the car really came in, see body_styles.py)."""
    if not isinstance(year, int) or isinstance(year, bool):
        return []
    known = sorted(
        k
        for k, v in _variants().items()
        if _matches_name(v, make, model) and v["years"][0] <= year <= v["years"][1]
    )
    return known or _auto_keys(make, model, year, styles)


def resolve(make: str, model: str, year: int, choice: str | None = None, styles=()) -> Resolution:
    found = candidates(make, model, year, styles)
    if not found:
        return Resolution("unknown", None, [])
    if choice in found:
        return Resolution("selected", choice, [])
    variants = _variants()
    if len(found) == 1 and (found[0] not in variants or not variants[found[0]]["ambiguous"]):
        return Resolution("selected", found[0], [])
    return Resolution("choose", None, [(k, label(k)) for k in found])


def _auto_parts(variant_key: str) -> tuple[str, str, str] | None:
    """(make-and-model slug, year, body) of an auto key, or None."""
    if not isinstance(variant_key, str) or not variant_key.startswith("auto-") or len(variant_key) > 200:
        return None
    for body in BODY_STYLES:
        m = re.fullmatch(r"auto-([a-z0-9]+(?:-[a-z0-9]+)*)-(\d{4})-" + body, variant_key)
        if m:
            return m.group(1), m.group(2), body
    return None


def is_allowed(variant_key: str) -> bool:
    return isinstance(variant_key, str) and (variant_key in _variants() or _auto_parts(variant_key) is not None)


def variant_matches_car(variant_key: str, make: str, model: str) -> bool:
    v = _variants().get(variant_key) if isinstance(variant_key, str) else None
    if v is not None:
        return _matches_name(v, make, model)
    if _auto_parts(variant_key) is None or not _slug(make) or not _slug(model):
        return False
    return variant_key.startswith(f"auto-{_slug(make)}-{_slug(model)}-") and bool(
        _AUTO_TAIL.fullmatch(variant_key[len(f"auto-{_slug(make)}-{_slug(model)}-"):])
    )


def label(variant_key: str) -> str:
    v = _variants().get(variant_key)
    if v is not None:
        return v["label"]
    parts = _auto_parts(variant_key)
    if parts is None:
        raise KeyError(variant_key)
    name, year, body = parts
    return f"{name.replace('-', ' ').title()} {year} {body.title()}"
