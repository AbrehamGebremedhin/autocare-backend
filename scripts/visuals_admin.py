"""Operator CLI for vehicle artwork: review what the worker parked, approve it, or seed curated art.

Usage: uv run python scripts/visuals_admin.py {list-review | approve VARIANT --revision N | reject VARIANT |
       requeue VARIANT | seed VARIANT --image PATH [--approve]}
Exit codes: 0 done, 1 usage / not found / nothing changed, 2 refused. One short line per result; shas, signed URLs
and filesystem paths are never printed. Approve writes the object first and activates the revision second, so a
crash in between leaves only an unreferenced object that the sweep removes after its grace period.
"""
import argparse
import asyncio
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.CRUD import vehicle_visual_crud as crud  # noqa: E402
from app.db import pool  # noqa: E402
from app.services.vehicle_catalog import is_allowed  # noqa: E402
from app.services.visual_image import MAX_SOURCE_BYTES, InvalidArtwork, normalize, validate_published  # noqa: E402
from app.services.visual_store import VisualStore  # noqa: E402

WIDTH, HEIGHT = 1024, 420


def _store() -> VisualStore:
    return VisualStore()


def _delete_draft(store: VisualStore, variant: str, token: int) -> None:
    """Best effort: a leftover draft is reclaimed by the sweep."""
    try:
        store.delete_draft(store.draft_rel(variant, crud.STYLE_VERSION, token))
    except (ValueError, OSError):
        pass


async def _list_review() -> int:
    for row in await crud.list_review():
        print(f"{row['variant_key']} revision={row['published_revision']} state={row['state']} attempts={row['attempt_count']}")
    return 0


async def _approve(variant: str, revision: int) -> int:
    store = _store()
    row = await crud.get_visual(variant)
    if row is None:
        print("not found")
        return 1
    if row["state"] != "review":
        print("refused: not in review")
        return 2
    if row["published_revision"] != revision:
        print("refused: stale revision")
        return 2
    token = row["fencing_token"]
    try:
        data = store.read_draft(store.draft_rel(variant, crud.STYLE_VERSION, token))
    except (OSError, ValueError):
        print("refused: draft missing")
        return 2
    try:
        sha, _, _ = validate_published(data)
    except InvalidArtwork as exc:
        print(f"refused: {exc.code}")
        return 2
    if sha != row["draft_sha256"]:
        print("refused: draft mismatch")
        return 2
    store.put_object(sha, data)  # object first, activate second
    if not await crud.publish(row["id"], revision, sha, WIDTH, HEIGHT):
        print("refused: stale revision")
        return 2
    _delete_draft(store, variant, token)
    print(f"approved {variant} revision={revision + 1}")
    return 0


async def _reject(variant: str) -> int:
    row = await crud.get_visual(variant)
    if row is not None and await crud.reject(variant):
        _delete_draft(_store(), variant, row["fencing_token"])
        print(f"rejected {variant}")
        return 0
    print("nothing to reject")
    return 1


async def _requeue(variant: str) -> int:
    if await crud.requeue(variant):
        print(f"requeued {variant}")
        return 0
    print("nothing to requeue")
    return 1


def _read_source(path: str) -> bytes | None:
    try:
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_SOURCE_BYTES:  # lstat: a symlink is not a regular file
            return None
        with open(path, "rb") as f:
            return f.read(MAX_SOURCE_BYTES + 1)
    except OSError:
        return None


async def _seed(variant: str, image: str, approve: bool) -> int:
    if not is_allowed(variant):
        print("unknown variant")
        return 1
    source = _read_source(image)
    if source is None or len(source) > MAX_SOURCE_BYTES:
        print("unreadable image: need a regular file of at most 20 MB")
        return 1
    try:
        webp, sha, width, height = normalize(source)
    except InvalidArtwork as exc:
        print(f"refused: {exc.code}")
        return 2
    row, _ = await crud.ensure_job(variant, "curated")
    if row is None:
        print("not found")
        return 1
    store = _store()
    token = row["fencing_token"]  # a row in review keeps its draft at its current token
    store.write_draft(variant, crud.STYLE_VERSION, token, webp)
    try:
        row = await crud.seed_review(variant, sha, width, height)
    except ValueError:
        _delete_draft(store, variant, token)
        print("refused: job running")
        return 2
    print(f"seeded {variant} revision={row['published_revision']}")
    if not approve:
        return 0
    row = await crud.get_visual(variant)
    return await _approve(variant, row["published_revision"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visuals_admin")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-review")
    p = sub.add_parser("approve")
    p.add_argument("variant")
    p.add_argument("--revision", type=int, required=True)
    for name in ("reject", "requeue"):
        sub.add_parser(name).add_argument("variant")
    p = sub.add_parser("seed")
    p.add_argument("variant")
    p.add_argument("--image", required=True)
    p.add_argument("--approve", action="store_true")
    return parser


async def _run(args: argparse.Namespace) -> int:
    await pool.init_pool()
    try:
        if args.command == "list-review":
            return await _list_review()
        if args.command == "approve":
            return await _approve(args.variant, args.revision)
        if args.command == "reject":
            return await _reject(args.variant)
        if args.command == "requeue":
            return await _requeue(args.variant)
        return await _seed(args.variant, args.image, args.approve)
    finally:
        await pool.close_pool()


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:  # argparse exits 2 for usage errors; here 2 means refused
        return 0 if exc.code == 0 else 1
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
