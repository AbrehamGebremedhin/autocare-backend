"""Delete stored media originals older than N days and strip their refs from the messages.

Usage: uv run python scripts/prune_media.py [--days 90] [--dry-run]
Only files shaped MEDIA_DIR/{user_id}/{session_id}/{message_id}/{n}{ext} are touched; symlinks are never followed.
Safe to run again: a second run finds nothing left to do. Content is never printed.
"""
import argparse
import asyncio
import os
import re
import stat
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.db import pool  # noqa: E402

_ID = r"[A-Za-z0-9_-]+"
_SHAPE = re.compile(rf"^{_ID}/{_ID}/{_ID}/\d+\.(?:jpg|mp4|m4a)$")

# One statement per session: rebuilds each message's media array without the pruned paths, in order, keeping every
# other field and message; a session with nothing to strip is not written. Returns the number of messages touched.
_STRIP_SQL = """
WITH s AS (SELECT id, messages FROM "Chatsession" WHERE id = $1 FOR UPDATE),
touched AS (
    SELECT count(*) AS n FROM s, jsonb_array_elements(s.messages) e
    WHERE jsonb_typeof(e -> 'media') = 'array'
      AND EXISTS (SELECT 1 FROM jsonb_array_elements(e -> 'media') m WHERE coalesce(m ->> 'path', '') = ANY($2::text[]))
),
upd AS (
    UPDATE "Chatsession" c
    SET messages = (
        SELECT coalesce(jsonb_agg(
            CASE WHEN jsonb_typeof(e.elem -> 'media') = 'array'
                 THEN jsonb_set(e.elem, '{media}', coalesce((
                     SELECT jsonb_agg(m ORDER BY mo) FROM jsonb_array_elements(e.elem -> 'media') WITH ORDINALITY AS x(m, mo)
                     WHERE coalesce(m ->> 'path', '') <> ALL($2::text[])), '[]'::jsonb))
                 ELSE e.elem END
            ORDER BY e.ord), '[]'::jsonb)
        FROM jsonb_array_elements(s.messages) WITH ORDINALITY AS e(elem, ord))
    FROM s
    WHERE c.id = s.id AND (SELECT n FROM touched) > 0
    RETURNING 1
)
SELECT n FROM touched
"""


def find_expired(root: Path, days: int, now: float | None = None) -> list[Path]:
    """Regular media files under `root` last modified more than `days` days ago and laid out as the store writes them."""
    root = Path(root)
    cutoff = (time.time() if now is None else now) - days * 86400
    found = []
    for dirpath, _, files in os.walk(root):  # os.walk does not follow directory symlinks
        for name in files:
            path = Path(dirpath, name)
            if not _SHAPE.fullmatch(path.relative_to(root).as_posix()):
                continue
            try:
                st = os.lstat(path)
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode) and st.st_mtime < cutoff:
                found.append(path)
    return sorted(found)


async def strip_refs(session_id: str, paths: list[str]) -> int:
    """Remove the refs with these relative paths from the session's messages; returns messages touched."""
    return int(await pool.fetchval(_STRIP_SQL, session_id, paths) or 0)


def _remove_empty_dirs(root: Path, files: list[Path]) -> None:
    for folder in sorted({p.parent for p in files}, key=lambda d: len(d.parts), reverse=True):
        for d in (folder, *folder.parents):
            if d == root:
                break
            try:
                d.rmdir()  # only succeeds on an empty directory
            except OSError:
                break


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


async def prune(root: Path, days: int, dry_run: bool) -> tuple[int, int, int]:
    """Returns (files, bytes, messages touched). Refs are stripped before the files go, so a crash in between
    leaves only files that the next run finds again."""
    root = Path(root)
    files = find_expired(root, days)
    sizes = {}
    for p in files:
        try:
            sizes[p] = os.lstat(p).st_size
        except OSError:
            pass
    files = [p for p in files if p in sizes]
    if dry_run:
        return len(files), sum(sizes.values()), 0

    by_session: dict[str, list[Path]] = {}
    for p in files:
        by_session.setdefault(p.relative_to(root).parts[1], []).append(p)
    touched = deleted = freed = 0
    for session_id, paths in by_session.items():
        if _is_uuid(session_id):  # any other folder name cannot belong to a row
            touched += await strip_refs(session_id, [p.relative_to(root).as_posix() for p in paths])
        for p in paths:
            try:
                p.unlink()
            except FileNotFoundError:
                continue
            deleted += 1
            freed += sizes[p]
    _remove_empty_dirs(root, files)
    return deleted, freed, touched


async def main(days: int, dry_run: bool) -> None:
    root = Path(get_settings().MEDIA_DIR)
    if dry_run:
        files, size, _ = await prune(root, days, True)
        print(f"dry run: would remove {files} files, {size} bytes older than {days} days")
        return
    await pool.init_pool()
    try:
        files, size, touched = await prune(root, days, False)
    finally:
        await pool.close_pool()
    print(f"pruned {files} files, {size} bytes, {touched} messages updated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.days < 0:
        parser.error("--days must not be negative")
    asyncio.run(main(args.days, args.dry_run))
