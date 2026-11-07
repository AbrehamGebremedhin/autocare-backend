"""scripts/prune_media.py: file discovery, deletion and the SQL it sends (fake pool, no database)."""

import importlib.util
import os
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("prune_media", ROOT / "scripts" / "prune_media.py")
prune_media = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prune_media)

S1 = "11111111-1111-1111-1111-111111111111"
S2 = "22222222-2222-2222-2222-222222222222"
DAY = 86400


def _file(root: Path, rel: str, age_days: float, size: int = 10) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    t = time.time() - age_days * DAY
    os.utime(p, (t, t))
    return p


@pytest.fixture
def media(tmp_path):
    return tmp_path / "media"


def test_find_expired_picks_only_old_files(media):
    old = _file(media, f"u1/{S1}/m1/0.jpg", 100)
    new = _file(media, f"u1/{S1}/m2/0.jpg", 10)
    assert prune_media.find_expired(media, 90) == [old]
    assert prune_media.find_expired(media, 5) == sorted([old, new])


def test_find_expired_ignores_odd_paths(media):
    _file(media, "stray.jpg", 200)
    _file(media, f"u1/{S1}/0.jpg", 200)  # too shallow
    _file(media, f"u1/{S1}/m1/0.txt", 200)  # unknown extension
    _file(media, f"u1/{S1}/m1/a.jpg", 200)  # not {n}
    _file(media, f"u1/{S1}/m1/.0.jpg.abc.tmp", 200)
    _file(media, f"u1/{S1}/m1/deeper/0.jpg", 200)
    _file(media, f"u 1/{S1}/m1/0.jpg", 200)
    assert prune_media.find_expired(media, 90) == []


def test_find_expired_never_follows_symlinks(media, tmp_path):
    outside = tmp_path / "outside"
    _file(outside, "m1/0.jpg", 200)
    (media / f"u1/{S1}/m1").mkdir(parents=True)
    target = _file(tmp_path, "real.jpg", 200)
    try:
        (media / f"u1/{S1}/m9").symlink_to(outside / "m1", target_is_directory=True)
        (media / f"u1/{S1}/m1/0.jpg").symlink_to(target)
    except OSError:
        pytest.skip("symlinks not permitted")
    assert prune_media.find_expired(media, 90) == []


async def test_dry_run_deletes_nothing_and_touches_no_database(media, fake_pool):
    f = _file(media, f"u1/{S1}/m1/0.jpg", 100, size=7)
    assert await prune_media.prune(media, 90, dry_run=True) == (1, 7, 0)
    assert f.exists() and fake_pool.calls == []


async def test_prune_strips_then_deletes_and_removes_empty_folders(media, fake_pool):
    a = _file(media, f"u1/{S1}/m1/0.jpg", 100, size=5)
    b = _file(media, f"u1/{S1}/m1/1.mp4", 100, size=6)
    c = _file(media, f"u2/{S2}/m3/0.m4a", 100, size=7)
    fresh = _file(media, f"u1/{S1}/m2/0.jpg", 1)
    fake_pool.value = 1
    assert await prune_media.prune(media, 90, dry_run=False) == (3, 18, 2)
    assert not (a.exists() or b.exists() or c.exists()) and fresh.exists()
    assert not (media / "u2").exists() and not (media / f"u1/{S1}/m1").exists()  # emptied folders go
    assert (media / f"u1/{S1}/m2").exists() and media.exists()
    assert sorted((args[0], args[1]) for _, _, args in fake_pool.calls) == sorted(
        [(S1, [f"u1/{S1}/m1/0.jpg", f"u1/{S1}/m1/1.mp4"]), (S2, [f"u2/{S2}/m3/0.m4a"])]
    )
    assert await prune_media.prune(media, 90, dry_run=False) == (0, 0, 0)  # idempotent


async def test_non_uuid_session_folder_is_deleted_without_a_query(media, fake_pool):
    f = _file(media, "u1/s-1/m1/0.jpg", 100)
    assert await prune_media.prune(media, 90, dry_run=False) == (1, 10, 0)
    assert not f.exists() and fake_pool.calls == []


async def test_strip_refs_is_one_parameterised_statement(fake_pool):
    fake_pool.value = 3
    assert await prune_media.strip_refs(S1, ["a/b/c/0.jpg"]) == 3
    [(method, sql, args)] = fake_pool.calls
    assert method == "fetchval" and args == (S1, ["a/b/c/0.jpg"])
    assert sql.count('UPDATE "Chatsession"') == 1
    assert "$1" in sql and "$2" in sql and "a/b/c" not in sql
    assert "WITH ORDINALITY" in sql and "jsonb_set" in sql  # order kept, other fields untouched


async def test_strip_refs_none_means_zero(fake_pool):
    fake_pool.value = None
    assert await prune_media.strip_refs(S1, ["p"]) == 0
