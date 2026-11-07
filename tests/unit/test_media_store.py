"""MediaStore against a real temp directory."""

import errno
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.config import get_settings
from app.services.media_store import MediaStore


@pytest.fixture
def root(tmp_path):
    return tmp_path / "media"


@pytest.fixture
def src(tmp_path):
    f = tmp_path / "upload.bin"
    f.write_bytes(b"original bytes")
    return f


def _tree(path: Path) -> list[str]:
    return sorted(p.relative_to(path).as_posix() for p in path.rglob("*")) if path.exists() else []


async def test_upload_copies_to_the_scheme_and_returns_a_relative_ref(root, src):
    ref = await MediaStore(root).upload("u1", "s-1", "m1", 0, "video", ".mp4", src)
    assert ref == {"kind": "video", "path": "u1/s-1/m1/0.mp4"}
    assert (root / "u1/s-1/m1/0.mp4").read_bytes() == b"original bytes"
    assert src.exists()  # copied, not moved
    assert _tree(root / "u1/s-1/m1") == ["0.mp4"]  # no temp file left behind


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
async def test_upload_uses_private_modes(root, src):
    await MediaStore(root).upload("u1", "s1", "m1", 0, "image", ".jpg", src)
    assert stat.S_IMODE((root / "u1/s1/m1/0.jpg").stat().st_mode) == 0o600
    assert stat.S_IMODE((root / "u1/s1/m1").stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "u1").stat().st_mode) == 0o700


async def test_default_root_comes_from_settings(tmp_path, src, monkeypatch):
    monkeypatch.setattr(get_settings(), "MEDIA_DIR", str(tmp_path / "cfg"))
    await MediaStore().upload("u1", "s1", "m1", 0, "audio", ".m4a", src)
    assert (tmp_path / "cfg/u1/s1/m1/0.m4a").exists()


@pytest.mark.parametrize("bad", ["../x", "..", "a/b", "a" + chr(92) + "b", "/etc", "C:" + chr(92) + "x", "", "a b", "x" + chr(0) + "y"])
@pytest.mark.parametrize("slot", [0, 1, 2])
async def test_bad_ids_raise_and_write_nothing(root, src, tmp_path, bad, slot):
    ids = ["u1", "s1", "m1"]
    ids[slot] = bad
    with pytest.raises(ValueError):
        await MediaStore(root).upload(*ids, 0, "image", ".jpg", src)
    assert _tree(root) == [] and not (tmp_path / "x").exists()


@pytest.mark.parametrize("ext", [".exe", "jpg", ".JPG", "", ".jpg/../x", ".bin"])
async def test_bad_extension_raises(root, src, ext):
    with pytest.raises(ValueError):
        await MediaStore(root).upload("u1", "s1", "m1", 0, "image", ext, src)
    assert _tree(root) == []


def _symlink(link: Path, target: Path):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not permitted here")


async def test_planted_symlink_is_not_followed_on_write(root, src, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir()
    _symlink(root / "u1", outside)
    with pytest.raises(ValueError):
        await MediaStore(root).upload("u1", "s1", "m1", 0, "image", ".jpg", src)
    assert _tree(outside) == []


async def test_planted_symlink_is_not_followed_on_delete(root, tmp_path):
    outside = tmp_path / "outside"
    (outside / "m1").mkdir(parents=True)
    (outside / "m1/keep.jpg").write_bytes(b"precious")
    (root / "u1").mkdir(parents=True)
    _symlink(root / "u1/s1", outside)
    assert await MediaStore(root).delete_prefix("u1", "s1") == 0
    assert (outside / "m1/keep.jpg").read_bytes() == b"precious"
    # the parent being a link is refused too
    _symlink(root / "u2", outside)
    assert await MediaStore(root).delete_prefix("u2", "m1") == 0
    assert (outside / "m1/keep.jpg").exists()


async def test_symlink_inside_the_session_folder_is_removed_not_followed(root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.jpg").write_bytes(b"precious")
    (root / "u1/s1").mkdir(parents=True)
    _symlink(root / "u1/s1/link", outside)
    await MediaStore(root).delete_prefix("u1", "s1")
    assert not (root / "u1/s1").exists() and (outside / "keep.jpg").exists()


async def test_delete_prefix_removes_the_session_only_and_is_idempotent(root, src):
    store = MediaStore(root)
    await store.upload("u1", "s1", "m1", 0, "image", ".jpg", src)
    await store.upload("u1", "s1", "m2", 0, "video", ".mp4", src)
    await store.upload("u1", "s2", "m1", 0, "image", ".jpg", src)
    assert await store.delete_prefix("u1", "s1") == 2
    assert not (root / "u1/s1").exists() and (root / "u1/s2/m1/0.jpg").exists()
    assert await store.delete_prefix("u1", "s1") == 0


async def test_delete_prefix_never_raises(root):
    store = MediaStore(root)
    assert await store.delete_prefix("../..", "x") == 0
    assert await store.delete_prefix("u1", "") == 0


async def test_write_failure_is_raised_to_the_caller_and_leaves_no_temp(root, src, monkeypatch):
    def boom(*a, **k):
        raise PermissionError("read-only")

    monkeypatch.setattr("app.services.media_store.os.replace", boom)
    with pytest.raises(PermissionError):
        await MediaStore(root).upload("u1", "s1", "m1", 0, "image", ".jpg", src)
    assert _tree(root / "u1/s1/m1") == []


async def test_low_disk_space_refuses_the_write(root, src, monkeypatch):
    monkeypatch.setattr("app.services.media_store.shutil.disk_usage", lambda p: SimpleNamespace(free=1024))
    with pytest.raises(OSError) as e:
        await MediaStore(root).upload("u1", "s1", "m1", 0, "image", ".jpg", src)
    assert e.value.errno == errno.ENOSPC
    assert not (root / "u1").exists()


async def test_enough_free_space_writes_normally(root, src, monkeypatch):
    monkeypatch.setattr("app.services.media_store.shutil.disk_usage", lambda p: SimpleNamespace(free=10**12))
    await MediaStore(root).upload("u1", "s1", "m1", 0, "image", ".jpg", src)
    assert (root / "u1/s1/m1/0.jpg").exists()


def _fill(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


@pytest.fixture
def quota(monkeypatch):
    def set_quota(mb: int):
        monkeypatch.setattr(get_settings(), "MEDIA_USER_QUOTA_MB", mb)

    return set_quota


async def test_upload_under_quota_writes(root, src, quota):
    quota(1)
    _fill(root / "u1/s0/m0/0.jpg", 1024 * 1024 - 100)
    ref = await MediaStore(root).upload("u1", "s1", "m1", 0, "image", ".jpg", src)  # 14 bytes still fit
    assert (root / ref["path"]).read_bytes() == b"original bytes"


async def test_upload_over_quota_is_refused_and_writes_nothing(root, src, quota):
    quota(1)
    _fill(root / "u1/s0/m0/0.jpg", 1024 * 1024 - 10)
    before = _tree(root)
    with pytest.raises(OSError) as exc:
        await MediaStore(root).upload("u1", "s1", "m1", 0, "image", ".jpg", src)  # 14 bytes do not
    assert exc.value.errno == errno.EDQUOT
    assert _tree(root) == before


async def test_quota_zero_disables_the_cap(root, src, quota):
    quota(0)
    _fill(root / "u1/s0/m0/0.jpg", 3 * 1024 * 1024)
    await MediaStore(root).upload("u1", "s1", "m1", 0, "image", ".jpg", src)
    assert (root / "u1/s1/m1/0.jpg").exists()


async def test_other_users_files_do_not_count(root, src, quota):
    quota(1)
    _fill(root / "u2/s0/m0/0.jpg", 5 * 1024 * 1024)
    await MediaStore(root).upload("u1", "s1", "m1", 0, "image", ".jpg", src)
    assert (root / "u1/s1/m1/0.jpg").exists()


async def test_symlinks_are_not_followed_when_counting(root, src, quota, tmp_path):
    quota(1)
    big = tmp_path / "elsewhere"
    _fill(big / "huge.bin", 5 * 1024 * 1024)
    (root / "u1").mkdir(parents=True)
    try:
        (root / "u1/linkdir").symlink_to(big, target_is_directory=True)
        (root / "u1/linkfile").symlink_to(big / "huge.bin")
    except OSError:
        pytest.skip("symlinks not permitted")
    await MediaStore(root).upload("u1", "s1", "m1", 0, "image", ".jpg", src)
    assert (root / "u1/s1/m1/0.jpg").exists()
