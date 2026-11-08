"""VisualStore against a real temp directory."""

import hashlib
import os
import stat
import subprocess
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.services.visual_store import VisualStore

DATA = b"RIFF....WEBPfake"
SHA = hashlib.sha256(DATA).hexdigest()


@pytest.fixture
def root(tmp_path):
    return tmp_path / "visuals"


@pytest.fixture
def store(root):
    return VisualStore(root)


def _tree(path: Path) -> list[str]:
    return sorted(p.relative_to(path).as_posix() for p in path.rglob("*")) if path.exists() else []


def _link_dir(link: Path, target: Path) -> None:
    """Symlink, or a junction on Windows when symlinks need privilege; skip if neither works."""
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        pass
    if os.name == "nt":
        res = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
        if res.returncode == 0 and link.exists():
            return
    pytest.skip("neither symlinks nor junctions can be created here")


def _age(path: Path, seconds: float) -> None:
    t = path.stat().st_mtime - seconds
    os.utime(path, (t, t))


# --- objects -----------------------------------------------------------------


def test_put_object_round_trip(store, root):
    store.put_object(SHA, DATA)
    p = store.object_path(SHA)
    assert p is not None and p.read_bytes() == DATA
    assert p.resolve().is_relative_to(root.resolve())
    assert store.has_object(SHA)
    assert _tree(root / "objects") == [f"{SHA}.webp"]  # no temp file left


def test_put_object_is_idempotent(store):
    store.put_object(SHA, DATA)
    store.put_object(SHA, DATA)
    assert store.object_path(SHA).read_bytes() == DATA


@pytest.mark.parametrize("sha", ["0" * 64, SHA.upper(), SHA[:63], SHA + "0", "../" + SHA, "", SHA + "\n"])
def test_put_object_refuses_bad_or_mismatched_sha(store, root, sha):
    with pytest.raises(ValueError):
        store.put_object(sha, DATA)
    assert _tree(root) == [] or _tree(root / "objects") == []


@pytest.mark.parametrize(
    "name",
    ["..", "a/b", "a\\b", "%2e%2e", SHA.upper(), SHA[:63], SHA + "0", SHA + "\n", "", "../" + SHA, "x" * 64, SHA + ".webp"],
)
def test_object_path_rejects_bad_names(store, name):
    store.put_object(SHA, DATA)
    assert store.object_path(name) is None
    assert store.has_object(name) is False


def test_object_path_missing_is_none(store):
    assert store.object_path(SHA) is None


def test_put_object_failure_leaves_no_file(store, root, monkeypatch):
    def boom(*a, **k):
        raise PermissionError("read-only")

    monkeypatch.setattr("app.services.visual_store.os.replace", boom)
    with pytest.raises(PermissionError):
        store.put_object(SHA, DATA)
    assert _tree(root / "objects") == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_private_modes(store, root):
    store.put_object(SHA, DATA)
    rel = store.write_draft("v1", "s1", 1, DATA)
    for d in [root, root / "objects", root / "drafts", root / "drafts/v1", root / "drafts/v1/s1"]:
        assert stat.S_IMODE(d.stat().st_mode) == 0o700, d
    for f in [root / "objects" / f"{SHA}.webp", root / rel]:
        assert stat.S_IMODE(f.stat().st_mode) == 0o600, f


def test_default_root_comes_from_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "VISUALS_DIR", str(tmp_path / "cfg"))
    VisualStore().put_object(SHA, DATA)
    assert (tmp_path / "cfg/objects" / f"{SHA}.webp").exists()


def test_symlinked_objects_dir_is_refused(store, root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir()
    _link_dir(root / "objects", outside)
    with pytest.raises(ValueError):
        store.put_object(SHA, DATA)
    assert _tree(outside) == []
    (outside / f"{SHA}.webp").write_bytes(DATA)
    assert store.object_path(SHA) is None
    assert store.has_object(SHA) is False


def test_symlinked_object_file_is_not_served(store, root, tmp_path):
    (root / "objects").mkdir(parents=True)
    target = tmp_path / "secret.webp"
    target.write_bytes(DATA)
    try:
        (root / "objects" / f"{SHA}.webp").symlink_to(target)
    except OSError:
        pytest.skip("file symlinks not permitted here")
    assert store.object_path(SHA) is None
    with pytest.raises(ValueError):
        store.put_object(SHA, DATA)


# --- drafts ------------------------------------------------------------------


def test_draft_rel_shape():
    assert VisualStore.draft_rel("sedan-2020", "v1_a", 7) == "drafts/sedan-2020/v1_a/7.webp"


@pytest.mark.parametrize("bad", ["../x", "..", "a/b", "a\\b", "/etc", "C:\\x", "", "a b", "x\0y", "a\n"])
@pytest.mark.parametrize("slot", [0, 1])
def test_draft_rel_bad_components(bad, slot):
    parts = ["v1", "s1"]
    parts[slot] = bad
    with pytest.raises(ValueError):
        VisualStore.draft_rel(*parts, 1)


@pytest.mark.parametrize("token", [-1, True, 1.5, "1", None])
def test_draft_rel_bad_token(token):
    with pytest.raises(ValueError):
        VisualStore.draft_rel("v1", "s1", token)


def test_write_then_read_draft(store, root):
    rel = store.write_draft("v1", "s1", 3, DATA)
    assert rel == "drafts/v1/s1/3.webp"
    assert store.read_draft(rel) == DATA
    assert _tree(root / "drafts/v1/s1") == ["3.webp"]


def test_write_draft_bad_id_writes_nothing(store, root):
    with pytest.raises(ValueError):
        store.write_draft("../x", "s1", 1, DATA)
    assert _tree(root) == []


def test_draft_write_is_atomic(store, root, monkeypatch):
    store.write_draft("v1", "s1", 1, b"old")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("app.services.visual_store.os.replace", boom)
    with pytest.raises(OSError):
        store.write_draft("v1", "s1", 1, DATA)
    assert _tree(root / "drafts/v1/s1") == ["1.webp"]
    assert store.read_draft("drafts/v1/s1/1.webp") == b"old"


@pytest.mark.parametrize(
    "rel",
    [
        "../x",
        "drafts/../objects/x.webp",
        "drafts/v1/../s1/1.webp",
        "/drafts/v1/s1/1.webp",
        "drafts/v1/s1/1.webp\n",
        "drafts/v1/s1/-1.webp",
        "drafts/v1/s1/1.png",
        "drafts/v1/1.webp",
        "drafts/v1/s1/x/1.webp",
        "drafts/v1/s1/a.webp",
        "drafts/v1/s1/01.webp",
        "drafts\\v1\\s1\\1.webp",
        "C:/drafts/v1/s1/1.webp",
        "objects/" + SHA + ".webp",
        "",
    ],
)
def test_read_draft_and_delete_draft_refuse_bad_paths(store, root, tmp_path, rel):
    store.write_draft("v1", "s1", 1, DATA)
    (root / "objects").mkdir(exist_ok=True)
    (root / "objects" / f"{SHA}.webp").write_bytes(DATA)
    before = _tree(root)
    with pytest.raises(ValueError):
        store.read_draft(rel)
    with pytest.raises(ValueError):
        store.delete_draft(rel)
    assert _tree(root) == before


def test_read_draft_missing_raises_file_not_found(store):
    with pytest.raises(FileNotFoundError):
        store.read_draft("drafts/v1/s1/9.webp")


def test_delete_draft_removes_file_and_prunes_empty_dirs(store, root):
    a = store.write_draft("v1", "s1", 1, DATA)
    b = store.write_draft("v1", "s2", 1, DATA)
    store.delete_draft(a)
    assert _tree(root / "drafts") == ["v1", "v1/s2", "v1/s2/1.webp"]
    store.delete_draft(a)  # missing: no raise
    store.delete_draft(b)
    assert (root / "drafts").is_dir() and _tree(root / "drafts") == []


def test_symlinked_drafts_dir_is_refused(store, root, tmp_path):
    outside = tmp_path / "outside"
    (outside / "v1/s1").mkdir(parents=True)
    (outside / "v1/s1/1.webp").write_bytes(DATA)
    root.mkdir()
    _link_dir(root / "drafts", outside)
    with pytest.raises(ValueError):
        store.write_draft("v2", "s1", 1, DATA)
    with pytest.raises(ValueError):
        store.read_draft("drafts/v1/s1/1.webp")
    with pytest.raises(ValueError):
        store.delete_draft("drafts/v1/s1/1.webp")  # refused via link: nothing deleted
    assert (outside / "v1/s1/1.webp").read_bytes() == DATA
    assert _tree(outside) == ["v1", "v1/s1", "v1/s1/1.webp"]


def test_symlinked_variant_dir_is_refused(store, root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "drafts").mkdir(parents=True)
    _link_dir(root / "drafts/v1", outside)
    with pytest.raises(ValueError):
        store.write_draft("v1", "s1", 1, DATA)
    assert _tree(outside) == []


# --- sweep -------------------------------------------------------------------


def test_sweep_removes_old_drafts_and_unreferenced_objects_only(store, root):
    other = b"other"
    other_sha = hashlib.sha256(other).hexdigest()
    third = b"third"
    third_sha = hashlib.sha256(third).hexdigest()
    store.put_object(SHA, DATA)  # old, referenced
    store.put_object(other_sha, other)  # old, unreferenced
    store.put_object(third_sha, third)  # fresh, unreferenced (uploaded, not yet activated)
    for sha in (SHA, other_sha):
        _age(root / "objects" / f"{sha}.webp", 7200)
    old = store.write_draft("v1", "s1", 1, DATA)
    new = store.write_draft("v1", "s1", 2, DATA)
    _age(root / old, 7200)

    removed = store.sweep(3600, {SHA})

    assert removed == 2
    assert store.has_object(SHA) and not store.has_object(other_sha) and store.has_object(third_sha)
    assert not (root / old).exists() and (root / new).exists()


def test_sweep_prunes_empty_draft_dirs(store, root):
    rel = store.write_draft("v1", "s1", 1, DATA)
    _age(root / rel, 7200)
    assert store.sweep(3600, set()) == 1
    assert _tree(root / "drafts") == []


def test_sweep_uses_injectable_clock(store, root):
    rel = store.write_draft("v1", "s1", 1, DATA)
    mtime = (root / rel).stat().st_mtime
    assert store.sweep(3600, set(), now=mtime + 10) == 0
    assert store.sweep(3600, set(), now=mtime + 3601) == 1


def test_sweep_on_missing_root_returns_zero(store):
    assert store.sweep(0, set()) == 0


def test_sweep_never_raises_and_keeps_going(store, root, monkeypatch):
    a = store.write_draft("v1", "s1", 1, DATA)
    b = store.write_draft("v1", "s1", 2, DATA)
    for rel in (a, b):
        _age(root / rel, 7200)
    real = os.unlink
    calls = []

    def flaky(path, *args, **kwargs):
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError("locked")
        return real(path, *args, **kwargs)

    monkeypatch.setattr("app.services.visual_store.os.unlink", flaky)
    assert store.sweep(3600, set()) == 1
    assert len(calls) == 2


def test_sweep_does_not_follow_links(store, root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.webp").write_bytes(DATA)
    _age(outside / "keep.webp", 7200)
    (root / "drafts").mkdir(parents=True)
    _link_dir(root / "drafts/v1", outside)
    store.sweep(0, set())
    assert (outside / "keep.webp").exists()


def test_reput_of_an_old_object_refreshes_its_grace_period(store, root):
    store.put_object(SHA, DATA)
    _age(root / "objects" / f"{SHA}.webp", 7200)
    store.put_object(SHA, DATA)
    assert store.sweep(3600, set()) == 0
    assert store.has_object(SHA)


def test_delete_draft_logs_and_swallows_oserror(store, monkeypatch, caplog):
    rel = store.write_draft("v1", "s1", 1, DATA)

    def boom(*a, **k):
        raise PermissionError("locked")

    monkeypatch.setattr("app.services.visual_store.os.unlink", boom)
    with caplog.at_level("WARNING", logger="visual_store"):
        store.delete_draft(rel)
    assert "PermissionError" in caplog.text and "v1" not in caplog.text


def test_sweep_logs_one_summary_with_class_names(store, root, monkeypatch, caplog):
    for n in (1, 2):
        _age(root / store.write_draft("v1", "s1", n, DATA), 7200)

    def boom(*a, **k):
        raise PermissionError("locked")

    monkeypatch.setattr("app.services.visual_store.os.unlink", boom)
    with caplog.at_level("WARNING", logger="visual_store"):
        assert store.sweep(3600, set()) == 0
    records = [r for r in caplog.records if r.name == "visual_store"]
    assert len(records) == 1 and "PermissionError" in records[0].getMessage() and "2" in records[0].getMessage()
    assert "v1" not in records[0].getMessage()


def test_sweep_logs_nothing_when_clean(store, root, caplog):
    _age(root / store.write_draft("v1", "s1", 1, DATA), 7200)
    with caplog.at_level("WARNING", logger="visual_store"):
        assert store.sweep(3600, set()) == 1
    assert caplog.records == []
