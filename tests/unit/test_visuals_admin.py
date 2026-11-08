"""scripts/visuals_admin.py: the operator CLI over a fake repository and a tmp_path VisualStore (no database)."""

import importlib.util
import io
import re
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.services import visual_image
from app.services.visual_store import VisualStore

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("visuals_admin", ROOT / "scripts" / "visuals_admin.py")
va = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(va)

V = "toyota-echo-2001-sedan"
STYLE = va.crud.STYLE_VERSION


def _png(color=(200, 30, 30, 255)) -> bytes:
    img = Image.new("RGBA", (600, 200), (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle((20, 20, 580, 180), fill=color)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _webp(color=(200, 30, 30, 255)) -> tuple[bytes, str]:
    data, sha, _, _ = visual_image.normalize(_png(color))
    return data, sha


class FakeCrud:
    STYLE_VERSION = STYLE

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.calls: list[tuple] = []
        self.publish_result = True
        self.seed_error: Exception | None = None

    def add(self, variant=V, **kw):
        row = {"id": f"id-{variant}", "variant_key": variant, "state": "review", "published_revision": 0,
               "fencing_token": 0, "draft_sha256": None, "attempt_count": 1, "sha256": None, **kw}
        self.rows[variant] = row
        return row

    async def get_visual(self, v):
        return self.rows.get(v)

    async def ensure_job(self, v, provenance="curated"):
        self.calls.append(("ensure_job", v))
        if v in self.rows:
            return self.rows[v], False
        return self.add(v, state="queued", attempt_count=0), True

    async def list_review(self):
        return [r for r in self.rows.values() if r["state"] == "review"]

    async def publish(self, job_id, rev, sha, w, h):
        self.calls.append(("publish", job_id, rev, sha, w, h))
        return self.publish_result

    async def reject(self, v):
        self.calls.append(("reject", v))
        r = self.rows.get(v)
        return bool(r and r["state"] == "review")

    async def requeue(self, v):
        self.calls.append(("requeue", v))
        r = self.rows.get(v)
        return bool(r and r["state"] in ("failed", "review"))

    async def seed_review(self, v, sha, w, h):
        self.calls.append(("seed_review", v, sha, w, h))
        if self.seed_error:
            raise self.seed_error
        r = self.rows[v]
        r.update(state="review", draft_sha256=sha)
        return r


@pytest.fixture
def env(tmp_path, monkeypatch):
    fake = FakeCrud()
    store = VisualStore(tmp_path / "visuals")
    monkeypatch.setattr(va, "crud", fake)
    monkeypatch.setattr(va, "_store", lambda: store)

    async def noop(*a, **k):
        return None

    monkeypatch.setattr(va.pool, "init_pool", noop)
    monkeypatch.setattr(va.pool, "close_pool", noop)
    return fake, store, tmp_path


def _draft(fake, store, webp, sha, token=0, revision=0, **kw):
    fake.add(fencing_token=token, published_revision=revision, draft_sha256=sha, **kw)
    return store.write_draft(V, STYLE, token, webp)


def _clean(capsys, store):
    out = capsys.readouterr().out
    assert not re.search(r"[0-9a-f]{64}", out)
    assert not re.search(r"https?://|token=|signature|expires", out, re.I)
    assert str(store._base()) not in out
    return out


def test_approve_stale_revision_touches_nothing(env, capsys):
    fake, store, _ = env
    webp, sha = _webp()
    rel = _draft(fake, store, webp, sha, revision=2)
    assert va.main(["approve", V, "--revision", "1"]) == 2
    assert "refused: stale revision" in _clean(capsys, store)
    assert not store.has_object(sha) and not [c for c in fake.calls if c[0] == "publish"]
    assert store.read_draft(rel) == webp


def test_approve_not_in_review_and_unknown_row(env, capsys):
    fake, store, _ = env
    fake.add(state="ready", published_revision=1)
    assert va.main(["approve", V, "--revision", "1"]) == 2
    assert "refused: not in review" in _clean(capsys, store)
    assert va.main(["approve", "nope", "--revision", "0"]) == 1


def test_approve_success(env, capsys):
    fake, store, _ = env
    webp, sha = _webp()
    rel = _draft(fake, store, webp, sha, token=7, revision=3)
    assert va.main(["approve", V, "--revision", "3"]) == 0
    out = _clean(capsys, store)
    assert "approved" in out and "4" in out
    assert store.has_object(sha)
    assert ("publish", f"id-{V}", 3, sha, 1024, 420) in fake.calls
    with pytest.raises(FileNotFoundError):
        store.read_draft(rel)


def test_approve_publish_false_leaves_old_object(env, capsys):
    fake, store, _ = env
    old, old_sha = _webp((1, 2, 200, 255))
    store.put_object(old_sha, old)
    webp, sha = _webp()
    rel = _draft(fake, store, webp, sha, revision=1, sha256=old_sha)
    fake.publish_result = False
    assert va.main(["approve", V, "--revision", "1"]) == 2
    assert "refused: stale revision" in _clean(capsys, store)
    assert store.has_object(old_sha) and store.read_draft(rel) == webp
    assert fake.rows[V]["sha256"] == old_sha


def test_approve_draft_mismatch(env, capsys):
    fake, store, _ = env
    webp, _ = _webp()
    _draft(fake, store, webp, "0" * 64)
    assert va.main(["approve", V, "--revision", "0"]) == 2
    assert "refused: draft mismatch" in _clean(capsys, store)
    assert not [c for c in fake.calls if c[0] == "publish"]


def test_approve_invalid_draft(env, capsys):
    fake, store, _ = env
    _draft(fake, store, b"not an image", "0" * 64)
    assert va.main(["approve", V, "--revision", "0"]) == 2
    assert "refused: not_image" in _clean(capsys, store)


def test_approve_animated_draft(env, capsys):
    fake, store, _ = env
    frames = [Image.new("RGBA", (1024, 420), c) for c in ((255, 0, 0, 255), (0, 0, 255, 255))]
    buf = io.BytesIO()
    frames[0].save(buf, "WEBP", save_all=True, append_images=frames[1:], duration=50)
    _draft(fake, store, buf.getvalue(), "0" * 64)
    assert va.main(["approve", V, "--revision", "0"]) == 2
    assert capsys.readouterr().out.startswith("refused: ")
    assert not [c for c in fake.calls if c[0] == "publish"]


def test_approve_missing_draft_is_refused(env, capsys):
    fake, store, _ = env
    fake.add(draft_sha256="0" * 64)
    assert va.main(["approve", V, "--revision", "0"]) == 2
    assert "refused: draft missing" in _clean(capsys, store)


def test_seed_writes_draft_at_current_token(env, capsys):
    fake, store, tmp = env
    fake.add(state="failed", fencing_token=7)
    img = tmp / "a.png"
    img.write_bytes(_png())
    assert va.main(["seed", V, "--image", str(img)]) == 0
    out = _clean(capsys, store)
    assert "seeded" in out
    _, sha = _webp()
    assert store.read_draft(store.draft_rel(V, STYLE, 7))
    assert ("seed_review", V, sha, 1024, 420) in fake.calls
    assert fake.rows[V]["state"] == "review"


def test_seed_creates_row_when_missing(env, capsys):
    fake, store, tmp = env
    img = tmp / "a.png"
    img.write_bytes(_png())
    assert va.main(["seed", V, "--image", str(img)]) == 0
    assert fake.rows[V]["state"] == "review"
    assert store.read_draft(store.draft_rel(V, STYLE, 0))


def test_seed_refusals(env, capsys):
    fake, store, tmp = env
    good = tmp / "a.png"
    good.write_bytes(_png())
    link = tmp / "link.png"
    try:
        link.symlink_to(good)
    except OSError:
        link = None
    big = tmp / "big.bin"
    big.write_bytes(b"\0" * 20_000_001)
    bad = tmp / "bad.png"
    bad.write_bytes(b"hello")
    assert va.main(["seed", V, "--image", str(tmp / "missing.png")]) == 1
    assert va.main(["seed", V, "--image", str(tmp)]) == 1
    assert va.main(["seed", V, "--image", str(big)]) == 1
    assert va.main(["seed", "not-a-variant", "--image", str(good)]) == 1
    assert "unknown variant" in capsys.readouterr().out
    if link is not None:
        assert va.main(["seed", V, "--image", str(link)]) == 1
    assert not fake.calls
    assert va.main(["seed", V, "--image", str(bad)]) == 2
    assert "refused: not_image" in _clean(capsys, store)
    assert not fake.calls


def test_seed_on_generating_row_never_touches_the_workers_draft(env, capsys):
    fake, store, tmp = env
    fake.add(state="generating", fencing_token=3)
    worker_rel = store.write_draft(V, STYLE, 3, b"worker draft")
    img = tmp / "a.png"
    img.write_bytes(_png())
    assert va.main(["seed", V, "--image", str(img)]) == 2
    assert "refused: job running" in _clean(capsys, store)
    assert store.read_draft(worker_rel) == b"worker draft"
    assert not [c for c in fake.calls if c[0] == "seed_review"]


def test_seed_claimed_after_precheck_deletes_only_its_own_draft(env, capsys):
    fake, store, tmp = env
    fake.add(state="failed", fencing_token=3)
    fake.seed_error = ValueError("live lease")
    img = tmp / "a.png"
    img.write_bytes(_png())
    assert va.main(["seed", V, "--image", str(img)]) == 2
    assert "refused: job running" in _clean(capsys, store)
    with pytest.raises(FileNotFoundError):
        store.read_draft(store.draft_rel(V, STYLE, 3))


def test_seed_approve_refuses_when_draft_changed(env, capsys):
    fake, store, tmp = env
    img = tmp / "a.png"
    img.write_bytes(_png())
    orig = fake.seed_review

    async def seed_then_swap(*a):
        row = await orig(*a)
        row["draft_sha256"] = "1" * 64
        return row

    fake.seed_review = seed_then_swap
    assert va.main(["seed", V, "--image", str(img), "--approve"]) == 2
    assert "refused: draft changed" in _clean(capsys, store)
    assert not [c for c in fake.calls if c[0] == "publish"]


SECRET = "ab" * 32


def _quiet(capsys, name):
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert name in text and SECRET not in text and "secret" not in text


def test_put_object_oserror_is_quiet_exit_3(env, capsys, monkeypatch):
    fake, store, _ = env
    webp, sha = _webp()
    _draft(fake, store, webp, sha)

    def boom(*a):
        raise OSError(f"/secret/objects/.{SECRET}.webp.tmp")

    monkeypatch.setattr(store, "put_object", boom)
    assert va.main(["approve", V, "--revision", "0"]) == 3
    _quiet(capsys, "OSError")
    assert not [c for c in fake.calls if c[0] == "publish"]


def test_write_draft_oserror_is_quiet_exit_3(env, capsys, monkeypatch):
    fake, store, tmp = env
    img = tmp / "a.png"
    img.write_bytes(_png())

    def boom(*a):
        raise OSError(f"/secret/drafts/{SECRET}")

    monkeypatch.setattr(store, "write_draft", boom)
    assert va.main(["seed", V, "--image", str(img)]) == 3
    _quiet(capsys, "OSError")


def test_database_outage_mid_approve_is_exit_3(env, capsys, monkeypatch):
    fake, store, _ = env
    webp, sha = _webp()
    _draft(fake, store, webp, sha)

    async def down(*a):
        raise RuntimeError(f"connection to secret-host failed {SECRET}")

    monkeypatch.setattr(fake, "publish", down)
    assert va.main(["approve", V, "--revision", "0"]) == 3
    _quiet(capsys, "RuntimeError")


def test_seed_approve_end_to_end(env, capsys):
    fake, store, tmp = env
    img = tmp / "a.png"
    img.write_bytes(_png())
    assert va.main(["seed", V, "--image", str(img), "--approve"]) == 0
    out = _clean(capsys, store)
    assert "seeded" in out and "approved" in out
    _, sha = _webp()
    assert store.has_object(sha)
    assert ("publish", f"id-{V}", 0, sha, 1024, 420) in fake.calls


def test_reject(env, capsys):
    fake, store, _ = env
    webp, sha = _webp()
    rel = _draft(fake, store, webp, sha, token=5)
    assert va.main(["reject", V]) == 0
    assert "rejected" in _clean(capsys, store)
    with pytest.raises(FileNotFoundError):
        store.read_draft(rel)
    fake.rows[V]["state"] = "ready"
    assert va.main(["reject", V]) == 1
    assert "nothing to reject" in capsys.readouterr().out
    assert va.main(["reject", "ghost"]) == 1


def test_requeue(env, capsys):
    fake, store, _ = env
    fake.add(state="failed")
    assert va.main(["requeue", V]) == 0
    assert "requeued" in capsys.readouterr().out
    fake.rows[V]["state"] = "ready"
    assert va.main(["requeue", V]) == 1
    assert "nothing to requeue" in capsys.readouterr().out


def test_list_review_prints_variant_and_revision_only(env, capsys):
    fake, store, _ = env
    fake.add(published_revision=4, draft_sha256="a" * 64, sha256="b" * 64)
    fake.add("honda-civic-2008-sedan", state="ready")
    assert va.main(["list-review"]) == 0
    out = _clean(capsys, store)
    assert out.count("\n") == 1 and V in out and "4" in out and "review" in out
    assert "honda" not in out


def test_usage_errors_exit_1(env):
    assert va.main([]) == 1
    assert va.main(["approve", V]) == 1
