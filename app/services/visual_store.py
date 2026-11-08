"""Private artwork store on a local volume under VISUALS_DIR.

  drafts/{variant_key}/{style_version}/{token}.webp   unpublished, private, fenced by an integer token
  objects/{sha256}.webp                                published, immutable, checksum-addressed

Callers hand in ids and hashes that may originate from a URL, so every path is built from validated parts and
symlinks/junctions are refused along the way. Methods are synchronous file I/O; async callers use asyncio.to_thread.
"""

import hashlib
import os
import re
import stat
import time
import uuid
from pathlib import Path

from app.core.config import get_settings
from app.services.media_store import _component

_SHA = re.compile(r"[0-9a-f]{64}")
_DRAFT = re.compile(r"drafts/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)/(0|[1-9][0-9]*)\.webp")


def _is_link(path: Path) -> bool:
    return path.is_symlink() or os.path.isjunction(path)


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _is_regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


class VisualStore:
    def __init__(self, root: str | os.PathLike | None = None):
        self._root = root

    def _base(self) -> Path:
        return Path(self._root if self._root is not None else get_settings().VISUALS_DIR).resolve()

    def _dir(self, *parts: str, create: bool = False) -> Path:
        """base/parts as a directory; raises ValueError if any existing component is a symlink or junction."""
        current = self._base()
        for part in parts:
            current = current / part
            if _is_link(current):
                raise ValueError("refusing to use a symlinked directory")
        if create:
            current.mkdir(mode=0o700, parents=True, exist_ok=True)
            if _is_link(current) or not current.resolve().is_relative_to(self._base()):
                raise ValueError("path escapes the visuals directory")
        return current

    @staticmethod
    def draft_rel(variant_key: str, style_version: str, token: int) -> str:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError("invalid fencing token")
        return f"drafts/{_component(variant_key)}/{_component(style_version)}/{token}.webp"

    @staticmethod
    def _parse_draft(rel: object) -> tuple[str, str, str]:
        m = _DRAFT.fullmatch(rel) if isinstance(rel, str) else None
        if not m:
            raise ValueError("invalid draft path")
        return m.group(1), m.group(2), f"{m.group(3)}.webp"

    @staticmethod
    def _atomic_write(folder: Path, name: str, data: bytes) -> Path:
        dest = folder / name
        if _is_link(dest):
            raise ValueError("destination is a symlink")
        tmp = folder / f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            os.chmod(tmp, 0o600)
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        return dest

    def write_draft(self, variant_key: str, style_version: str, token: int, data: bytes) -> str:
        rel = self.draft_rel(variant_key, style_version, token)
        folder = self._dir("drafts", variant_key, style_version, create=True)
        self._atomic_write(folder, f"{token}.webp", data)
        return rel

    def _draft_file(self, rel: str) -> Path:
        variant_key, style_version, name = self._parse_draft(rel)
        return self._dir("drafts", variant_key, style_version) / name

    def read_draft(self, rel: str) -> bytes:
        path = self._draft_file(rel)
        if _is_link(path):
            raise ValueError("draft is a symlink")
        return path.read_bytes()

    def delete_draft(self, rel: str) -> None:
        """Remove one draft and prune emptied parents; refuses invalid paths and never raises."""
        try:
            path = self._draft_file(rel)
            if _is_regular(path):
                os.unlink(path)
        except (ValueError, OSError):
            return
        for parent in (path.parent, path.parent.parent):
            try:
                os.rmdir(parent)
            except OSError:
                break

    def put_object(self, sha256: str, data: bytes) -> None:
        if not _valid_sha(sha256) or hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError("object hash mismatch")
        folder = self._dir("objects", create=True)
        dest = folder / f"{sha256}.webp"
        if _is_link(dest):
            raise ValueError("destination is a symlink")
        if _is_regular(dest) and dest.stat().st_size == len(data):
            return  # content-addressed and immutable: already stored
        self._atomic_write(folder, dest.name, data)

    def object_path(self, sha256: str) -> Path | None:
        if not _valid_sha(sha256):
            return None
        try:
            path = self._dir("objects") / f"{sha256}.webp"
        except ValueError:
            return None
        return path if _is_regular(path) else None

    def has_object(self, sha256: str) -> bool:
        return self.object_path(sha256) is not None

    def sweep(self, older_than_seconds: int, keep_shas: set[str], now: float | None = None) -> int:
        """Delete drafts and unreferenced objects older than the grace period; returns files removed, never raises."""
        cutoff = (time.time() if now is None else now) - older_than_seconds
        removed = 0
        try:
            removed += self._sweep_dir("drafts", cutoff, lambda name: True)
            removed += self._sweep_dir(
                "objects", cutoff, lambda name: name.endswith(".webp") and name[:-5] not in keep_shas
            )
        except Exception:  # best effort by contract
            pass
        return removed

    def _sweep_dir(self, top: str, cutoff: float, doomed) -> int:
        try:
            folder = self._dir(top)
        except ValueError:
            return 0
        removed, dirs = 0, []
        for dirpath, dirnames, files in os.walk(folder):
            dirnames[:] = [d for d in dirnames if not _is_link(Path(dirpath, d))]  # junctions are not symlinks
            dirs.extend(Path(dirpath, d) for d in dirnames)
            for name in files:
                path = Path(dirpath, name)
                try:
                    st = os.lstat(path)
                    if stat.S_ISREG(st.st_mode) and st.st_mtime < cutoff and doomed(name):
                        os.unlink(path)
                        removed += 1
                except OSError:
                    continue
        for d in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
            try:
                os.rmdir(d)  # only succeeds when empty
            except OSError:
                pass
        return removed
