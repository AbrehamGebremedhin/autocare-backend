"""Uploaded originals on a local volume: MEDIA_DIR/{user_id}/{session_id}/{message_id}/{n}{ext}.

Refs handed back carry paths relative to MEDIA_DIR only; absolute server paths never leave this module.
"""

import asyncio
import errno
import os
import re
import shutil
import uuid
from pathlib import Path

from app.core.config import get_settings
from app.utils.logger import get_logger_instance

logger = get_logger_instance("media_store")

_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_EXTS = {".jpg", ".mp4", ".m4a"}


def _component(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("invalid path component")
    return value


class MediaStore:
    def __init__(self, root: str | os.PathLike | None = None):
        self._root = root

    def _base(self) -> Path:
        return Path(self._root if self._root is not None else get_settings().MEDIA_DIR).resolve()

    def _save(self, src: Path, user_id: str, session_id: str, message_id: str, n: int, ext: str) -> str:
        parts = [_component(user_id), _component(session_id), _component(message_id)]
        if ext not in _EXTS:
            raise ValueError("invalid extension")
        name = f"{int(n)}{ext}"
        base = self._base()
        folder = base.joinpath(*parts)
        if not folder.resolve().is_relative_to(base):  # a planted symlink would resolve outside
            raise ValueError("path escapes the media directory")
        base.mkdir(mode=0o700, parents=True, exist_ok=True)
        floor = get_settings().MEDIA_MIN_FREE_MB * 1024 * 1024
        if shutil.disk_usage(base).free < floor:
            raise OSError(errno.ENOSPC, "media volume is low on free space")
        current = base
        for part in parts:
            current = current / part
            if not current.is_symlink():
                current.mkdir(mode=0o700, exist_ok=True)
        folder = folder.resolve()
        if not folder.is_relative_to(base):
            raise ValueError("path escapes the media directory")
        dest = folder / name
        if dest.is_symlink():
            raise ValueError("destination is a symlink")
        tmp = folder / f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            shutil.copyfile(src, tmp)
            os.chmod(tmp, 0o600)
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        return dest.relative_to(base).as_posix()

    async def upload(
        self, user_id: str, session_id: str, message_id: str, n: int, kind: str, ext: str, src: str | os.PathLike
    ) -> dict:
        """Copy the validated local file `src` into the store; returns a MediaRef-shaped dict."""
        path = await asyncio.to_thread(self._save, Path(src), user_id, session_id, message_id, n, ext)
        return {"kind": kind, "path": path}

    def _remove(self, user_id: str, session_id: str) -> int:
        base = self._base()
        folder = base / _component(user_id) / _component(session_id)
        if not folder.exists() and not folder.is_symlink():
            return 0
        if folder.is_symlink() or folder.resolve() != folder:
            raise ValueError("refusing to delete through a symlink")
        if not folder.resolve().is_relative_to(base):
            raise ValueError("path escapes the media directory")
        count = sum(len(files) for _, _, files in os.walk(folder))  # os.walk does not follow symlinks
        shutil.rmtree(folder)
        return count

    async def delete_prefix(self, user_id: str, session_id: str) -> int:
        """Remove the session's folder; returns the number of files removed. Best effort, never raises."""
        try:
            return await asyncio.to_thread(self._remove, user_id, session_id)
        except Exception as exc:
            await logger.warning(f"media cleanup failed: {type(exc).__name__}")
            return 0
