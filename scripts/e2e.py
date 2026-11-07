"""End-to-end check against a running stack: health, sign in, add a car, chat (text, photo, audio, video), clean up.

Prints one OK/FAIL line per step and per message with time-to-first-token and total time. Exits non-zero
on any failure, and also when a media message was skipped for a missing file unless --allow-skip is given.
With --register, an account that does not exist yet (login answers 401) is registered first.
Never prints tokens or the password.
Usage: uv run scripts/e2e.py --base http://localhost:8000 --email you@example.com --password ...
       [--register] [--media-dir tests/fixtures/e2e] [--allow-skip]
"""
import argparse
import json
import mimetypes
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
CAR = {"make": "Toyota", "model": "Echo", "year": 2001}
POLL_TIMEOUT_S = 300
POLL_EVERY_S = 5
# (label, text, file name in --media-dir or None)
MESSAGES = [
    ("text", "My 2001 Toyota Echo is overheating in traffic. What should I check first?", None),
    ("photo", "What is this under the hood?", "coolant-reservoir.jpg"),
    ("audio", "Listen to this noise.", "engine-noise.m4a"),
    ("video", "Here is a short clip of the engine bay.", "engine-bay.mp4"),
]


def parse_sse(lines: Iterable[str]) -> Iterator[tuple[str, dict]]:
    """Yield (event, data) for each complete SSE frame; `lines` are text lines without newlines."""
    event, data = "message", []
    for line in lines:
        if line == "":
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:  # stream ended without the trailing blank line
        yield event, json.loads("\n".join(data))


def check_events(events: list[tuple[str, dict]]) -> list[str]:
    """Problems with one turn's events; empty means the turn is good."""
    names = [e for e, _ in events]
    problems = [f"error event: {d.get('message', '')}" for e, d in events if e == "error"]
    if "token" not in names:
        problems.append("no token event")
    if names.count("state") != 1:
        problems.append(f"expected exactly 1 state event, got {names.count('state')}")
    return problems


@dataclass
class Result:
    label: str
    problems: list[str]
    first_token_s: float | None = None
    total_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.problems

    def line(self) -> str:
        ttft = "n/a" if self.first_token_s is None else f"{self.first_token_s:.1f}s"
        head = f"{'OK  ' if self.ok else 'FAIL'} {self.label:<6} first token {ttft}, total {self.total_s:.1f}s"
        return head if self.ok else f"{head} -- {'; '.join(self.problems)}"


def send_message(client: httpx.Client, session_id: str, label: str, text: str, media: Path | None) -> Result:
    start = time.monotonic()
    first = None
    events: list[tuple[str, dict]] = []
    handle = media.open("rb") if media else None
    try:
        files = None
        if handle:
            ctype = mimetypes.guess_type(media.name)[0] or "application/octet-stream"
            files = [("files", (media.name, handle, ctype))]
        with client.stream("POST", f"/api/v1/sessions/{session_id}/messages", data={"text": text}, files=files) as resp:
            if resp.status_code != 200:
                return Result(label, [f"HTTP {resp.status_code}"], None, time.monotonic() - start)
            for event, data in parse_sse(resp.iter_lines()):
                if event == "token" and first is None:
                    first = time.monotonic() - start
                events.append((event, data))
    except (httpx.HTTPError, ValueError) as exc:
        return Result(label, [type(exc).__name__], first, time.monotonic() - start)
    finally:
        if handle:
            handle.close()
    return Result(label, check_events(events), first, time.monotonic() - start)


def wait_for_kb(client: httpx.Client, car_id: str) -> str:
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while True:
        resp = client.get("/api/v1/cars")
        resp.raise_for_status()
        status = next((c.get("kb_status", "pending") for c in resp.json() if c.get("id") == car_id), "pending")
        if status != "pending":
            return status
        if time.monotonic() > deadline:
            raise TimeoutError(f"kb_status still pending after {POLL_TIMEOUT_S}s")
        time.sleep(POLL_EVERY_S)


def health_problems(body: dict) -> list[str]:
    """Problems in a /health body; empty means the stack is ready."""
    return [f"{key} down" for key in ("database", "deepseek") if body.get(key) is not True]


def sign_in(client: httpx.Client, email: str, password: str, register: bool) -> str | None:
    """Access token from login, or from register when login says 401 and `register` is set; None on failure."""
    creds = {"email": email, "password": password}
    resp = client.post("/api/v1/auth/login", json=creds)
    if resp.status_code == 401 and register:
        print("     login 401, registering the account")
        resp = client.post("/api/v1/auth/register", json=creds)
    if resp.status_code != 200:
        print(f"FAIL sign in: HTTP {resp.status_code}")
        return None
    return resp.json()["access_token"]


def run(base: str, email: str, password: str, media_dir: Path, allow_skip: bool, register: bool) -> int:
    failures = skipped = 0
    with httpx.Client(base_url=base, timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        try:
            resp = client.get("/health")
            problems = health_problems(resp.json())
            if resp.status_code != 200:
                problems = problems or [f"HTTP {resp.status_code}"]
        except (httpx.HTTPError, ValueError) as exc:
            problems = [type(exc).__name__]
        if problems:
            print(f"FAIL health: {'; '.join(problems)}")
            return 1
        print("OK   health")

        token = sign_in(client, email, password, register)
        if token is None:
            return 1
        client.headers["Authorization"] = f"Bearer {token}"
        print("OK   sign in")

        resp = client.post("/api/v1/cars", json=CAR)
        if resp.status_code != 200:
            print(f"FAIL add car: HTTP {resp.status_code}")
            return 1
        car_id = resp.json()["id"]
        try:
            print(f"OK   car added, kb_status={wait_for_kb(client, car_id)}")
        except (TimeoutError, httpx.HTTPError) as exc:
            print(f"FAIL kb wait: {exc}")
            return 1

        resp = client.post("/api/v1/sessions", json={"car_id": car_id})
        if resp.status_code != 200:
            print(f"FAIL create session: HTTP {resp.status_code}")
            return 1
        session_id = resp.json()["id"]
        try:
            for label, text, name in MESSAGES:
                media = media_dir / name if name else None
                if media and not media.is_file():
                    print(f"SKIP {label:<6} missing media file {media}")
                    skipped += 1
                    continue
                result = send_message(client, session_id, label, text, media)
                print(result.line(), flush=True)
                failures += not result.ok
        finally:
            try:
                status = client.delete(f"/api/v1/sessions/{session_id}").status_code
            except httpx.HTTPError as exc:
                status = type(exc).__name__
            print(f"{'OK  ' if status == 200 else 'FAIL'} delete session ({status})")
            failures += status != 200
    if skipped and not allow_skip:
        print(f"FAIL {skipped} message(s) skipped; pass --allow-skip to accept")
    return 1 if failures or (skipped and not allow_skip) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--register", action="store_true", help="register the account if login answers 401")
    parser.add_argument("--media-dir", type=Path, default=ROOT / "tests" / "fixtures" / "e2e")
    parser.add_argument("--allow-skip", action="store_true", help="exit 0 even if media files are missing")
    args = parser.parse_args()
    sys.exit(run(args.base.rstrip("/"), args.email, args.password, args.media_dir, args.allow_skip, args.register))
