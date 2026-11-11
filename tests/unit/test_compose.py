"""Static checks on the deploy files: docker-compose.yml, Dockerfile, .dockerignore, and no hosted-backend leftovers."""
import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_TARGET = "/docker-entrypoint-initdb.d/01-schema.sql"


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def _env(service: dict) -> dict:
    env = service.get("environment", {})
    if isinstance(env, list):
        env = dict(item.split("=", 1) for item in env)
    return env


def _mounts(service: dict) -> dict:
    """target -> (source, read_only) for short-syntax volume entries."""
    out = {}
    for entry in service.get("volumes", []):
        source, target, *mode = entry.split(":")
        out[target] = (source, "ro" in mode)
    return out


def test_services(compose):
    assert set(compose["services"]) == {"db", "api", "visuals-worker"}


def test_db_is_not_published_and_restarts(compose):
    db = compose["services"]["db"]
    assert db["image"] == "postgres:16-alpine"
    assert "ports" not in db
    assert db["restart"] == "unless-stopped"


def test_db_credentials_and_required_password(compose):
    env = _env(compose["services"]["db"])
    assert env["POSTGRES_USER"] == "autocare"
    assert env["POSTGRES_DB"] == "autocare"
    assert re.fullmatch(r"\$\{POSTGRES_PASSWORD:\?[^}]+\}", env["POSTGRES_PASSWORD"])


def test_db_healthcheck_uses_pg_isready(compose):
    test = " ".join(compose["services"]["db"]["healthcheck"]["test"])
    assert "pg_isready" in test and "-U autocare" in test and "-d autocare" in test


def test_schema_mounted_read_only_from_an_existing_file(compose):
    source, read_only = _mounts(compose["services"]["db"])[SCHEMA_TARGET]
    assert read_only
    assert (ROOT / source).is_file()
    assert Path(source).as_posix().endswith("app/db/sql/schema.sql")


def test_named_volumes_back_postgres_data_and_media(compose):
    assert {"pgdata", "media"} <= set(compose["volumes"])
    assert _mounts(compose["services"]["db"])["/var/lib/postgresql/data"][0] == "pgdata"
    assert _mounts(compose["services"]["api"])["/data/media"][0] == "media"


def test_visuals_schema_mounted_after_the_main_schema(compose):
    source, read_only = _mounts(compose["services"]["db"])["/docker-entrypoint-initdb.d/02-vehicle_visuals.sql"]
    assert read_only
    assert (ROOT / source).is_file()
    assert Path(source).as_posix().endswith("app/db/sql/vehicle_visuals.sql")


def test_visuals_volume_is_declared_and_mounted_on_api(compose):
    assert "visuals" in compose["volumes"]
    assert _mounts(compose["services"]["api"])["/data/visuals"][0] == "visuals"


def test_visuals_worker_service(compose):
    worker = compose["services"]["visuals-worker"]
    assert worker["build"] == "." and worker["env_file"] == ".env"
    assert worker["command"][-3:] == ["python", "-m", "app.workers.vehicle_visuals"]
    assert worker["command"][:2] == ["uv", "run"]  # bare python is not the venv's
    assert worker["depends_on"]["db"]["condition"] == "service_healthy"
    assert worker["restart"] == "unless-stopped"
    assert "ports" not in worker
    assert worker["stop_grace_period"] == "130s"  # an in-flight provider job can run 120 s
    assert _mounts(worker)["/data/visuals"][0] == "visuals"
    assert _env(worker)["DATABASE_URL"] == _env(compose["services"]["api"])["DATABASE_URL"]


def test_api_waits_for_healthy_db(compose):
    api = compose["services"]["api"]
    assert api["depends_on"]["db"]["condition"] == "service_healthy"
    assert api["env_file"] == ".env"
    assert api["restart"] == "unless-stopped"
    assert api["ports"] == ["127.0.0.1:8000:8000"]  # plain HTTP stays on the host; Caddy terminates TLS


def test_api_database_url_points_at_db_service(compose):
    url = urlsplit(_env(compose["services"]["api"])["DATABASE_URL"])
    assert url.scheme == "postgresql"
    assert url.hostname == "db" and url.port == 5432
    assert url.username == "autocare" and url.path == "/autocare"
    assert url.password == "${POSTGRES_PASSWORD}"


def test_api_trusts_forwarded_headers_only_from_configured_proxy(compose):
    assert _env(compose["services"]["api"])["FORWARDED_ALLOW_IPS"] == "${FORWARDED_ALLOW_IPS:-127.0.0.1}"


def test_api_healthcheck_unchanged(compose):
    hc = compose["services"]["api"]["healthcheck"]
    assert "http://localhost:8000/health" in " ".join(hc["test"])
    assert hc["start_period"] == "90s"


@pytest.mark.parametrize("status, healthy", [(200, True), (503, False)])
def test_api_healthcheck_fails_on_non_2xx(compose, status, healthy):
    """Run the healthcheck's python one-liner against a local server answering `status`."""
    import subprocess
    import sys
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        test = compose["services"]["api"]["healthcheck"]["test"]
        assert test[:3] == ["CMD", "python", "-c"]
        code = test[3].replace("http://localhost:8000", f"http://127.0.0.1:{server.server_port}")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=20)
    finally:
        server.shutdown()
    assert (result.returncode == 0) is healthy


def _dockerfile() -> list[str]:
    return (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()


def _index(lines: list[str], pattern: str) -> int:
    return next(i for i, line in enumerate(lines) if re.search(pattern, line))


def test_dockerfile_creates_media_dir_before_dropping_root():
    lines = _dockerfile()
    media = _index(lines, r"^RUN mkdir -p /data/media && chown autocare:autocare /data/media$")
    assert media < _index(lines, r"^USER autocare$")


def test_dockerfile_creates_visuals_dir_owned_by_the_app_user():
    lines = _dockerfile()
    visuals = _index(lines, r"^RUN mkdir -p /data/visuals && chown autocare:autocare /data/visuals$")
    assert visuals < _index(lines, r"^USER autocare$")


def test_dockerfile_bake_step_placeholders():
    text = "\n".join(_dockerfile())
    bake = text[text.index("python -m app.services.hearing") - 400 : text.index("python -m app.services.hearing")]
    assert "DATABASE_URL=postgresql://" in bake
    assert "JWT_SECRET_KEY=" in bake and "DEEPSEEK_API_KEY=" in bake


def test_dockerfile_cmd_runs_one_uvicorn_worker_behind_proxy_headers():
    cmd = next(line for line in _dockerfile() if line.startswith("CMD "))
    assert "uvicorn" in cmd and '"--proxy-headers"' in cmd
    assert "--workers" not in cmd  # one worker: rate limits and the Whisper/CLAP models are per process
    assert '"--timeout-graceful-shutdown", "45"' in cmd  # in-flight replies get 45 s of the 60 s stop grace period


def test_dockerignore_keeps_scripts_and_drops_car_data():
    ignored = {line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()}
    assert "car_data/" in ignored and ".env*" in ignored
    assert "backups/" in ignored  # database dumps and media tarballs must never reach the image
    assert not any(entry.strip("/") in {"scripts", "scripts/*", "app"} for entry in ignored)


def test_backup_script_dumps_db_and_media_and_rotates():
    raw = (ROOT / "scripts" / "backup.sh").read_bytes()
    assert b"\r" not in raw  # bash on the VPS chokes on CRLF
    script = raw.decode("utf-8")
    assert script.startswith("#!/usr/bin/env bash\n") and "set -euo pipefail" in script
    assert "docker compose exec -T db pg_dump -U autocare" in script and "| gzip" in script
    assert '"$volume:/data:ro"' in script and "tar -czf -" in script
    assert "KEEP=14" in script and "crontab" in script.lower()
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "backups/" in gitignore


def test_no_hosted_backend_leftovers():
    needle = "supa" + "base"  # split so this file does not match itself
    hits = []
    paths = [ROOT / "main.py", *(ROOT / "app").rglob("*"), *(ROOT / "scripts").rglob("*")]
    for path in paths:
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if needle in path.read_text(encoding="utf-8", errors="ignore").lower():
            hits.append(path.relative_to(ROOT).as_posix())
    assert hits == []


def test_every_service_rotates_its_logs(compose):
    for name in ("db", "api", "visuals-worker"):
        assert compose["services"][name]["logging"] == {
            "driver": "json-file", "options": {"max-size": "10m", "max-file": "5"},
        }


def test_the_database_outlives_the_api_under_memory_pressure(compose):
    s = compose["services"]
    assert s["db"]["oom_score_adj"] == -500
    assert s["api"]["oom_score_adj"] == s["visuals-worker"]["oom_score_adj"] == 500


def test_app_containers_drop_capabilities(compose):
    for name in ("api", "visuals-worker"):
        s = compose["services"][name]
        assert s["cap_drop"] == ["ALL"] and s["security_opt"] == ["no-new-privileges:true"] and s["pids_limit"] == 512
    assert not {"cap_drop", "security_opt", "pids_limit"} & set(compose["services"]["db"])


def test_api_has_time_to_finish_replies_on_stop(compose):
    assert compose["services"]["api"]["stop_grace_period"] == "60s"


def test_compose_preloads_models_on_the_api_only(compose):
    assert _env(compose["services"]["api"])["PRELOAD_MODELS"] == "true"
    assert "PRELOAD_MODELS" not in _env(compose["services"]["visuals-worker"])
