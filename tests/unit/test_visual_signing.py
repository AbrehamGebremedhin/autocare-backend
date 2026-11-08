from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

import pytest

from app.core.config import Settings, get_settings
from app.services import visual_signing

SHA = "ab" * 32
NOW = 1_800_000_000.0


@pytest.fixture(autouse=True)
def base_url(monkeypatch):
    monkeypatch.setattr(get_settings(), "PUBLIC_BASE_URL", "https://api.example.com/")
    monkeypatch.setattr(get_settings(), "VISUAL_URL_TTL_SECONDS", 900)


def _parts(url):
    q = parse_qs(urlsplit(url).query)
    return q["exp"][0], q["sig"][0]


def test_settings_defaults():
    f = Settings.model_fields
    assert f["VISUALS_DIR"].default == "/data/visuals"
    assert f["PUBLIC_BASE_URL"].default == ""
    assert f["VISUAL_URL_TTL_SECONDS"].default == 900
    assert f["VISUAL_PREPARE_PER_HOUR"].default == 20
    assert f["VISUAL_JOBS_PER_HOUR"].default == 60
    assert f["VISUAL_POLL_SECONDS"].default == 10
    assert f["VISUAL_PROVIDER_ENABLED"].default is False
    assert f["VISUAL_PROVIDER_DAILY_BUDGET"].default == 20
    assert f["VISUAL_CURATED_DIR"].default == "/data/visuals/curated"


def test_sign_then_verify_round_trips():
    url, iso = visual_signing.sign_url(SHA, now=NOW)
    exp, sig = _parts(url)
    assert exp == str(int(NOW) + 900)
    assert datetime.fromisoformat(iso) == datetime.fromtimestamp(int(exp), tz=timezone.utc)
    assert visual_signing.verify(SHA, exp, sig, now=NOW + 10)


def test_expired_is_refused():
    exp, sig = _parts(visual_signing.sign_url(SHA, now=NOW)[0])
    assert not visual_signing.verify(SHA, exp, sig, now=NOW + 901)


def test_far_future_exp_is_refused():
    exp, sig = _parts(visual_signing.sign_url(SHA, now=NOW + 5000)[0])
    assert not visual_signing.verify(SHA, exp, sig, now=NOW)


def test_tampered_sha_or_exp_is_refused():
    exp, sig = _parts(visual_signing.sign_url(SHA, now=NOW)[0])
    assert not visual_signing.verify("cd" * 32, exp, sig, now=NOW)
    assert not visual_signing.verify(SHA, str(int(exp) + 1), sig, now=NOW)
    assert not visual_signing.verify(SHA, exp, ("0" if sig[0] != "0" else "1") + sig[1:], now=NOW)


@pytest.mark.parametrize("bad", ["", "abc", "-1", "9" * 30, "é" * 10, "٣" * 5, "a" * 3000, "1" * 3000, "ab" * 31, "AB" * 32, "ab" * 33])
def test_garbage_exp_and_sig_never_raise(bad):
    exp, sig = _parts(visual_signing.sign_url(SHA, now=NOW)[0])
    assert visual_signing.verify(SHA, bad, sig, now=NOW) is False
    assert visual_signing.verify(SHA, exp, bad, now=NOW) is False
    assert visual_signing.verify(bad, exp, sig, now=NOW) is False


def test_url_is_https_and_contains_no_query_beyond_exp_sig():
    url, _ = visual_signing.sign_url(SHA, now=NOW)
    parts = urlsplit(url)
    assert parts.scheme == "https"
    assert parts.path == f"/api/v1/visuals/files/{SHA}.webp"
    assert sorted(parse_qs(parts.query)) == ["exp", "sig"]
    assert "//api" not in url.replace("https://", "")


@pytest.mark.parametrize("value", ["", "http://api.example.com"])
def test_unset_or_non_https_public_base_url_yields_no_url(monkeypatch, value):
    monkeypatch.setattr(get_settings(), "PUBLIC_BASE_URL", value)
    assert visual_signing.sign_url(SHA, now=NOW) == ("", "")
