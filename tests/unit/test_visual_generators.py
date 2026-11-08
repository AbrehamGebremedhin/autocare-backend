"""Generators: the curated file source, the (unconfigured) provider, and build_generator. No network."""

import pytest

from app.services import visual_generators as vg

pytestmark = pytest.mark.usefixtures("no_network")


async def test_curated_reads_a_file(tmp_path):
    (tmp_path / "toyota-echo.png").write_bytes(b"png-bytes")
    gen = vg.CuratedGenerator(str(tmp_path))
    assert gen.charged is False
    assert await gen.generate("toyota-echo", "workshop-v1") == b"png-bytes"


async def test_curated_prefers_webp_then_png_then_jpg(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"jpg")
    (tmp_path / "a.png").write_bytes(b"png")
    gen = vg.CuratedGenerator(str(tmp_path))
    assert await gen.generate("a", "s") == b"png"
    (tmp_path / "a.webp").write_bytes(b"webp")
    assert await gen.generate("a", "s") == b"webp"


async def test_curated_no_source(tmp_path):
    with pytest.raises(vg.GeneratorError) as err:
        await vg.CuratedGenerator(str(tmp_path)).generate("missing", "s")
    assert err.value.code == "no_source"


@pytest.mark.parametrize("key", ["../x", "a/b", "a\\b", "", "a.b", "..", "a\n"])
async def test_curated_rejects_hostile_keys_without_touching_disk(tmp_path, monkeypatch, key):
    monkeypatch.setattr(vg.Path, "is_file", lambda self: pytest.fail("disk touched"))
    monkeypatch.setattr(vg.Path, "stat", lambda self, **kw: pytest.fail("disk touched"))
    with pytest.raises(vg.GeneratorError) as err:
        await vg.CuratedGenerator(str(tmp_path)).generate(key, "s")
    assert err.value.code == "invalid_variant"


async def test_curated_file_over_limit_is_too_large(tmp_path, monkeypatch):
    (tmp_path / "big.png").write_bytes(b"x" * 10)
    monkeypatch.setattr(vg, "MAX_SOURCE_BYTES", 5)
    with pytest.raises(vg.GeneratorError) as err:
        await vg.CuratedGenerator(str(tmp_path)).generate("big", "s")
    assert err.value.code == "too_large"


async def test_curated_reads_off_the_loop_thread(tmp_path, monkeypatch):
    import threading

    (tmp_path / "t.png").write_bytes(b"1")
    seen = []
    real = vg.CuratedGenerator._read

    def spy(self, key):
        seen.append(threading.current_thread())
        return real(self, key)

    monkeypatch.setattr(vg.CuratedGenerator, "_read", spy)
    await vg.CuratedGenerator(str(tmp_path)).generate("t", "s")
    assert seen and seen[0] is not threading.main_thread()


async def test_provider_call_is_not_configured():
    with pytest.raises(vg.GeneratorError) as err:
        await vg.ProviderGenerator()._call("anything")
    assert err.value.code == "provider_not_configured"
    assert vg.ProviderGenerator.charged is True


async def test_provider_prompt_is_the_catalog_label_only(monkeypatch):
    monkeypatch.setattr(vg, "label", lambda key: "Toyota Echo 2001")
    prompts = []

    async def call(self, prompt):
        prompts.append(prompt)
        return b"img"

    monkeypatch.setattr(vg.ProviderGenerator, "_call", call)
    assert await vg.ProviderGenerator().generate("toyota-echo", "workshop-v1") == b"img"
    assert prompts == [vg.PROMPT_TEMPLATE.format(label="Toyota Echo 2001")]


async def test_provider_timeout_maps_to_provider_timeout(monkeypatch):
    import asyncio

    monkeypatch.setattr(vg, "label", lambda key: "x")
    monkeypatch.setattr(vg, "PROVIDER_TIMEOUT_SECONDS", 0.05)

    async def slow(self, prompt):
        await asyncio.sleep(5)

    monkeypatch.setattr(vg.ProviderGenerator, "_call", slow)
    with pytest.raises(vg.GeneratorError) as err:
        await vg.ProviderGenerator().generate("k", "s")
    assert err.value.code == "provider_timeout"


def test_build_generator_is_curated_by_default_and_never_builds_the_provider(monkeypatch):
    def boom(self, *a, **k):
        raise AssertionError("provider constructed")

    monkeypatch.setattr(vg.ProviderGenerator, "__init__", boom)
    gen = vg.build_generator()
    assert isinstance(gen, vg.CuratedGenerator) and gen.charged is False


def test_build_generator_provider_only_when_enabled(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "VISUAL_PROVIDER_ENABLED", True)
    assert isinstance(vg.build_generator(), vg.ProviderGenerator)
