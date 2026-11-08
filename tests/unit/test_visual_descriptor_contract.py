"""The mobile client's acceptance checklist, re-implemented, against every descriptor the service can produce."""
import copy
import re

from tests.unit.test_vehicle_visuals import CIVIC_CAR, ECHO, ECHO_CAR, LADA_CAR, SHA, run, vv, world  # noqa: F401

STATES = ("ready", "pending", "fallback")
REASONS = ("unknown_variant", "choose_variant", "temporarily_unavailable")


def client_accepts(d: dict) -> bool:
    if d.get("schema_version") != 1 or d.get("preset") not in (None, "garage_roll_v1"):
        return False
    state = d.get("state")
    if state not in STATES:
        return False
    if "options" in d and d.get("reason") != "choose_variant":
        return False
    if state == "ready":
        a = d.get("artwork")
        if not isinstance(a, dict):
            return False
        if (a.get("format"), a.get("width"), a.get("height"), a.get("baseline_y")) != ("flat_webp", 1024, 420, 366):
            return False
        if not re.fullmatch(r"[0-9a-f]{64}", a.get("sha256") or ""):
            return False
        if not d.get("variant_key") or not d.get("style_version"):
            return False
        if not str(a.get("download_url", "")).startswith("https://"):
            return False
    if state == "fallback" and d.get("reason") not in REASONS:
        return False
    if state == "pending":
        ms = d.get("retry_after_ms")
        if not isinstance(ms, int) or not 2000 <= ms <= 60000:
            return False
    return True


def _all_states(world):  # noqa: F811
    world.row(ECHO, "ready", 1, SHA)
    yield run(vv.get_descriptor("u", ECHO_CAR))  # ready
    world.rows.clear()
    yield run(vv.get_descriptor("u", ECHO_CAR))  # pending, no row
    world.row(ECHO, "review")
    yield run(vv.get_descriptor("u", ECHO_CAR))  # pending, review
    world.row(ECHO, "failed")
    yield run(vv.get_descriptor("u", ECHO_CAR))  # fallback temporarily_unavailable
    yield run(vv.get_descriptor("u", {**LADA_CAR, "make": "!!!"}))  # unknown_variant
    yield run(vv.get_descriptor("u", LADA_CAR))  # choose_variant with body styles
    yield run(vv.get_descriptor("u", CIVIC_CAR))  # choose_variant


def test_every_produced_state_is_accepted(world):  # noqa: F811
    seen = set()
    for d in _all_states(world):
        dumped = d.to_dict()
        assert client_accepts(dumped), dumped
        assert None not in dumped.values()
        seen.add((d.state, d.reason))
    assert seen == {("ready", None), ("pending", None), ("fallback", "temporarily_unavailable"),
                    ("fallback", "unknown_variant"), ("fallback", "choose_variant")}


def test_hand_broken_descriptors_are_rejected(world):  # noqa: F811
    world.row(ECHO, "ready", 1, SHA)
    good = run(vv.get_descriptor("u", ECHO_CAR)).to_dict()
    assert client_accepts(good)

    def broken(mutate):
        d = copy.deepcopy(good)
        mutate(d)
        return d

    assert not client_accepts(broken(lambda d: d.update(schema_version=2)))
    assert not client_accepts(broken(lambda d: d["artwork"].update(download_url="http://x.test/a")))
    assert not client_accepts(broken(lambda d: d["artwork"].update(sha256="a" * 63)))
    assert not client_accepts(broken(lambda d: d["artwork"].update(width=1000)))
    assert not client_accepts(broken(lambda d: d.update(options=[])))
    assert not client_accepts({"schema_version": 1, "state": "pending", "retry_after_ms": 100})
