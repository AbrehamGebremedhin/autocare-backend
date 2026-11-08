import pytest

from app.services import vehicle_catalog as vc

TWO_VARIANT_CATALOG = {
    "catalog_version": "test",
    "variants": [
        {"key": "acme-roadster-a", "label": "Acme Roadster A", "make": "acme",
         "models": ["roadster"], "years": [2000, 2010], "ambiguous": False},
        {"key": "acme-roadster-b", "label": "Acme Roadster B", "make": "acme",
         "models": ["roadster"], "years": [2005, 2015], "ambiguous": False},
        {"key": "acme-solo-x", "label": "Acme Solo X", "make": "acme",
         "models": ["solo"], "years": [2000, 2010], "ambiguous": True},
    ],
}


@pytest.fixture
def two_variants(monkeypatch):
    monkeypatch.setattr(vc, "_load", lambda: TWO_VARIANT_CATALOG)


def test_normalize():
    assert vc.normalize("  Toyota  ECHO! ") == "toyota echo"
    assert vc.normalize("Citroën") == "citroen"
    assert vc.normalize("") == ""
    assert vc.normalize(None) == ""


def test_echo_2001_selected():
    r = vc.resolve("Toyota", "Echo", 2001)
    assert (r.kind, r.variant_key, r.options) == ("selected", "toyota-echo-2001-sedan", [])


def test_alias_selected():
    assert vc.resolve("toyota", "Yaris Sedan", 2003).variant_key == "toyota-echo-2001-sedan"


def test_civic_demo_selected():
    assert vc.resolve("Honda", "Civic", 2008).variant_key == "honda-civic-2008-sedan"


@pytest.mark.parametrize("args", [("Toyota", "Echo", 1999), ("Nope", "Echo", 2001), ("Toyota", "Camry", 2001)])
def test_unknown(args):
    r = vc.resolve(*args)
    assert (r.kind, r.variant_key, r.options) == ("unknown", None, [])


def test_two_variants_choose_labels_only(two_variants):
    r = vc.resolve("Acme", "Roadster", 2007)
    assert r.kind == "choose" and r.variant_key is None
    assert r.options == [("acme-roadster-a", "Acme Roadster A"), ("acme-roadster-b", "Acme Roadster B")]
    assert vc.resolve("Acme", "Roadster", 2001).variant_key == "acme-roadster-a"


def test_valid_choice_selected(two_variants):
    r = vc.resolve("Acme", "Roadster", 2007, choice="acme-roadster-b")
    assert (r.kind, r.variant_key) == ("selected", "acme-roadster-b")


def test_choice_for_other_model_ignored(two_variants):
    r = vc.resolve("Acme", "Roadster", 2007, choice="acme-solo-x")
    assert r.kind == "choose"
    assert not vc.variant_matches_car("acme-solo-x", "Acme", "Roadster")
    assert vc.variant_matches_car("acme-solo-x", "ACME", "Solo")


def test_single_ambiguous_candidate_chooses(two_variants):
    r = vc.resolve("Acme", "Solo", 2005)
    assert r.kind == "choose" and r.options == [("acme-solo-x", "Acme Solo X")]


def test_is_allowed():
    assert vc.is_allowed("toyota-echo-2001-sedan")
    for bad in ["../x", "", "nope", None, 5, b"x"]:
        assert vc.is_allowed(bad) is False


def test_label():
    assert vc.label("toyota-echo-2001-sedan") == "Toyota Echo sedan (2000-2005)"
