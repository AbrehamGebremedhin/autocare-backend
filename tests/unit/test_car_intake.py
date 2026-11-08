"""Intake of the owner's transmission: validated at the edge, per owner, editable, and part of the diagnosis prompt."""
import pytest

from app.services.prompts import _render_car

NEW_CAR = {"make": "Toyota", "model": "Corolla", "year": 2006}


def test_add_car_passes_the_transmission_to_the_service(client, auth_headers, mocker):
    mocker.patch("app.api.v1.car_route.ingest_car_manual")
    add = mocker.patch(
        "app.api.v1.car_route.user_profile_service.add_car_to_user",
        return_value={"success": True, "car_info": {"id": "toyota-corolla-2006", **NEW_CAR, "transmission": "cvt"}},
    )
    r = client.post("/api/v1/cars", json={**NEW_CAR, "transmission": "cvt"}, headers=auth_headers)
    assert r.status_code == 200 and r.json()["transmission"] == "cvt"
    assert add.call_args.kwargs["transmission"] == "cvt"


def test_transmission_is_optional_and_nullable(client, auth_headers, mocker):
    mocker.patch("app.api.v1.car_route.ingest_car_manual")
    add = mocker.patch(
        "app.api.v1.car_route.user_profile_service.add_car_to_user",
        return_value={"success": True, "car_info": {"id": "toyota-corolla-2006", **NEW_CAR}},
    )
    r = client.post("/api/v1/cars", json=NEW_CAR, headers=auth_headers)
    assert r.status_code == 200 and r.json()["transmission"] is None
    assert add.call_args.kwargs["transmission"] is None


@pytest.mark.parametrize("bad", ["Manual", "stick", "", "manual; drop table", 5])
def test_unknown_transmission_is_422(client, auth_headers, bad):
    assert client.post("/api/v1/cars", json={**NEW_CAR, "transmission": bad}, headers=auth_headers).status_code == 422
    assert client.patch("/api/v1/cars/c1", json={"transmission": bad}, headers=auth_headers).status_code == 422


def test_patch_sets_it_for_a_car_in_the_garage(client, auth_headers, mocker):
    setter = mocker.patch("app.api.v1.car_route.user_profile_service.set_transmission", return_value=True)
    r = client.patch("/api/v1/cars/c1", json={"transmission": "manual"}, headers=auth_headers)
    assert r.status_code == 200 and r.json() == {"id": "c1", "transmission": "manual"}
    assert setter.call_args.args[1:] == ("c1", "manual")


def test_patch_on_a_car_that_is_not_theirs_is_404(client, auth_headers, mocker):
    mocker.patch("app.api.v1.car_route.user_profile_service.set_transmission", return_value=False)
    assert client.patch("/api/v1/cars/c1", json={"transmission": "manual"}, headers=auth_headers).status_code == 404


def test_patch_needs_a_token(client):
    assert client.patch("/api/v1/cars/c1", json={"transmission": "manual"}).status_code == 401


@pytest.mark.parametrize(
    "transmission, tail",
    [("manual", ", manual transmission"), ("cvt", ", CVT transmission"), ("dual_clutch", ", dual-clutch transmission"),
     (None, ""), ("junk", "")],
)
def test_prompt_names_the_transmission_only_for_known_values(transmission, tail):
    car = {"make": "Toyota", "model": "Corolla", "year": 2006, "transmission": transmission}
    assert _render_car(car) == "2006 Toyota Corolla" + tail
