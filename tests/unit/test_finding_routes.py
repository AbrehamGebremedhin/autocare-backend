"""/cars/{car_id}/findings: the owner lists, corrects, retracts, resolves and deletes. Synthetic data only."""
from datetime import datetime, timezone
from uuid import uuid4

import asyncpg
import pytest

from app.services import findings

CAR = "toyota-echo-2001"


class FakeFindingStore:
    """FindingCRUD over a list; like the SQL, every method names the owner and the car."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, user_id, kind, text, source="model_suggested", status="active", car_id=CAR):
        row = {"id": str(uuid4()), "user_id": user_id, "car_id": car_id, "kind": kind, "text": text,
               "norm": findings._norm(text), "status": status, "source": source,
               "evidence_quote": "q" if source == "user_confirmed" else None,
               "session_id": None, "created_at": datetime(2026, 11, 2, tzinfo=timezone.utc),
               "updated_at": datetime(2026, 11, 2, tzinfo=timezone.utc), "superseded_by": None}
        self.rows.append(row)
        return row

    def _mine(self, user_id, car_id, finding_id=None):
        return [r for r in self.rows if r["user_id"] == user_id and r["car_id"] == car_id
                and (finding_id is None or r["id"] == finding_id)]

    @staticmethod
    def _out(row):
        return None if row is None else {k: v for k, v in row.items() if k != "user_id"}

    async def for_car(self, user_id, car_id):
        return [dict(r) for r in self._mine(user_id, car_id)]

    async def history(self, user_id, car_id, exclude_session):
        return [r for r in await self.for_car(user_id, car_id) if r["status"] == "active"]

    async def apply(self, user_id, car_id, session_id, ops):
        for op in ops:
            if op[0] == "insert":
                self.add(user_id, op[1]["kind"], op[1]["text"], op[1]["source"], car_id=car_id)

    async def listing(self, user_id, car_id, status):
        return [self._out(r) for r in self._mine(user_id, car_id) if status in (None, r["status"])]

    async def edit(self, user_id, car_id, finding_id, text, norm, quote):
        for r in self._mine(user_id, car_id, finding_id):
            if r["status"] == "active":
                clash = any(o is not r and o["status"] == "active" and o["kind"] == r["kind"] and o["norm"] == norm
                            for o in self._mine(user_id, car_id))
                if norm and clash:
                    raise asyncpg.UniqueViolationError("dup")
                r.update(text=text or r["text"], norm=norm or r["norm"], source="user_confirmed", evidence_quote=quote)
                return self._out(r)

    async def retract(self, user_id, car_id, finding_id):
        for r in self._mine(user_id, car_id, finding_id):
            if r["status"] == "active":
                r["status"] = "retracted"
                return self._out(r)

    async def resolve(self, user_id, car_id, finding_id, new_row):
        for r in self._mine(user_id, car_id, finding_id):
            if r["status"] == "active":
                new = self.add(user_id, "repair_done", new_row["text"], "user_confirmed")
                new["evidence_quote"] = new_row["evidence_quote"]
                r.update(status="superseded", superseded_by=new["id"])
                return self._out(new)

    async def delete_one(self, user_id, car_id, finding_id):
        found = self._mine(user_id, car_id, finding_id)
        self.rows = [r for r in self.rows if r not in found]
        return bool(found)

    async def delete_all(self, user_id, car_id):
        gone = self._mine(user_id, car_id)
        self.rows = [r for r in self.rows if r not in gone]
        return len(gone)


@pytest.fixture
def store(monkeypatch):
    fake = FakeFindingStore()
    monkeypatch.setattr(findings, "crud", fake)
    return fake


def url(finding_id="", tail=""):
    return f"/api/v1/cars/{CAR}/findings" + (f"/{finding_id}" if finding_id else "") + tail


def test_list_shows_everything_of_mine_and_nothing_internal(client, auth_headers, store, user_id, other_user_id):
    mine = store.add(user_id, "confirmed_finding", "rear-left bearing worn", "user_confirmed")
    store.add(user_id, "suspected_cause", "tone ring", status="retracted")
    store.add(other_user_id, "confirmed_finding", "the other owner's note", "user_confirmed")
    r = client.get(url(), headers=auth_headers)
    assert r.status_code == 200 and [f["text"] for f in r.json()] == ["rear-left bearing worn", "tone ring"]
    assert all({"user_id", "norm"}.isdisjoint(f) for f in r.json()) and r.json()[0]["id"] == mine["id"]
    assert [f["status"] for f in client.get(url() + "?status=retracted", headers=auth_headers).json()] == ["retracted"]
    assert client.get(url() + "?status=bogus", headers=auth_headers).status_code == 422


def test_another_users_car_and_findings_are_404(client, auth_headers_other_user, store, user_id):
    row = store.add(user_id, "confirmed_finding", "mine", "user_confirmed")
    assert client.get(url(), headers=auth_headers_other_user).json() == []  # no rows of theirs for this car
    for call in (lambda: client.patch(url(row["id"]), json={"confirm": True}, headers=auth_headers_other_user),
                 lambda: client.post(url(row["id"], "/retract"), headers=auth_headers_other_user),
                 lambda: client.delete(url(row["id"]), headers=auth_headers_other_user)):
        assert call().status_code == 404
    assert client.delete(url(), headers=auth_headers_other_user).json() == {"deleted": 0}  # nothing of theirs to delete
    assert [r["text"] for r in store.rows] == ["mine"]


def test_unknown_finding_is_404_and_a_bad_id_422(client, auth_headers, store):
    assert client.post(url(uuid4(), "/retract"), headers=auth_headers).status_code == 404
    assert client.delete(url(uuid4()), headers=auth_headers).status_code == 404
    assert client.delete(url("nope"), headers=auth_headers).status_code == 422


def test_confirm_promotes_a_suggestion_to_user_confirmed(client, auth_headers, store, user_id):
    row = store.add(user_id, "suspected_cause", "Worn wheel bearing")
    r = client.patch(url(row["id"]), json={"confirm": True}, headers=auth_headers)
    assert r.status_code == 200 and r.json()["source"] == "user_confirmed"
    assert r.json()["evidence_quote"] == findings.APP_CONFIRMED and r.json()["text"] == "Worn wheel bearing"


def test_editing_the_text_stores_the_owners_words_as_the_quote(client, auth_headers, store, user_id):
    row = store.add(user_id, "suspected_cause", "Worn wheel bearing")
    r = client.patch(url(row["id"]), json={"text": "  rear-left bearing worn "}, headers=auth_headers)
    assert r.json()["text"] == "rear-left bearing worn" and r.json()["evidence_quote"] == "rear-left bearing worn"
    assert r.json()["source"] == "user_confirmed"


def test_edit_needs_text_or_confirm_and_a_collision_is_409(client, auth_headers, store, user_id):
    a = store.add(user_id, "suspected_cause", "a")
    store.add(user_id, "suspected_cause", "b")
    assert client.patch(url(a["id"]), json={}, headers=auth_headers).status_code == 422
    assert client.patch(url(a["id"]), json={"text": "x" * 301}, headers=auth_headers).status_code == 422
    assert client.patch(url(a["id"]), json={"text": "b"}, headers=auth_headers).status_code == 409


def test_retract_keeps_the_row_but_marks_it(client, auth_headers, store, user_id):
    row = store.add(user_id, "suspected_cause", "tone ring")
    assert client.post(url(row["id"], "/retract"), headers=auth_headers).json()["status"] == "retracted"
    assert store.rows[0]["status"] == "retracted"
    assert client.post(url(row["id"], "/retract"), headers=auth_headers).status_code == 404  # already withdrawn


def test_resolve_adds_a_repair_and_supersedes_the_finding(client, auth_headers, store, user_id):
    row = store.add(user_id, "confirmed_finding", "rear-left bearing worn", "user_confirmed")
    r = client.post(url(row["id"], "/resolve"), json={"what_fixed_it": "replaced the rear-left hub"}, headers=auth_headers)
    assert r.status_code == 200 and r.json()["kind"] == "repair_done" and r.json()["source"] == "user_confirmed"
    assert store.rows[0]["status"] == "superseded" and store.rows[0]["superseded_by"] == r.json()["id"]
    assert client.post(url(row["id"], "/resolve"), json={"what_fixed_it": "x"}, headers=auth_headers).status_code == 404
    assert client.post(url(row["id"], "/resolve"), json={}, headers=auth_headers).status_code == 422


def test_delete_one_and_delete_all(client, auth_headers, store, user_id, other_user_id):
    a = store.add(user_id, "suspected_cause", "a")
    store.add(user_id, "suspected_cause", "b")
    store.add(other_user_id, "suspected_cause", "theirs")
    assert client.delete(url(a["id"]), headers=auth_headers).json() == {"deleted": a["id"]}
    assert client.delete(url(), headers=auth_headers).json() == {"deleted": 1}
    assert [r["text"] for r in store.rows] == ["theirs"]


def test_every_route_needs_a_login(client, store):
    assert client.get(url()).status_code in (401, 403)
    assert client.delete(url()).status_code in (401, 403)


def test_findings_stay_visible_and_deletable_after_the_car_leaves_the_garage(client, auth_headers, store, user_id, fake_pool):
    import asyncio

    from app.schemas.session import DiagnosisState, SuspectedCause
    from app.services.user_profile_service import UserProfileService

    fake_pool.status = "UPDATE 1"
    assert asyncio.run(UserProfileService().remove_car_from_user(user_id, CAR))["success"] is True
    # a write that gets through (record_turn alone does not check the garage; the turn does, see test_car_history)
    state = DiagnosisState(suspected_causes=[SuspectedCause(cause="ABS tone ring damaged")])
    asyncio.run(findings.record_turn(user_id, CAR, str(uuid4()), [], "", state))
    listed = client.get(url(), headers=auth_headers)
    assert listed.status_code == 200 and [f["text"] for f in listed.json()] == ["ABS tone ring damaged"]
    one = listed.json()[0]["id"]
    assert client.post(url(one, "/retract"), headers=auth_headers).status_code == 200
    assert client.delete(url(one), headers=auth_headers).json() == {"deleted": one}
    assert client.delete(url(), headers=auth_headers).json() == {"deleted": 0}
    assert client.get(url(), headers=auth_headers).json() == []
