"""user_profiles queries run in a worker thread, so a slow Supabase call does not stall other requests or SSE streams."""
import threading
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

from app.services.user_profile_service import UserProfileService

USER = "11111111-1111-4111-8111-111111111111"


def _service(rows: list) -> tuple[UserProfileService, list]:
    threads: list = []

    def execute():
        threads.append(threading.current_thread())
        return MagicMock(data=rows)

    db = MagicMock()
    for chain in (
        db.table.return_value.select.return_value.eq.return_value,
        db.table.return_value.insert.return_value,
        db.table.return_value.update.return_value.eq.return_value,
    ):
        chain.execute.side_effect = execute

    @asynccontextmanager
    async def get_connection():
        yield db

    service = UserProfileService(car_crud=MagicMock())
    service.db_handler = MagicMock(get_connection=get_connection)
    return service, threads


async def test_profile_read_runs_off_the_event_loop():
    service, threads = _service([{"user_id": USER, "cars": ["c1"]}])
    assert (await service._get_user_profile(USER))["cars"] == ["c1"]
    assert threads and all(t is not threading.main_thread() for t in threads)


async def test_profile_create_and_update_run_off_the_event_loop():
    service, threads = _service([])
    await service._get_or_create_user_profile(USER)  # select (empty) then insert
    assert await service._update_user_profile(USER, {"cars": []}) is True
    assert len(threads) == 3
    assert all(t is not threading.main_thread() for t in threads)
