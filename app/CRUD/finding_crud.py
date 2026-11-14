from app.db import pool

# Every statement names user_id (and car_id where the caller has one): there is no read or write by id alone.
_COLUMNS = (
    "id, car_id, kind, text, norm, status, source, evidence_quote, session_id, created_at, updated_at, superseded_by"
)
_DEDUPE = "(user_id, car_id, kind, norm) WHERE status = 'active'"


class FindingCRUD:
    async def in_garage(self, user_id: str, car_id: str) -> bool:
        """Is the car in the owner's garage? A turn reads and writes findings only for a car that is."""
        return await pool.fetchval(
            "SELECT $2::text = ANY(cars) FROM user_profiles WHERE user_id = $1::uuid", user_id, car_id
        ) is True

    async def for_car(self, user_id: str, car_id: str) -> list[dict]:
        """Every row of this owner's car, any status (the write rules need the retracted and superseded too)."""
        return await pool.fetch(
            f"SELECT {_COLUMNS} FROM car_findings WHERE user_id = $1 AND car_id = $2", user_id, car_id
        )

    async def history(self, user_id: str, car_id: str, exclude_session: str | None) -> list[dict]:
        """The active rows of earlier sessions: what the prompt's Car history is built from."""
        return await pool.fetch(
            f"SELECT {_COLUMNS} FROM car_findings WHERE user_id = $1 AND car_id = $2 AND status = 'active'"
            " AND session_id IS DISTINCT FROM $3::uuid",
            user_id, car_id, exclude_session,
        )

    async def listing(self, user_id: str, car_id: str, status: str | None) -> list[dict]:
        return await pool.fetch(
            f"SELECT {_COLUMNS} FROM car_findings WHERE user_id = $1 AND car_id = $2"
            " AND ($3::text IS NULL OR status = $3) ORDER BY created_at DESC, id",
            user_id, car_id, status,
        )

    async def get(self, user_id: str, car_id: str, finding_id: str) -> dict | None:
        return await pool.fetchrow(
            f"SELECT {_COLUMNS} FROM car_findings WHERE user_id = $1 AND car_id = $2 AND id = $3::uuid",
            user_id, car_id, finding_id,
        )

    async def apply(self, user_id: str, car_id: str, session_id: str | None, ops: list[tuple]) -> None:
        """A write plan (findings.plan_writes) in one transaction; inserts come first, so a supersede link
        always points at a row that exists. A concurrent identical insert is skipped by the unique index."""
        async with pool.transaction() as conn:
            for op in sorted(ops, key=lambda o: o[0] != "insert"):
                match op:
                    case ("insert", row):
                        await conn.execute(
                            "INSERT INTO car_findings (id, user_id, car_id, kind, text, norm, source, evidence_quote,"
                            " session_id, created_at, updated_at)"
                            " VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8, $9::uuid, $10, $10)"
                            f" ON CONFLICT {_DEDUPE} DO NOTHING",
                            row["id"], user_id, car_id, row["kind"], row["text"], row["norm"], row["source"],
                            row["evidence_quote"], session_id, row["created_at"],
                        )
                    case ("promote", row_id, quote):
                        await conn.execute(
                            "UPDATE car_findings SET source = 'user_confirmed', evidence_quote = $4, session_id = $5::uuid,"
                            " updated_at = now() WHERE id = $1::uuid AND user_id = $2::uuid AND car_id = $3 AND status = 'active'",
                            row_id, user_id, car_id, quote, session_id,
                        )
                    case ("supersede", old_id, new_id):
                        await conn.execute(
                            "UPDATE car_findings SET status = 'superseded', superseded_by = $4::uuid, updated_at = now()"
                            " WHERE id = $1::uuid AND user_id = $2::uuid AND car_id = $3 AND status = 'active'",
                            old_id, user_id, car_id, new_id,
                        )
                    case ("delete", row_id):
                        await conn.execute(
                            "DELETE FROM car_findings WHERE id = $1::uuid AND user_id = $2::uuid AND car_id = $3",
                            row_id, user_id, car_id,
                        )

    async def edit(self, user_id: str, car_id: str, finding_id: str, text: str | None, norm: str | None, quote: str) -> dict | None:
        """The owner's own edit or confirmation: the row becomes user_confirmed. None when it is not theirs, or
        the new text collides with another active row of the same kind (the caller maps that to 409)."""
        return await pool.fetchrow(
            "UPDATE car_findings SET text = COALESCE($4, text), norm = COALESCE($5, norm), source = 'user_confirmed',"
            " evidence_quote = $6, updated_at = now()"
            f" WHERE user_id = $1::uuid AND car_id = $2 AND id = $3::uuid AND status = 'active' RETURNING {_COLUMNS}",
            user_id, car_id, finding_id, text, norm, quote,
        )

    async def retract(self, user_id: str, car_id: str, finding_id: str) -> dict | None:
        return await pool.fetchrow(
            "UPDATE car_findings SET status = 'retracted', updated_at = now()"
            f" WHERE user_id = $1::uuid AND car_id = $2 AND id = $3::uuid AND status = 'active' RETURNING {_COLUMNS}",
            user_id, car_id, finding_id,
        )

    async def resolve(self, user_id: str, car_id: str, finding_id: str, new_row: dict) -> dict | None:
        """Add the owner's repair row and supersede the finding it fixed, together. None when the finding is not
        an active row of theirs."""
        async with pool.transaction() as conn:
            old = await conn.fetchrow(
                "SELECT id, session_id FROM car_findings WHERE user_id = $1::uuid AND car_id = $2 AND id = $3::uuid"
                " AND status = 'active' FOR UPDATE",
                user_id, car_id, finding_id,
            )
            if old is None:
                return None
            created = await conn.fetchrow(
                "INSERT INTO car_findings (id, user_id, car_id, kind, text, norm, source, evidence_quote)"
                " VALUES ($1::uuid, $2::uuid, $3, 'repair_done', $4, $5, 'user_confirmed', $6)"
                f" ON CONFLICT {_DEDUPE} DO UPDATE SET updated_at = now() RETURNING {_COLUMNS}",
                new_row["id"], user_id, car_id, new_row["text"], new_row["norm"], new_row["evidence_quote"],
            )
            await conn.execute(
                "UPDATE car_findings SET status = 'superseded', superseded_by = $2::uuid, updated_at = now()"
                " WHERE id = $1::uuid AND id <> $2::uuid AND user_id = $3::uuid AND car_id = $4",
                finding_id, created["id"], user_id, car_id,
            )
            return created

    async def delete_one(self, user_id: str, car_id: str, finding_id: str) -> bool:
        return await pool.fetchval(
            "DELETE FROM car_findings WHERE user_id = $1::uuid AND car_id = $2 AND id = $3::uuid RETURNING true",
            user_id, car_id, finding_id,
        ) is True

    async def delete_all(self, user_id: str, car_id: str) -> int:
        status = await pool.execute("DELETE FROM car_findings WHERE user_id = $1::uuid AND car_id = $2", user_id, car_id)
        return int(status.rsplit(" ", 1)[-1])
