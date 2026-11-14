from typing import Annotated, Literal
from uuid import UUID, uuid4

import asyncpg
from fastapi import APIRouter, Depends, Path
from pydantic import BaseModel, Field, model_validator

from app.schemas.Car import MAX_CAR_ID_CHARS, NO_NUL
from app.services import findings
from app.utils.auth_middleware import get_current_user
from app.utils.exceptions import RecordNotFoundException, ResourceConflictException

# Scoped by the caller and the car id in every query, never by the garage: what is stored about a car stays
# visible and deletable after the car leaves the garage. A car that is not theirs just has no rows (empty list, 404).
router = APIRouter(prefix="/cars/{car_id}/findings", tags=["Findings"])

CarId = Annotated[str, Path(max_length=MAX_CAR_ID_CHARS, pattern=NO_NUL)]
_TEXT = {"min_length": 1, "max_length": findings.TEXT_CHARS, "pattern": NO_NUL}


class FindingEdit(BaseModel):
    text: str | None = Field(None, **_TEXT)
    confirm: bool = False

    @model_validator(mode="after")
    def _says_something(self):
        if self.text is None and not self.confirm:
            raise ValueError("send new text, or confirm: true")
        return self


class Resolution(BaseModel):
    what_fixed_it: str = Field(**_TEXT)


def _public(row: dict | None) -> dict:
    if row is None:
        raise RecordNotFoundException("Finding")
    return {k: v for k, v in row.items() if k != "norm"}


@router.get("", summary="Everything stored about one of the user's cars")
async def list_findings(
    car_id: CarId, status: Literal["active", "superseded", "retracted"] | None = None, user: dict = Depends(get_current_user)
):
    return [_public(r) for r in await findings.crud.listing(user["id"], car_id, status)]


@router.patch("/{finding_id}", summary="Correct a finding, or confirm a suggestion; the owner's word makes it user_confirmed")
async def edit_finding(car_id: CarId, finding_id: UUID, body: FindingEdit, user: dict = Depends(get_current_user)):
    user_id = user["id"]
    text = findings.clean(body.text) if body.text else None
    try:
        row = await findings.crud.edit(
            user_id, car_id, str(finding_id), text, findings._norm(text) if text else None, text or findings.APP_CONFIRMED
        )
    except asyncpg.UniqueViolationError:
        raise ResourceConflictException("The car already has that finding.") from None
    return _public(row)


@router.post("/{finding_id}/retract", summary="Withdraw a finding: kept in the list, never sent to the model again")
async def retract_finding(car_id: CarId, finding_id: UUID, user: dict = Depends(get_current_user)):
    return _public(await findings.crud.retract(user["id"], car_id, str(finding_id)))


@router.post("/{finding_id}/resolve", summary="Record what fixed a problem: adds a repair and supersedes the finding")
async def resolve_finding(car_id: CarId, finding_id: UUID, body: Resolution, user: dict = Depends(get_current_user)):
    text = findings.clean(body.what_fixed_it)
    new_row = {"id": str(uuid4()), "text": text, "norm": findings._norm(text), "evidence_quote": text}
    return _public(await findings.crud.resolve(user["id"], car_id, str(finding_id), new_row))


@router.delete("/{finding_id}", summary="Delete one finding for good")
async def delete_finding(car_id: CarId, finding_id: UUID, user: dict = Depends(get_current_user)):
    if not await findings.crud.delete_one(user["id"], car_id, str(finding_id)):
        raise RecordNotFoundException("Finding")
    return {"deleted": str(finding_id)}


@router.delete("", summary="Delete everything stored about this car")
async def delete_all_findings(car_id: CarId, user: dict = Depends(get_current_user)):
    return {"deleted": await findings.crud.delete_all(user["id"], car_id)}
