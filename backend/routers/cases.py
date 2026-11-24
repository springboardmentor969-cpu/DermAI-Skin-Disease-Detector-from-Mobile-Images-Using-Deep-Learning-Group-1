from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from bson import ObjectId

from app.database import cases_collection
from app.dependencies.auth import get_current_user
from models.case import create_case_document
from schemas.case import CaseCreate, CaseResponse


router = APIRouter(
    prefix="/cases",
    tags=["Case Management"]
)


# ============================================================
# CONVERT CASE DOCUMENT TO API RESPONSE
# ============================================================

def case_to_response(case):
    return CaseResponse(
        id=str(case["_id"]),
        title=case["title"],
        description=case.get("description"),
        status=case.get("status", "pending"),

        # Doctor's clinical note
        clinical_note=case.get(
            "clinical_note",
            case.get("clinical_notes", "")
        ),

        created_at=case["created_at"].isoformat(),
        updated_at=case["updated_at"].isoformat()
    )


# ============================================================
# GET OWNED CASE
# ============================================================

def get_owned_case(case_id: str, user_id: ObjectId):
    try:
        object_id = ObjectId(case_id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid case ID."
        )

    case = cases_collection.find_one({
        "_id": object_id,
        "user_id": user_id
    })

    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found."
        )

    return case


# ============================================================
# CREATE CASE
# ============================================================

@router.post(
    "",
    response_model=CaseResponse,
    status_code=status.HTTP_201_CREATED
)
def create_case(
    case: CaseCreate,
    current_user=Depends(get_current_user)
):
    user_id = current_user["_id"]

    case_document = create_case_document(
        user_id=user_id,
        title=case.title,
        description=case.description
    )

    result = cases_collection.insert_one(case_document)

    case_document["_id"] = result.inserted_id

    return case_to_response(case_document)


# ============================================================
# GET ALL CASES OF CURRENT USER
# ============================================================

@router.get(
    "",
    response_model=list[CaseResponse]
)
def get_my_cases(
    current_user=Depends(get_current_user)
):
    user_id = current_user["_id"]

    cases = cases_collection.find(
        {"user_id": user_id}
    ).sort("created_at", -1)

    return [
        case_to_response(case)
        for case in cases
    ]


# ============================================================
# GET ONE CASE
# ============================================================

@router.get(
    "/{case_id}",
    response_model=CaseResponse
)
def get_case(
    case_id: str,
    current_user=Depends(get_current_user)
):
    case = get_owned_case(
        case_id=case_id,
        user_id=current_user["_id"]
    )

    return case_to_response(case)


# ============================================================
# UPDATE CASE
# ============================================================

@router.put(
    "/{case_id}",
    response_model=CaseResponse
)
def update_case(
    case_id: str,
    case_data: CaseCreate,
    current_user=Depends(get_current_user)
):
    case = get_owned_case(
        case_id=case_id,
        user_id=current_user["_id"]
    )

    updated_at = datetime.now(timezone.utc)

    cases_collection.update_one(
        {"_id": case["_id"]},
        {
            "$set": {
                "title": case_data.title,
                "description": case_data.description,
                "updated_at": updated_at
            }
        }
    )

    updated_case = cases_collection.find_one(
        {"_id": case["_id"]}
    )

    return case_to_response(updated_case)


# ============================================================
# DELETE CASE
# ============================================================

@router.delete(
    "/{case_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
def delete_case(
    case_id: str,
    current_user=Depends(get_current_user)
):
    case = get_owned_case(
        case_id=case_id,
        user_id=current_user["_id"]
    )

    cases_collection.delete_one(
        {"_id": case["_id"]}
    )

    return None