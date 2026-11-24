# ============================================================
# DERMAI - DOCTOR / CLINICIAN ROUTES
# ============================================================

from datetime import datetime, timezone

from bson import ObjectId

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from pydantic import BaseModel, Field

from app.database import (
    users_collection,
    cases_collection,
    images_collection,
    analyses_collection,
)

from app.dependencies.auth import (
    get_current_user,
)


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/doctor",
    tags=["Doctor / Clinician"]
)


# ============================================================
# DOCTOR ROLE CHECK
# ============================================================

def require_doctor(
    current_user=Depends(get_current_user)
):
    if current_user.get("role") != "doctor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Doctor access required."
        )

    return current_user


# ============================================================
# CLINICAL NOTE SCHEMA
# ============================================================

class ClinicalNoteUpdate(BaseModel):
    clinical_note: str = Field(
        default="",
        max_length=5000
    )


# ============================================================
# DATETIME SERIALIZER
# ============================================================

def serialize_datetime(value):
    if isinstance(value, datetime):
        return value.isoformat()

    return value


# ============================================================
# HELPER - GET AI STATUS
# ============================================================

def get_ai_status(latest_analysis):
    """
    Convert analysis status into the two statuses
    required by the Doctor Dashboard:

        completed
        pending
    """

    if not latest_analysis:
        return "pending"

    analysis_status = str(
        latest_analysis.get(
            "analysis_status",
            "pending"
        )
    ).lower().strip()

    if analysis_status in {
        "completed",
        "complete",
        "success",
        "successful"
    }:
        return "completed"

    return "pending"


# ============================================================
# GET ALL CASES FOR DOCTOR
# ============================================================

@router.get("/cases")
def get_doctor_cases(
    current_user=Depends(require_doctor)
):

    cases = cases_collection.find(
        {}
    ).sort(
        "created_at",
        -1
    )

    results = []

    for case in cases:

        # ----------------------------------------------------
        # Patient
        # ----------------------------------------------------

        patient = users_collection.find_one(
            {
                "_id": case["user_id"]
            }
        )

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        images = list(
            images_collection.find(
                {
                    "case_id": case["_id"]
                }
            ).sort(
                "uploaded_at",
                -1
            )
        )

        # ----------------------------------------------------
        # Latest Analysis
        # ----------------------------------------------------

        latest_analysis = None

        if images:

            image_ids = [
                image["_id"]
                for image in images
            ]

            latest_analysis = (
                analyses_collection.find_one(
                    {
                        "image_id": {
                            "$in": image_ids
                        }
                    },
                    sort=[
                        (
                            "analysis_completed_at",
                            -1
                        ),
                        (
                            "analysis_started_at",
                            -1
                        ),
                        (
                            "started_at",
                            -1
                        )
                    ]
                )
            )

        # ----------------------------------------------------
        # AI STATUS
        # ----------------------------------------------------

        ai_status = get_ai_status(
            latest_analysis
        )

        # ----------------------------------------------------
        # Build Response
        # ----------------------------------------------------

        results.append({

            "id": str(
                case["_id"]
            ),

            "case_id": str(
                case["_id"]
            ),

            "title": case.get(
                "title",
                "Untitled Assessment"
            ),

            "description": case.get(
                "description"
            ),

            # AI status
            "status": ai_status,

            "patient_name": (
                patient.get(
                    "full_name",
                    "Unknown Patient"
                )
                if patient
                else "Unknown Patient"
            ),

            "patient_email": (
                patient.get(
                    "email",
                    ""
                )
                if patient
                else ""
            ),

            "patient_id": (
                str(
                    patient["_id"]
                )
                if patient
                else None
            ),

            "created_at": serialize_datetime(
                case.get("created_at")
            ),

            "updated_at": serialize_datetime(
                case.get("updated_at")
            ),

            # Doctor clinical note
            "clinical_note": case.get(
                "clinical_note",
                case.get(
                    "clinical_notes",
                    ""
                )
            ),

            "follow_up_status": case.get(
                "follow_up_status",
                "not_scheduled"
            ),

            "image_count": len(images),

            "analysis_status": ai_status,

            "analysis": (
                {
                    "id": str(
                        latest_analysis["_id"]
                    ),

                    "status": ai_status,

                    "predicted_condition":
                        latest_analysis.get(
                            "predicted_condition"
                        ),

                    "confidence":
                        latest_analysis.get(
                            "confidence"
                        ),

                    "top_predictions":
                        latest_analysis.get(
                            "top_predictions",
                            []
                        ),

                    "model":
                        latest_analysis.get(
                            "model"
                        ),

                    "device":
                        latest_analysis.get(
                            "device"
                        ),

                    "completed_at":
                        serialize_datetime(
                            latest_analysis.get(
                                "analysis_completed_at"
                            )
                        ),

                    "medical_disclaimer":
                        latest_analysis.get(
                            "medical_disclaimer"
                        )
                }
                if latest_analysis
                else None
            )
        })

    return results


# ============================================================
# GET SINGLE DOCTOR CASE
# ============================================================

@router.get("/cases/{case_id}")
def get_doctor_case(
    case_id: str,
    current_user=Depends(require_doctor)
):

    # --------------------------------------------------------
    # Validate Case ID
    # --------------------------------------------------------

    try:
        case_object_id = ObjectId(
            case_id
        )

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid case ID."
        )

    # --------------------------------------------------------
    # Find Case
    # --------------------------------------------------------

    case = cases_collection.find_one(
        {
            "_id": case_object_id
        }
    )

    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found."
        )

    # --------------------------------------------------------
    # Patient
    # --------------------------------------------------------

    patient = users_collection.find_one(
        {
            "_id": case["user_id"]
        }
    )

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    images = list(
        images_collection.find(
            {
                "case_id": case["_id"]
            }
        ).sort(
            "uploaded_at",
            -1
        )
    )

    image_results = []

    for image in images:

        # ----------------------------------------------------
        # Latest analysis for image
        # ----------------------------------------------------

        analysis = analyses_collection.find_one(
            {
                "image_id": image["_id"]
            },
            sort=[
                (
                    "analysis_completed_at",
                    -1
                ),
                (
                    "analysis_started_at",
                    -1
                ),
                (
                    "started_at",
                    -1
                )
            ]
        )

        ai_status = get_ai_status(
            analysis
        )

        image_results.append({

            "id": str(
                image["_id"]
            ),

            "filename": image.get(
                "original_filename"
            ),

            "mime_type": image.get(
                "mime_type"
            ),

            "width": image.get(
                "width"
            ),

            "height": image.get(
                "height"
            ),

            "quality_status": image.get(
                "quality_status",
                "pending"
            ),

            "uploaded_at":
                serialize_datetime(
                    image.get(
                        "uploaded_at"
                    )
                ),

            "analysis": (
                {
                    "id": str(
                        analysis["_id"]
                    ),

                    "status": ai_status,

                    "predicted_condition":
                        analysis.get(
                            "predicted_condition"
                        ),

                    "confidence":
                        analysis.get(
                            "confidence"
                        ),

                    "top_predictions":
                        analysis.get(
                            "top_predictions",
                            []
                        ),

                    "model":
                        analysis.get(
                            "model"
                        ),

                    "completed_at":
                        serialize_datetime(
                            analysis.get(
                                "analysis_completed_at"
                            )
                        )
                }
                if analysis
                else None
            )
        })

    # --------------------------------------------------------
# GET LATEST COMPLETED ANALYSIS FOR THE CASE
# --------------------------------------------------------

    latest_analysis = None

    for image in images:

      current_analysis = analyses_collection.find_one(
        {
            "image_id": image["_id"],
            "analysis_status": "completed"
        },
        sort=[
            ("analysis_completed_at", -1),
            ("analysis_started_at", -1),
            ("started_at", -1)
        ]
    )

    if current_analysis:
        if (
            latest_analysis is None
            or current_analysis.get("analysis_completed_at")
            > latest_analysis.get("analysis_completed_at")
        ):
            latest_analysis = current_analysis
    
    # --------------------------------------------------------
    # Determine overall AI status
    # --------------------------------------------------------

    overall_status = "pending"

    for image in image_results:

        image_analysis = image.get(
            "analysis"
        )

        if (
            image_analysis
            and image_analysis.get("status")
            == "completed"
        ):
            overall_status = "completed"
            break

    # --------------------------------------------------------
    # Return Case Details
    # --------------------------------------------------------

    return {

        "id": str(
            case["_id"]
        ),

        "title": case.get(
            "title"
        ),

        "description": case.get(
            "description"
        ),

        # AI status
        "status": overall_status,

        # Patient
        "patient": {

            "id": (
                str(
                    patient["_id"]
                )
                if patient
                else None
            ),

            "full_name": (
                patient.get(
                    "full_name"
                )
                if patient
                else "Unknown Patient"
            ),

            "email": (
                patient.get(
                    "email"
                )
                if patient
                else ""
            )
        },

        # Doctor clinical note
        "clinical_note": case.get(
            "clinical_note",
            case.get(
                "clinical_notes",
                ""
            )
        ),

        "follow_up_status": case.get(
            "follow_up_status",
            "not_scheduled"
        ),

        "created_at":
            serialize_datetime(
                case.get(
                    "created_at"
                )
            ),

        "updated_at":
            serialize_datetime(
                case.get(
                    "updated_at"
                )
            ),

        "analysis": (
    {
        "id": str(latest_analysis["_id"]),
        "status": "completed",

        "predicted_condition":
            latest_analysis.get("predicted_condition"),

        "confidence":
            latest_analysis.get("confidence"),

        "risk_assessment":
            latest_analysis.get("risk_assessment", {}),

        "risk_category":
            latest_analysis.get("risk_category"),

        "review_priority":
            latest_analysis.get("review_priority"),

        "assessment_type":
            latest_analysis.get("assessment_type"),

        "risk_message":
            latest_analysis.get("risk_message"),

        "top_predictions":
            latest_analysis.get("top_predictions", []),

        "model":
            latest_analysis.get("model"),

        "medical_disclaimer":
            latest_analysis.get("medical_disclaimer"),

        "completed_at":
            serialize_datetime(
                latest_analysis.get("analysis_completed_at")
            )
    }
    if latest_analysis
    else None
),

        "images": image_results
    }


# ============================================================
# SAVE / UPDATE CLINICAL NOTE
# ============================================================

@router.put("/cases/{case_id}/notes")
def update_clinical_note(
    case_id: str,
    note_data: ClinicalNoteUpdate,
    current_user=Depends(require_doctor)
):

    # --------------------------------------------------------
    # Validate Case ID
    # --------------------------------------------------------

    try:
        case_object_id = ObjectId(
            case_id
        )

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid case ID."
        )

    # --------------------------------------------------------
    # Check Case
    # --------------------------------------------------------

    case = cases_collection.find_one(
        {
            "_id": case_object_id
        }
    )

    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found."
        )

    # --------------------------------------------------------
    # Prepare Clinical Note
    # --------------------------------------------------------

    clinical_note = (
        note_data.clinical_note
        or ""
    ).strip()

    now = datetime.now(
        timezone.utc
    )

    # --------------------------------------------------------
    # Update MongoDB
    # --------------------------------------------------------

    cases_collection.update_one(
        {
            "_id": case_object_id
        },
        {
            "$set": {

                "clinical_note":
                    clinical_note,

                "clinical_note_updated_at":
                    now,

                "clinical_note_by":
                    current_user["_id"],

                "updated_at":
                    now
            }
        }
    )

    # --------------------------------------------------------
    # Return Success
    # --------------------------------------------------------

    return {

        "message":
            "Clinical note updated successfully.",

        "case_id":
            case_id,

        "clinical_note":
            clinical_note
    }


# ============================================================
# UPDATE FOLLOW-UP STATUS
# ============================================================

@router.put("/cases/{case_id}/follow-up")
def update_follow_up(
    case_id: str,
    follow_up_status: str,
    current_user=Depends(require_doctor)
):

    allowed_statuses = {
        "not_scheduled",
        "scheduled",
        "completed",
        "cancelled"
    }

    if follow_up_status not in allowed_statuses:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid follow-up status. "
                f"Allowed values: "
                f"{sorted(allowed_statuses)}"
            )
        )

    # --------------------------------------------------------
    # Validate Case ID
    # --------------------------------------------------------

    try:
        case_object_id = ObjectId(
            case_id
        )

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid case ID."
        )

    # --------------------------------------------------------
    # Update Follow-up
    # --------------------------------------------------------

    result = cases_collection.update_one(
        {
            "_id": case_object_id
        },
        {
            "$set": {

                "follow_up_status":
                    follow_up_status,

                "updated_at":
                    datetime.now(
                        timezone.utc
                    ),

                "follow_up_updated_by":
                    current_user["_id"]
            }
        }
    )

    # --------------------------------------------------------
    # Check Case
    # --------------------------------------------------------

    if result.matched_count == 0:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found."
        )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "message":
            "Follow-up status updated.",

        "case_id":
            case_id,

        "follow_up_status":
            follow_up_status
    }