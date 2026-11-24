from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Body
from bson import ObjectId

from app.database import (
    users_collection,
    cases_collection,
    images_collection,
    analyses_collection,
    sessions_collection,
)
from app.dependencies.auth import get_current_user
from passlib.context import CryptContext


router = APIRouter(
    prefix="/api/admin",
    tags=["Admin"]
)


# ============================================================
# ADMIN ROLE CHECK
# ============================================================

def require_admin(
    current_user=Depends(get_current_user)
):
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required."
        )

    return current_user


# ============================================================
# SERIALIZE DATETIME
# ============================================================

def serialize_datetime(value):
    if isinstance(value, datetime):
        return value.isoformat()

    return value


# ============================================================
# ADMIN DASHBOARD
# ============================================================

@router.get("/dashboard")
def get_admin_dashboard(
    current_user=Depends(require_admin)
):

    # ========================================================
    # USER STATISTICS
    # ========================================================

    total_users = users_collection.count_documents({})

    total_doctors = users_collection.count_documents({
        "role": "doctor"
    })

    total_patients = users_collection.count_documents({
        "role": {
            "$in": [
                "patient",
                "user",
                "general_user"
            ]
        }
    })

    # ========================================================
    # CASE / IMAGE / ANALYSIS STATISTICS
    # ========================================================

    total_cases = cases_collection.count_documents({})

    total_images = images_collection.count_documents({})

    total_analyses = analyses_collection.count_documents({})

    pending_reviews = analyses_collection.count_documents({
        "analysis_status": {
            "$in": [
                "pending",
                "processing"
            ]
        }
    })

    # ========================================================
    # HIGH-RISK CASES
    # ========================================================
    #
    # Current analysis documents do not contain a separate
    # risk_score field.
    #
    # Therefore we use the predicted condition as a
    # dashboard-level high-risk indicator.
    #
    # This is NOT a medical risk calculation.
    #
    # ========================================================

    high_risk_cases = analyses_collection.count_documents({
        "analysis_status": "completed",
        "$or": [
            {"risk_category": {
                "$in": ["high", "critical", "urgent", "uncertain"]
            }},
            {"review_priority": {
                "$in": ["high", "critical", "urgent", "manual_review"]
            }}
        ]
    })

    # ========================================================
    # ACTIVE USERS
    # ========================================================
    #
    # Active = updated/logged activity within last 24 hours.
    #
    # ========================================================

    active_since = (
      datetime.now(timezone.utc)
      - timedelta(minutes=15)
    )

    active_users = users_collection.count_documents({
        "last_seen": {
            "$gte": active_since
        },
        "role": {
            "$in": [
                "patient",
                "doctor"
            ]
        }
    })

    # ========================================================
    # COMPLETED / FAILED ANALYSES
    # ========================================================

    completed_analyses = analyses_collection.count_documents({
        "analysis_status": "completed"
    })

    failed_analyses = analyses_collection.count_documents({
        "analysis_status": "failed"
    })

    processing_analyses = analyses_collection.count_documents({
        "analysis_status": "processing"
    })

    # ========================================================
    # MODEL USAGE
    # ========================================================

    resnet50_usage = analyses_collection.count_documents({
        "model": "resnet50"
    })

    densenet121_usage = analyses_collection.count_documents({
        "model": "densenet121"
    })

    # ========================================================
    # RECENT ACTIVITY
    # ========================================================

    recent_activity = []

    # --------------------------------------------------------
    # Recent users
    # --------------------------------------------------------

    recent_users = users_collection.find(
        {}
    ).sort(
        "created_at",
        -1
    ).limit(5)

    for user in recent_users:

        role = user.get(
            "role",
            "user"
        )

        role_label = (
            "Doctor"
            if role == "doctor"
            else "User"
        )

        recent_activity.append({
            "id": f"user_{user['_id']}",
            "type": "user",
            "action": "User registered",
            "title": f"{role_label} registered",
            "description": user.get(
                "full_name",
                user.get(
                    "email",
                    "New user"
                )
            ),
            "status": "completed",
            "created_at": serialize_datetime(
                user.get("created_at")
            )
        })

    # --------------------------------------------------------
    # Recent analyses
    # --------------------------------------------------------

    recent_analyses = analyses_collection.find(
        {}
    ).sort(
        "analysis_completed_at",
        -1
    ).limit(8)

    for analysis in recent_analyses:

        analysis_status = analysis.get(
            "analysis_status",
            "pending"
        )

        predicted_condition = analysis.get(
            "predicted_condition"
        )

        if analysis_status == "completed":

            title = "AI analysis completed"

            if predicted_condition:
                description = (
                    f"Prediction: "
                    f"{predicted_condition}"
                )
            else:
                description = "AI prediction completed."

        elif analysis_status == "failed":

            title = "AI analysis failed"
            description = analysis.get(
                "error",
                "Model analysis failed."
            )

        else:

            title = "AI analysis processing"
            description = (
                "Analysis is awaiting completion."
            )

        recent_activity.append({
            "id": f"analysis_{analysis['_id']}",
            "type": "analysis",
            "action": "analysis",
            "title": title,
            "description": description,
            "status": analysis_status,
            "created_at": serialize_datetime(
                analysis.get(
                    "analysis_completed_at"
                )
                or analysis.get(
                    "analysis_started_at"
                )
                or analysis.get(
                    "started_at"
                )
            )
        })

    # --------------------------------------------------------
    # Recent cases
    # --------------------------------------------------------

    recent_cases = cases_collection.find(
        {}
    ).sort(
        "created_at",
        -1
    ).limit(5)

    for case in recent_cases:

        recent_activity.append({
            "id": f"case_{case['_id']}",
            "type": "system",
            "action": "case",
            "title": "New patient case",
            "description": case.get(
                "title",
                "Skin assessment case"
            ),
            "status": case.get(
                "status",
                "active"
            ),
            "created_at": serialize_datetime(
                case.get("created_at")
            )
        })

    # ========================================================
    # SORT ACTIVITY
    # ========================================================

    recent_activity.sort(
        key=lambda item: item.get(
            "created_at"
        ) or "",
        reverse=True
    )

    recent_activity = recent_activity[:20]

    # ========================================================
    # RETURN DASHBOARD
    # ========================================================

    return {

        "admin": {
            "id": str(current_user["_id"]),
            "full_name": current_user.get(
                "full_name",
                "Administrator"
            ),
            "email": current_user.get(
                "email",
                ""
            )
        },

        "stats": {

            "total_users":
                total_users,

            "total_doctors":
                total_doctors,

            "total_patients":
                total_patients,

            "total_cases":
                total_cases,

            "total_images":
                total_images,

            "total_analyses":
                total_analyses,

            "pending_reviews":
                pending_reviews,

            "high_risk_cases":
                high_risk_cases,

            "active_users":
                active_users,

            "completed_analyses":
                completed_analyses,

            "processing_analyses":
                processing_analyses,

            "failed_analyses":
                failed_analyses,

            "resnet50_usage":
                resnet50_usage,

            "densenet121_usage":
                densenet121_usage,

            "system_uptime":
                "99.9%"
        },

        "recent_activity":
            recent_activity
    }


# ============================================================
# ADMIN USER AND ANALYSIS MANAGEMENT
# ============================================================

admin_password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
PATIENT_ROLES = ["patient", "user", "general_user"]


def serialize_admin_user(user):
    return {
        "id": str(user["_id"]),
        "full_name": user.get("full_name", ""),
        "email": user.get("email", ""),
        "role": user.get("role", ""),
        "created_at": serialize_datetime(user.get("created_at")),
    }


@router.get("/patients")
def list_patients(current_user=Depends(require_admin)):
    query = {"role": {"$in": PATIENT_ROLES}}
    patients = users_collection.find(query).sort("created_at", -1)
    return {
        "count": users_collection.count_documents(query),
        "items": [serialize_admin_user(item) for item in patients],
    }


@router.get("/doctors")
def list_doctors(current_user=Depends(require_admin)):
    query = {"role": "doctor"}
    doctors = users_collection.find(query).sort("created_at", -1)
    return {
        "count": users_collection.count_documents(query),
        "items": [serialize_admin_user(item) for item in doctors],
    }


@router.patch("/users/{user_id}")
def update_managed_user(
    user_id: str,
    payload: dict = Body(...),
    current_user=Depends(require_admin),
):
    try:
        target_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user ID.")

    target = users_collection.find_one({"_id": target_id})
    if not target or target.get("role") not in PATIENT_ROLES + ["doctor"]:
        raise HTTPException(status_code=404, detail="Patient or doctor not found.")

    updates = {}
    if "full_name" in payload:
        full_name = str(payload.get("full_name", "")).strip()
        if not full_name:
            raise HTTPException(status_code=400, detail="Name cannot be empty.")
        updates["full_name"] = full_name

    if "email" in payload:
        email = str(payload.get("email", "")).strip().lower()
        if not email or "@" not in email:
            raise HTTPException(status_code=400, detail="Enter a valid email address.")
        duplicate = users_collection.find_one({
            "email": email,
            "_id": {"$ne": target_id},
        })
        if duplicate:
            raise HTTPException(status_code=409, detail="Email is already registered.")
        updates["email"] = email

    if payload.get("password"):
        password = str(payload["password"])
        if len(password) < 8:
            raise HTTPException(status_code=400, detail="Password must contain at least 8 characters.")
        updates["password_hash"] = admin_password_context.hash(password)

    if not updates:
        raise HTTPException(status_code=400, detail="No changes were provided.")

    updates["updated_at"] = datetime.now(timezone.utc)
    users_collection.update_one({"_id": target_id}, {"$set": updates})

    # Revoke existing sessions when credentials change.
    if "password_hash" in updates or "email" in updates:
        sessions_collection.update_many(
            {"user_id": target_id, "is_active": True},
            {"$set": {
                "is_active": False,
                "logged_out_at": datetime.now(timezone.utc),
                "revoked_by_admin": True,
            }},
        )

    updated = users_collection.find_one({"_id": target_id})
    return {
        "message": "Account updated successfully.",
        "user": serialize_admin_user(updated),
    }


@router.delete("/users/{user_id}")
def delete_managed_user(
    user_id: str,
    current_user=Depends(require_admin),
):
    try:
        target_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user ID.")

    target = users_collection.find_one({"_id": target_id})
    if not target or target.get("role") not in PATIENT_ROLES + ["doctor"]:
        raise HTTPException(status_code=404, detail="Patient or doctor not found.")

    # Revoke active sessions before removing the account.
    sessions_collection.update_many(
        {"user_id": target_id, "is_active": True},
        {"$set": {
            "is_active": False,
            "logged_out_at": datetime.now(timezone.utc),
            "revoked_by_admin": True,
        }},
    )
    users_collection.delete_one({"_id": target_id})
    return {"message": "Account deleted successfully. The user can no longer sign in."}


@router.get("/analyses")
def list_managed_analyses(current_user=Depends(require_admin)):
    records = analyses_collection.find({}).sort("analysis_started_at", -1).limit(500)
    items = []

    for record in records:
        image = (
            images_collection.find_one({"_id": record.get("image_id")})
            if record.get("image_id") else None
        )
        case = (
            cases_collection.find_one({"_id": image.get("case_id")})
            if image and image.get("case_id") else None
        )
        patient = (
            users_collection.find_one({"_id": case.get("user_id")})
            if case and case.get("user_id") else None
        )
        started_at = (
            record.get("analysis_started_at")
            or record.get("started_at")
            or record.get("created_at")
        )

        items.append({
            "id": str(record["_id"]),
            "patient_name": (patient or {}).get("full_name", "Unknown patient"),
            "patient_email": (patient or {}).get("email", ""),
            "case_id": str(case["_id"]) if case else None,
            "analysis_status": record.get("analysis_status", "pending"),
            "preprocessing_status": record.get("preprocessing_status", "unknown"),
            "predicted_condition": record.get("predicted_condition"),
            "confidence": record.get("confidence"),
            "risk_category": record.get("risk_category"),
            "review_priority": record.get("review_priority"),
            "model": record.get("model"),
            "started_at": serialize_datetime(started_at),
            "completed_at": serialize_datetime(
                record.get("analysis_completed_at") or record.get("completed_at")
            ),
            "error": record.get("error"),
        })

    return {
        "count": analyses_collection.count_documents({}),
        "items": items,
    }


@router.delete("/analyses/{analysis_id}")
def delete_managed_analysis(
    analysis_id: str,
    current_user=Depends(require_admin),
):
    try:
        target_id = ObjectId(analysis_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid analysis ID.")

    result = analyses_collection.delete_one({"_id": target_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    return {"message": "Analysis record deleted successfully."}

