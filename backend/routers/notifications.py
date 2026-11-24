from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status

from app.database import (
    users_collection,
    cases_collection,
    images_collection,
    analyses_collection,
)
from app.dependencies.auth import get_current_user


router = APIRouter(
    prefix="/notifications",
    tags=["Notifications"],
)


# ============================================================
# HELPERS
# ============================================================

def serialize_datetime(value):
    if isinstance(value, datetime):
        return value.isoformat()

    return value


def normalize_status(value):
    return str(value or "").strip().lower().replace("_", "-")


def confidence_value(value):
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if value > 1:
        value = value / 100

    return value


def get_event_time(document):
    return (
        document.get("analysis_completed_at")
        or document.get("completed_at")
        or document.get("analysis_started_at")
        or document.get("started_at")
        or document.get("created_at")
        or datetime.now(timezone.utc)
    )


def make_notification(
    notification_id,
    title,
    message,
    level,
    created_at,
    notification_type="info",
    case_id=None,
    analysis_id=None,
    read_ids=None,
):
    read_ids = read_ids or []

    return {
        "id": notification_id,
        "title": title,
        "message": message,
        "level": level,
        "type": notification_type,
        "case_id": str(case_id) if case_id else None,
        "analysis_id": str(analysis_id) if analysis_id else None,
        "created_at": serialize_datetime(created_at),
        "read": notification_id in read_ids,
    }


def get_read_ids(current_user):
    return [
        str(value)
        for value in current_user.get(
            "notification_read_ids",
            []
        )
    ]


# ============================================================
# PATIENT NOTIFICATIONS
# ============================================================

def build_patient_notifications(
    current_user,
    read_ids,
):
    notifications = []

    cases = list(
        cases_collection.find(
            {
                "user_id": current_user["_id"]
            }
        ).sort(
            "updated_at",
            -1,
        )
    )

    for case in cases:
        case_id = case["_id"]
        case_title = case.get(
            "title",
            "Untitled Case",
        )

        images = list(
            images_collection.find(
                {
                    "case_id": case_id
                }
            ).sort(
                "uploaded_at",
                -1,
            )
        )

        # ----------------------------------------------------
        # IMAGE QUALITY
        # ----------------------------------------------------

        for image in images:
            quality_status = normalize_status(
                image.get("quality_status")
            )

            if quality_status in {
                "poor",
                "warning",
            }:
                notification_id = (
                    f"patient:image-quality:{image['_id']}"
                )

                level = (
                    "high"
                    if quality_status == "poor"
                    else "medium"
                )

                notifications.append(
                    make_notification(
                        notification_id=notification_id,
                        title="Image quality needs attention",
                        message=(
                            f"The uploaded image for "
                            f'"{case_title}" '
                            f"was marked as "
                            f"{quality_status}. "
                            "Consider uploading a clearer image."
                        ),
                        level=level,
                        created_at=image.get(
                            "uploaded_at",
                            image.get("created_at"),
                        ),
                        notification_type="image_quality",
                        case_id=case_id,
                        read_ids=read_ids,
                    )
                )

        # ----------------------------------------------------
        # ANALYSIS NOTIFICATIONS
        # ----------------------------------------------------

        image_ids = [
            image["_id"]
            for image in images
        ]

        if not image_ids:
            continue

        analyses = list(
            analyses_collection.find(
                {
                    "image_id": {
                        "$in": image_ids
                    }
                }
            ).sort(
                [
                    (
                        "analysis_completed_at",
                        -1,
                    ),
                    (
                        "analysis_started_at",
                        -1,
                    ),
                    (
                        "started_at",
                        -1,
                    ),
                ]
            )
        )

        for analysis in analyses:
            analysis_id = analysis["_id"]
            analysis_status = normalize_status(
                analysis.get("analysis_status")
            )

            event_time = get_event_time(
                analysis
            )

            # Analysis completed
            if analysis_status in {
                "completed",
                "complete",
                "success",
                "successful",
            }:
                notification_id = (
                    f"patient:analysis-completed:{analysis_id}"
                )

                condition = (
                    analysis.get(
                        "predicted_condition"
                    )
                    or "AI-assisted result"
                )

                notifications.append(
                    make_notification(
                        notification_id=notification_id,
                        title="Analysis completed",
                        message=(
                            f'Your analysis for "{case_title}" '
                            f"is ready. AI prediction: "
                            f"{condition}."
                        ),
                        level="informational",
                        created_at=event_time,
                        notification_type="analysis_completed",
                        case_id=case_id,
                        analysis_id=analysis_id,
                        read_ids=read_ids,
                    )
                )

                # Low confidence
                confidence = confidence_value(
                    analysis.get("confidence")
                )

                if (
                    confidence is not None
                    and confidence < 0.60
                ):
                    notification_id = (
                        f"patient:low-confidence:{analysis_id}"
                    )

                    notifications.append(
                        make_notification(
                            notification_id=notification_id,
                            title="Low-confidence prediction",
                            message=(
                                "The AI prediction has "
                                f"{confidence * 100:.1f}% confidence. "
                                "Professional review is recommended."
                            ),
                            level="medium",
                            created_at=event_time,
                            notification_type="low_confidence",
                            case_id=case_id,
                            analysis_id=analysis_id,
                            read_ids=read_ids,
                        )
                    )

                # High-risk where backend/model provides it
                risk = normalize_status(
                    analysis.get("risk_category")
                )

                if risk in {
                    "high",
                    "critical",
                }:
                    notification_id = (
                        f"patient:high-risk:{analysis_id}"
                    )

                    notifications.append(
                        make_notification(
                            notification_id=notification_id,
                            title="High-risk prediction alert",
                            message=(
                                "The AI result has been marked "
                                "as high risk. Please seek "
                                "appropriate professional review."
                            ),
                            level="high",
                            created_at=event_time,
                            notification_type="high_risk",
                            case_id=case_id,
                            analysis_id=analysis_id,
                            read_ids=read_ids,
                        )
                    )

            # Failed analysis
            elif analysis_status in {
                "failed",
                "error",
            }:
                notification_id = (
                    f"patient:analysis-failed:{analysis_id}"
                )

                notifications.append(
                    make_notification(
                        notification_id=notification_id,
                        title="Analysis could not be completed",
                        message=(
                            f'The analysis for "{case_title}" '
                            "failed. Please try again."
                        ),
                        level="high",
                        created_at=event_time,
                        notification_type="analysis_failed",
                        case_id=case_id,
                        analysis_id=analysis_id,
                        read_ids=read_ids,
                    )
                )

        # ----------------------------------------------------
        # FOLLOW-UP
        # ----------------------------------------------------

        follow_up_status = normalize_status(
            case.get("follow_up_status")
        )

        if follow_up_status == "scheduled":
            notification_id = (
                f"patient:follow-up:{case_id}"
            )

            notifications.append(
                make_notification(
                    notification_id=notification_id,
                    title="Follow-up scheduled",
                    message=(
                        f'A follow-up has been scheduled '
                        f'for "{case_title}".'
                    ),
                    level="medium",
                    created_at=case.get(
                        "updated_at",
                        case.get("created_at"),
                    ),
                    notification_type="follow_up",
                    case_id=case_id,
                    read_ids=read_ids,
                )
            )

    return notifications


# ============================================================
# DOCTOR NOTIFICATIONS
# ============================================================

def build_doctor_notifications(
    current_user,
    read_ids,
):
    notifications = []

    cases = list(
        cases_collection.find({}).sort(
            "updated_at",
            -1,
        )
    )

    for case in cases:
        case_id = case["_id"]

        patient = users_collection.find_one(
            {
                "_id": case.get("user_id")
            }
        )

        patient_name = (
            patient.get(
                "full_name",
                "Unknown Patient",
            )
            if patient
            else "Unknown Patient"
        )

        case_title = case.get(
            "title",
            "Untitled Assessment",
        )

        images = list(
            images_collection.find(
                {
                    "case_id": case_id
                }
            ).sort(
                "uploaded_at",
                -1,
            )
        )

        image_ids = [
            image["_id"]
            for image in images
        ]

        latest_analysis = None

        if image_ids:
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
                            -1,
                        ),
                        (
                            "analysis_started_at",
                            -1,
                        ),
                        (
                            "started_at",
                            -1,
                        ),
                    ],
                )
            )

        # Completed analysis -> clinical review
        if latest_analysis:
            analysis_id = latest_analysis["_id"]
            analysis_status = normalize_status(
                latest_analysis.get(
                    "analysis_status"
                )
            )

            event_time = get_event_time(
                latest_analysis
            )

            if analysis_status in {
                "completed",
                "complete",
                "success",
                "successful",
            }:
                notification_id = (
                    f"doctor:review:{analysis_id}"
                )

                notifications.append(
                    make_notification(
                        notification_id=notification_id,
                        title="Clinical review recommended",
                        message=(
                            f"AI analysis for "
                            f"{patient_name}'s "
                            f'"{case_title}" '
                            "is ready for review."
                        ),
                        level="medium",
                        created_at=event_time,
                        notification_type="clinical_review",
                        case_id=case_id,
                        analysis_id=analysis_id,
                        read_ids=read_ids,
                    )
                )

                risk = normalize_status(
                    latest_analysis.get(
                        "risk_category"
                    )
                )

                if risk in {
                    "high",
                    "critical",
                }:
                    notification_id = (
                        f"doctor:high-risk:{analysis_id}"
                    )

                    notifications.append(
                        make_notification(
                            notification_id=notification_id,
                            title="High-risk case requires attention",
                            message=(
                                f"{patient_name}'s "
                                f'"{case_title}" '
                                "contains a high-risk AI prediction."
                            ),
                            level="high",
                            created_at=event_time,
                            notification_type="high_risk",
                            case_id=case_id,
                            analysis_id=analysis_id,
                            read_ids=read_ids,
                        )
                    )

            elif analysis_status in {
                "failed",
                "error",
            }:
                notification_id = (
                    f"doctor:analysis-failed:{analysis_id}"
                )

                notifications.append(
                    make_notification(
                        notification_id=notification_id,
                        title="AI analysis failed",
                        message=(
                            f"The analysis for "
                            f"{patient_name}'s "
                            f'"{case_title}" '
                            "could not be completed."
                        ),
                        level="high",
                        created_at=event_time,
                        notification_type="analysis_failed",
                        case_id=case_id,
                        analysis_id=analysis_id,
                        read_ids=read_ids,
                    )
                )

        else:
            # New case without analysis
            notification_id = (
                f"doctor:new-case:{case_id}"
            )

            notifications.append(
                make_notification(
                    notification_id=notification_id,
                    title="New patient case",
                    message=(
                        f"{patient_name} has a new "
                        f'case: "{case_title}".'
                    ),
                    level="informational",
                    created_at=case.get(
                        "created_at"
                    ),
                    notification_type="new_case",
                    case_id=case_id,
                    read_ids=read_ids,
                )
            )

    return notifications


# ============================================================
# ADMIN NOTIFICATIONS
# ============================================================

def build_admin_notifications(
    current_user,
    read_ids,
):
    notifications = []

    pending_reviews = cases_collection.count_documents(
        {
            "status": {
                "$in": [
                    "pending",
                    "active",
                ]
            }
        }
    )

    high_risk = analyses_collection.count_documents(
        {
            "risk_category": {
                "$in": [
                    "high",
                    "critical",
                    "High",
                    "Critical",
                ]
            }
        }
    )

    total_analyses = analyses_collection.count_documents({})

    if pending_reviews > 0:
        notification_id = (
            f"admin:pending-reviews:{pending_reviews}"
        )

        notifications.append(
            make_notification(
                notification_id=notification_id,
                title="Pending clinical reviews",
                message=(
                    f"There are {pending_reviews} "
                    "cases awaiting clinical review."
                ),
                level="medium",
                created_at=datetime.now(timezone.utc),
                notification_type="pending_reviews",
                read_ids=read_ids,
            )
        )

    if high_risk > 0:
        notification_id = (
            f"admin:high-risk:{high_risk}"
        )

        notifications.append(
            make_notification(
                notification_id=notification_id,
                title="High-risk cases detected",
                message=(
                    f"{high_risk} analysis record(s) "
                    "currently contain high-risk predictions."
                ),
                level="high",
                created_at=datetime.now(timezone.utc),
                notification_type="high_risk",
                read_ids=read_ids,
            )
        )

    if total_analyses > 0:
        latest_analysis = analyses_collection.find_one(
            {},
            sort=[
                (
                    "analysis_completed_at",
                    -1,
                ),
                (
                    "created_at",
                    -1,
                ),
            ],
        )

        if latest_analysis:
            analysis_id = latest_analysis["_id"]

            notifications.append(
                make_notification(
                    notification_id=(
                        f"admin:latest-analysis:{analysis_id}"
                    ),
                    title="New AI analysis activity",
                    message=(
                        "A new AI-assisted analysis "
                        "has been recorded on the platform."
                    ),
                    level="informational",
                    created_at=get_event_time(
                        latest_analysis
                    ),
                    notification_type="analysis_activity",
                    analysis_id=analysis_id,
                    read_ids=read_ids,
                )
            )

    return notifications


# ============================================================
# GET NOTIFICATIONS
# ============================================================

@router.get("")
def get_notifications(
    current_user=Depends(get_current_user),
):
    read_ids = get_read_ids(
        current_user
    )

    role = str(
        current_user.get(
            "role",
            "patient",
        )
    ).lower()

    if role == "doctor":
        notifications = build_doctor_notifications(
            current_user,
            read_ids,
        )

    elif role in {
        "admin",
        "administrator",
    }:
        notifications = build_admin_notifications(
            current_user,
            read_ids,
        )

    else:
        notifications = build_patient_notifications(
            current_user,
            read_ids,
        )

    notifications.sort(
        key=lambda item: item.get(
            "created_at",
            "",
        ),
        reverse=True,
    )

    notifications = notifications[:30]

    unread_count = sum(
        1
        for item in notifications
        if not item["read"]
    )

    return {
        "notifications": notifications,
        "unread_count": unread_count,
    }


# ============================================================
# MARK ONE AS READ
# ============================================================

@router.post("/{notification_id}/read")
def mark_notification_read(
    notification_id: str,
    current_user=Depends(get_current_user),
):
    users_collection.update_one(
        {
            "_id": current_user["_id"]
        },
        {
            "$addToSet": {
                "notification_read_ids": notification_id
            }
        },
    )

    return {
        "message": "Notification marked as read."
    }


# ============================================================
# MARK ALL AS READ
# ============================================================

@router.post("/read-all")
def mark_all_notifications_read(
    current_user=Depends(get_current_user),
):
    read_ids = get_read_ids(
        current_user
    )

    role = str(
        current_user.get(
            "role",
            "patient",
        )
    ).lower()

    if role == "doctor":
        notifications = build_doctor_notifications(
            current_user,
            read_ids,
        )

    elif role in {
        "admin",
        "administrator",
    }:
        notifications = build_admin_notifications(
            current_user,
            read_ids,
        )

    else:
        notifications = build_patient_notifications(
            current_user,
            read_ids,
        )

    ids = [
        item["id"]
        for item in notifications
    ]

    if ids:
        users_collection.update_one(
            {
                "_id": current_user["_id"]
            },
            {
                "$addToSet": {
                    "notification_read_ids": {
                        "$each": ids
                    }
                }
            },
        )

    return {
        "message": "All notifications marked as read.",
        "count": len(ids),
    }