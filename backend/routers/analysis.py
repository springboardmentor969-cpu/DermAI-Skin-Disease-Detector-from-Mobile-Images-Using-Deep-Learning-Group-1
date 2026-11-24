# ============================================================
# DERMAI - IMAGE ANALYSIS ROUTES
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import sys
from PIL import Image
import numpy as np
import traceback

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from bson import ObjectId
from fastapi.responses import FileResponse

# ============================================================
# DERMAI PROJECT ROOT
# ============================================================

DERMAI_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# MAKE DERMAI ROOT IMPORTABLE
# ============================================================

DERMAI_ROOT_STR = str(
    DERMAI_ROOT
)

if DERMAI_ROOT_STR not in sys.path:

    sys.path.insert(
        0,
        DERMAI_ROOT_STR,
    )

from xai.gradcam import generate_gradcam
# ============================================================
# DATABASE
# ============================================================

from app.database import (
    cases_collection,
    images_collection,
    analyses_collection,
)


# ============================================================
# AUTHENTICATION
# ============================================================

from app.dependencies.auth import (
    get_current_user,
)


# ============================================================
# DERMAI PREPROCESSING PIPELINE
# ============================================================

from preprocessing.image_preprocessing import (
    preprocess_image,
)


# ============================================================
# REAL MODEL INFERENCE
# ============================================================

from backend.services.inference import (
    predict_image,
)


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/cases",
    tags=["Image Analysis"],
)


# ============================================================
# DIRECTORIES
# ============================================================

PROCESSED_UPLOAD_DIR = (
    DERMAI_ROOT
    / "uploads"
    / "processed"
)


PROCESSED_UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPER — VERIFY CASE OWNERSHIP
# ============================================================

def get_owned_case(
    case_id: str,
    user_id: ObjectId,
):

    try:

        case_object_id = ObjectId(
            case_id
        )

    except Exception:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid case ID.",
        )


    case = cases_collection.find_one(
        {
            "_id": case_object_id,
            "user_id": user_id,
        }
    )


    if not case:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found.",
        )


    return case


# ============================================================
# HELPER — VERIFY IMAGE BELONGS TO CASE
# ============================================================

def get_owned_image(
    case_id: str,
    image_id: str,
    user_id: ObjectId,
):

    case = get_owned_case(
        case_id=case_id,
        user_id=user_id,
    )


    try:

        image_object_id = ObjectId(
            image_id
        )

    except Exception:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image ID.",
        )


    image = images_collection.find_one(
        {
            "_id": image_object_id,
            "case_id": case["_id"],
        }
    )


    if not image:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found.",
        )


    return case, image


# ============================================================
# GET ALL ANALYSES FOR A CASE
# ============================================================

@router.get(
    "/{case_id}/analyses"
)
def get_case_analyses(
    case_id: str,
    current_user=Depends(
        get_current_user
    ),
):

    # ========================================================
    # VERIFY CASE OWNERSHIP
    # ========================================================

    case = get_owned_case(
        case_id=case_id,
        user_id=current_user["_id"],
    )


    # ========================================================
    # GET ALL IMAGES FOR THIS CASE
    # ========================================================

    images = list(
        images_collection.find(
            {
                "case_id": case["_id"]
            }
        )
    )


    # ========================================================
    # NO IMAGES
    # ========================================================

    if not images:

        return []


    # ========================================================
    # CREATE IMAGE ID LIST
    # ========================================================

    image_ids = [
        image["_id"]
        for image in images
    ]


    # ========================================================
    # GET ANALYSES
    # ========================================================

    analyses = list(
        analyses_collection.find(
            {
                "image_id": {
                    "$in": image_ids
                }
            }
        ).sort(
            "started_at",
            -1
        )
    )


    # ========================================================
    # CONVERT MONGODB DOCUMENTS TO JSON
    # ========================================================

    results = []

    for analysis in analyses:

        results.append(
            {
                "id": str(
                    analysis["_id"]
                ),

                "case_id": str(
                    case["_id"]
                ),

                "image_id": str(
                    analysis["image_id"]
                ),

                "preprocessing_status":
                    analysis.get(
                        "preprocessing_status",
                        "pending",
                    ),

                "analysis_status":
                    analysis.get(
                        "analysis_status",
                        "pending",
                    ),

                "predicted_condition":
                    analysis.get(
                        "predicted_condition"
                    ),

                "confidence":
                    analysis.get(
                        "confidence"
                    ),

                "risk_assessment": analysis.get("risk_assessment", {}),

                "risk_category": analysis.get("risk_category"),

                "review_priority": analysis.get("review_priority"),

                "assessment_type": analysis.get("assessment_type"),

                "risk_message": analysis.get("risk_message"),

                "top_predictions":
                    analysis.get(
                        "top_predictions",
                        [],
                    ),

                "model":
                    analysis.get(
                        "model"
                    ),

                "device":
                    analysis.get(
                        "device"
                    ),

                "preprocessed_path":
                    analysis.get(
                        "preprocessed_path"
                    ),

                "error":
                    analysis.get(
                        "error"
                    ),

                "started_at":
                    analysis.get(
                        "started_at"
                    ),

                "completed_at":
                    analysis.get(
                        "completed_at"
                    ),

                "analysis_started_at":
                    analysis.get(
                        "analysis_started_at"
                    ),

                "analysis_completed_at":
                    analysis.get(
                        "analysis_completed_at"
                    ),

                "gradcam_status": analysis.get("gradcam_status", "not_available"),

                "gradcam_url": (
                    f"/cases/{case_id}/analyses/{analysis['_id']}/gradcam"
                    if analysis.get("gradcam_path")
                    else None
                ),

                "medical_disclaimer":
                    analysis.get(
                        "medical_disclaimer"
                    ),
            }
        )


    # ========================================================
    # RETURN ANALYSES
    # ========================================================

    return results


# ============================================================
# PREPROCESS UPLOADED IMAGE
# ============================================================

@router.post(
    "/{case_id}/images/{image_id}/preprocess"
)
def preprocess_uploaded_image(
    case_id: str,
    image_id: str,
    current_user=Depends(
        get_current_user
    ),
):

    # ========================================================
    # VERIFY OWNERSHIP
    # ========================================================

    case, image_document = get_owned_image(
        case_id=case_id,
        image_id=image_id,
        user_id=current_user["_id"],
    )


    # ========================================================
    # LOCATE ORIGINAL UPLOADED IMAGE
    # ========================================================

    original_path = Path(
        image_document["file_path"]
    )


    if not original_path.exists():

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Original image file "
                "not found on server."
            ),
        )


    # ========================================================
    # CHECK WHETHER PREPROCESSING ALREADY EXISTS
    # ========================================================

    existing_analysis = (
        analyses_collection.find_one(
            {
                "image_id":
                    image_document["_id"]
            }
        )
    )


    if existing_analysis:

        cases_collection.update_one(
            {"_id": case["_id"]},
            {"$set": {
                "status": "pending",
                "updated_at": datetime.now(timezone.utc),
            }},
        )

        return {
            "message":
                "Image has already been preprocessed.",

            "analysis_id":
                str(
                    existing_analysis["_id"]
                ),

            "image_id":
                str(
                    image_document["_id"]
                ),

            "preprocessing_status":
                existing_analysis.get(
                    "preprocessing_status",
                    "unknown",
                ),

            "preprocessed_path":
                existing_analysis.get(
                    "preprocessed_path"
                ),
        }


    # ========================================================
    # CREATE ANALYSIS RECORD
    # ========================================================

    started_at = datetime.now(
        timezone.utc
    )


    analysis_document = {

        "image_id":
            image_document["_id"],

        "preprocessing_status":
            "processing",

        "preprocessed_path":
            None,

        "analysis_status":
            "pending",

        "started_at":
            started_at,

        "completed_at":
            None,
    }


    analysis_result = (
        analyses_collection.insert_one(
            analysis_document
        )
    )


    analysis_id = (
        analysis_result.inserted_id
    )

    cases_collection.update_one(
        {"_id": case["_id"]},
        {"$set": {
            "status": "pending",
            "updated_at": datetime.now(timezone.utc),
        }},
    )


    # ========================================================
    # PROCESS IMAGE
    # ========================================================

    try:

        processed_array = preprocess_image(
            original_path
        )


        # ----------------------------------------------------
        # Validate NumPy array
        # ----------------------------------------------------

        if not isinstance(
            processed_array,
            np.ndarray,
        ):

            raise ValueError(
                "Preprocessing pipeline "
                "did not return a NumPy array."
            )


        if processed_array.ndim != 3:

            raise ValueError(
                "Unexpected processed image "
                f"dimensions: {processed_array.shape}"
            )


        if processed_array.shape[0] != 224:

            raise ValueError(
                "Expected height 224, got "
                f"{processed_array.shape[0]}"
            )


        if processed_array.shape[1] != 224:

            raise ValueError(
                "Expected width 224, got "
                f"{processed_array.shape[1]}"
            )


        if processed_array.shape[2] != 3:

            raise ValueError(
                "Expected 3 RGB channels, got "
                f"{processed_array.shape[2]}"
            )


        # ====================================================
        # CONVERT [0,1] ARRAY TO RGB IMAGE
        # ====================================================

        processed_array = np.clip(
            processed_array,
            0.0,
            1.0,
        )


        processed_uint8 = (
            processed_array * 255
        ).round().astype(
            np.uint8
        )


        processed_image = Image.fromarray(
            processed_uint8,
            mode="RGB",
        )


        # ====================================================
        # PROCESSED FILENAME
        # ====================================================

        original_stem = (
            original_path.stem
        )


        processed_filename = (
            f"{original_stem}_processed.png"
        )


        processed_path = (
            PROCESSED_UPLOAD_DIR
            / processed_filename
        )


        # ====================================================
        # SAVE PROCESSED IMAGE
        # ====================================================

        processed_image.save(
            processed_path,
            format="PNG",
        )


        # ====================================================
        # VERIFY SAVED IMAGE
        # ====================================================

        with Image.open(
            processed_path
        ) as verification_image:

            verification_image.load()

            width, height = (
                verification_image.size
            )

            mode = (
                verification_image.mode
            )


        if width != 224 or height != 224:

            raise ValueError(
                "Processed image has incorrect "
                f"dimensions: {width}x{height}"
            )


        if mode != "RGB":

            raise ValueError(
                "Processed image has incorrect "
                f"mode: {mode}"
            )


        # ====================================================
        # UPDATE ANALYSIS RECORD
        # ====================================================

        completed_at = datetime.now(
            timezone.utc
        )


        analyses_collection.update_one(
            {
                "_id": analysis_id
            },
            {
                "$set": {

                    "preprocessing_status":
                        "completed",

                    "preprocessed_path":
                        str(
                            processed_path
                        ),

                    "analysis_status":
                        "pending",

                    "completed_at":
                        completed_at,
                }
            }
        )


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return {

            "message":
                "Image preprocessing completed successfully.",

            "analysis_id":
                str(analysis_id),

            "case_id":
                str(case["_id"]),

            "image_id":
                str(
                    image_document["_id"]
                ),

            "preprocessing_status":
                "completed",

            "original_image": {

                "filename":
                    image_document[
                        "original_filename"
                    ],

                "width":
                    image_document[
                        "width"
                    ],

                "height":
                    image_document[
                        "height"
                    ],
            },

            "processed_image": {

                "path":
                    str(processed_path),

                "width":
                    width,

                "height":
                    height,

                "mode":
                    mode,

                "format":
                    "PNG",
            },

            "next_status":
                "pending",
        }


    # ========================================================
    # PREPROCESSING ERROR
    # ========================================================

    except Exception as error:

        processed_files = list(
            PROCESSED_UPLOAD_DIR.glob(
                f"{original_path.stem}_processed.*"
            )
        )


        for processed_file in (
            processed_files
        ):

            try:

                processed_file.unlink()

            except OSError:

                pass


        analyses_collection.update_one(
            {
                "_id": analysis_id
            },
            {
                "$set": {

                    "preprocessing_status":
                        "failed",

                    "analysis_status":
                        "failed",

                    "error":
                        str(error),

                    "completed_at":
                        datetime.now(
                            timezone.utc
                        ),
                }
            }
        )


        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Image preprocessing failed: "
                f"{error}"
            ),
        )


# ============================================================
# RUN REAL AI ANALYSIS
# ============================================================

@router.post(
    "/{case_id}/images/{image_id}/analyze"
)
def analyze_image(
    case_id: str,
    image_id: str,
    current_user=Depends(
        get_current_user
    ),
):

    # ========================================================
    # VERIFY CASE + IMAGE OWNERSHIP
    # ========================================================

    case, image_document = get_owned_image(
        case_id=case_id,
        image_id=image_id,
        user_id=current_user["_id"],
    )


    # ========================================================
    # FIND PREPROCESSING RECORD
    # ========================================================

    analysis_document = (
        analyses_collection.find_one(
            {
                "image_id":
                    image_document["_id"]
            }
        )
    )


    if not analysis_document:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Image has not been preprocessed yet. "
                "Run the preprocessing endpoint first."
            ),
        )


    # ========================================================
    # CHECK PREPROCESSING STATUS
    # ========================================================

    if (
        analysis_document.get(
            "preprocessing_status"
        )
        != "completed"
    ):

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Image preprocessing is not completed."
            ),
        )


    # ========================================================
    # GET PROCESSED IMAGE PATH
    # ========================================================

    preprocessed_path = (
        analysis_document.get(
            "preprocessed_path"
        )
    )


    if not preprocessed_path:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Processed image path is missing."
            ),
        )


    processed_path = Path(
        preprocessed_path
    )


    if not processed_path.exists():

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Processed image file "
                "was not found on server."
            ),
        )


    # ========================================================
    # UPDATE STATUS TO PROCESSING
    # ========================================================

    analysis_started_at = (
        datetime.now(
            timezone.utc
        )
    )


    analyses_collection.update_one(
        {
            "_id":
                analysis_document["_id"]
        },
        {
            "$set": {

                "analysis_status":
                    "processing",

                "analysis_started_at":
                    analysis_started_at,

                "error":
                    None,
            }
        }
    )

    cases_collection.update_one(
        {"_id": case["_id"]},
        {"$set": {
            "status": "pending",
            "updated_at": datetime.now(timezone.utc),
        }},
    )


    # ========================================================
    # RUN MODEL INFERENCE
    # ========================================================

    try:

        with Image.open(
            processed_path
        ) as processed_image:

            processed_image = (
                processed_image.convert(
                    "RGB"
                )
            )

            prediction = predict_image(
                processed_image,
                model_name="resnet50",
            )


        # ====================================================
        # EXTRACT PREDICTION
        # ====================================================

        predicted_condition = (
            prediction[
                "condition"
            ]
        )


        confidence = (
            prediction[
                "confidence"
            ]
        )


        top_predictions = (
            prediction[
                "top_predictions"
            ]
        )


        model_name = (
            prediction[
                "model"
            ]
        )

        # ====================================================
        # GENERATE GRAD-CAM EXPLANATION
        # ====================================================

        gradcam_path = None
        gradcam_status = "failed"
        gradcam_error = None

        try:

            gradcam_filename = (
                f"analysis_{analysis_document['_id']}_gradcam.png"
            )

            gradcam_result = generate_gradcam(
                image_path=processed_path,
                output_filename=gradcam_filename,
            )

            gradcam_path = gradcam_result["output_path"]

            gradcam_status = "completed"

            print(
                f"[DermAI] Grad-CAM generated: {gradcam_path}"
            )

        except Exception as gradcam_exception:

            gradcam_error = str(gradcam_exception)

            gradcam_status = "failed"

            print(
                f"[DermAI] Grad-CAM generation failed: "
                f"{gradcam_exception}"
            )

            traceback.print_exc()

        # Extract risk assessment
        risk_assessment = prediction.get("risk_assessment", {})

        risk_category = prediction.get(
            "risk_category",
            risk_assessment.get("risk_category")
        )

        review_priority = prediction.get(
            "review_priority",
            risk_assessment.get("review_priority")
        )

        assessment_type = prediction.get(
            "assessment_type",
            risk_assessment.get("assessment_type")
        )

        risk_message = prediction.get(
            "risk_message",
            risk_assessment.get("message")
        )

        medical_disclaimer = prediction.get(
            "medical_disclaimer",
            risk_assessment.get("disclaimer")
        )


        # ====================================================
        # COMPLETION TIME
        # ====================================================

        completed_at = (
            datetime.now(
                timezone.utc
            )
        )


        # ====================================================
        # UPDATE MONGODB
        # ====================================================

        analyses_collection.update_one(
            {
                "_id":
                    analysis_document["_id"]
            },
            {
                "$set": {

                    "analysis_status":
                        "completed",

                    "gradcam_path": gradcam_path,

                    "gradcam_status": gradcam_status,

                    "gradcam_error": gradcam_error,

                    "predicted_condition":
                        predicted_condition,

                    "confidence":
                        confidence,

                    "risk_assessment": risk_assessment,

                    "risk_category": risk_category,

                    "review_priority": review_priority,

                    "assessment_type": assessment_type,

                    "risk_message": risk_message,

                    "medical_disclaimer": medical_disclaimer,

                    "top_predictions":
                        top_predictions,

                    "model":
                        model_name,

                    "device":
                        prediction.get(
                            "device"
                        ),

                    "analysis_completed_at":
                        completed_at,
                }
            }
        )

        cases_collection.update_one(
            {"_id": case["_id"]},
            {"$set": {
                "status": "completed",
                "updated_at": completed_at,
            }},
        )


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return {

            "message":
                "AI image analysis completed successfully.",

            "analysis_id":
                str(
                    analysis_document["_id"]
                ),

            "case_id":
                str(case["_id"]),

            "image_id":
                str(
                    image_document["_id"]
                ),

            "analysis_status":
                "completed",

            "model":
                model_name,

            "predicted_condition":
                predicted_condition,

            "confidence":
                confidence,

            "risk_assessment": risk_assessment,

            "risk_category": risk_category,

            "review_priority": review_priority,

            "assessment_type": assessment_type,

            "risk_message": risk_message,

            "top_predictions":
                top_predictions,

            "device":
                prediction.get(
                    "device"
                ),

            "image_size":
                prediction.get(
                    "image_size"
                ),

            "gradcam_status": gradcam_status,

            "gradcam_url": (
                f"/cases/{case_id}/analyses/{analysis_document['_id']}/gradcam"
                if gradcam_path
                else None
            ),

            "medical_disclaimer":
                (
                    "DermAI provides AI-assisted "
                    "analysis for research and "
                    "educational purposes only. "
                    "The result is not a medical "
                    "diagnosis and should not replace "
                    "evaluation by a qualified "
                    "healthcare professional."
                ),
        }


    # ========================================================
    # MODEL / IMAGE FILE ERROR
    # ========================================================

    except FileNotFoundError as error:

        analyses_collection.update_one(
            {
                "_id":
                    analysis_document["_id"]
            },
            {
                "$set": {

                    "analysis_status":
                        "failed",

                    "error":
                        str(error),

                    "analysis_completed_at":
                        datetime.now(
                            timezone.utc
                        ),
                }
            }
        )

        cases_collection.update_one(
            {"_id": case["_id"]},
            {"$set": {
                "status": "pending",
                "updated_at": datetime.now(timezone.utc),
            }},
        )


        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        )


    # ========================================================
    # PYTORCH / MODEL RUNTIME ERROR
    # ========================================================

    except RuntimeError as error:

        analyses_collection.update_one(
            {
                "_id":
                    analysis_document["_id"]
            },
            {
                "$set": {

                    "analysis_status":
                        "failed",

                    "error":
                        str(error),

                    "analysis_completed_at":
                        datetime.now(
                            timezone.utc
                        ),
                }
            }
        )

        cases_collection.update_one(
            {"_id": case["_id"]},
            {"$set": {
                "status": "pending",
                "updated_at": datetime.now(timezone.utc),
            }},
        )


        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Model inference failed: "
                f"{error}"
            ),
        )


    # ========================================================
    # UNEXPECTED ERROR
    # ========================================================

    except Exception as error:

        print("\n" + "=" * 70)
        print("DERMAI ANALYSIS ERROR")
        print("=" * 70)
        print(
            f"ERROR TYPE: {type(error).__name__}"
        )
        print(
            f"ERROR: {error}"
        )
        traceback.print_exc()
        print("=" * 70 + "\n")


        analyses_collection.update_one(
            {
                "_id":
                    analysis_document["_id"]
            },
            {
                "$set": {

                    "analysis_status":
                        "failed",

                    "error":
                        str(error),

                    "analysis_completed_at":
                        datetime.now(
                            timezone.utc
                        ),
                }
            }
        )

        cases_collection.update_one(
            {"_id": case["_id"]},
            {"$set": {
                "status": "pending",
                "updated_at": datetime.now(timezone.utc),
            }},
        )


        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Image analysis failed: "
                f"{error}"
            ),
        )
# ============================================================
# GET GRAD-CAM VISUALIZATION
# ============================================================

@router.get(
    "/{case_id}/analyses/{analysis_id}/gradcam"
)
def get_gradcam_visualization(
    case_id: str,
    analysis_id: str,
    current_user=Depends(get_current_user),
):

    # --------------------------------------------------------
    # VERIFY CASE OWNERSHIP
    # --------------------------------------------------------

    case = get_owned_case(
        case_id=case_id,
        user_id=current_user["_id"],
    )

    # --------------------------------------------------------
    # VALIDATE ANALYSIS ID
    # --------------------------------------------------------

    try:

        analysis_object_id = ObjectId(
            analysis_id
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid analysis ID.",
        )

    # --------------------------------------------------------
    # FIND ANALYSIS
    # --------------------------------------------------------

    analysis = analyses_collection.find_one(
        {
            "_id": analysis_object_id
        }
    )

    if not analysis:

        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    # --------------------------------------------------------
    # VERIFY IMAGE BELONGS TO CASE
    # --------------------------------------------------------

    image = images_collection.find_one(
        {
            "_id": analysis["image_id"],
            "case_id": case["_id"],
        }
    )

    if not image:

        raise HTTPException(
            status_code=404,
            detail="Analysis does not belong to this case.",
        )

    # --------------------------------------------------------
    # CHECK GRAD-CAM
    # --------------------------------------------------------

    gradcam_path = analysis.get(
        "gradcam_path"
    )

    if not gradcam_path:

        raise HTTPException(
            status_code=404,
            detail="Grad-CAM visualization is unavailable.",
        )

    gradcam_file = Path(
        gradcam_path
    ).resolve()

    allowed_directory = (
        DERMAI_ROOT / "outputs" / "gradcam"
    ).resolve()

    # Prevent paths outside the Grad-CAM directory.
    if (
        not gradcam_file.is_relative_to(
            allowed_directory
        )
    ):

        raise HTTPException(
            status_code=403,
            detail="Invalid visualization path.",
        )

    if not gradcam_file.is_file():

        raise HTTPException(
            status_code=404,
            detail="Grad-CAM image file not found.",
        )

    # --------------------------------------------------------
    # RETURN IMAGE
    # --------------------------------------------------------

    return FileResponse(
        path=str(gradcam_file),
        media_type="image/png",
        filename=gradcam_file.name,
    )   