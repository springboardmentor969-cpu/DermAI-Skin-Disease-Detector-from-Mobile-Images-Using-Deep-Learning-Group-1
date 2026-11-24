import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import api from "../services/api";
import NotificationBell from "../components/NotificationBell";
import "./PatientDashboard.css";

function PatientDashboard() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { user, logout } = useAuth();

  const fileInputRef = useRef(null);

  /* =========================
     USER
  ========================= */

  const displayName = user?.full_name || "Patient";
  const firstName = displayName.split(" ")[0];

  /* =========================
     NAVIGATION
  ========================= */

  const currentSection = searchParams.get("section") || "dashboard";
  const selectedCaseId = searchParams.get("case");

  const setSection = (section) => {
    if (section === "dashboard") {
      setSearchParams({});
    } else {
      setSearchParams({ section });
    }
  };

  const openCase = (caseId) => {
    if (!caseId) return;

    setSearchParams({
      section: "case",
      case: caseId,
    });
  };

  /* =========================
     CASE STATE
  ========================= */

  const [cases, setCases] = useState([]);
  const [loadingCases, setLoadingCases] = useState(true);
  const [caseError, setCaseError] = useState("");

  const [showCreateCase, setShowCreateCase] = useState(false);

  const [caseForm, setCaseForm] = useState({
    title: "",
    description: "",
  });

  const [creatingCase, setCreatingCase] = useState(false);
  const [createCaseError, setCreateCaseError] = useState("");

  /* =========================
     CASE DETAIL STATE
  ========================= */

  const [selectedCase, setSelectedCase] = useState(null);
  const [loadingCaseDetail, setLoadingCaseDetail] = useState(false);
  const [caseDetailError, setCaseDetailError] = useState("");

  /* =========================
     IMAGE STATE
  ========================= */

  const [images, setImages] = useState([]);
  const [loadingImages, setLoadingImages] = useState(false);
  const [imageError, setImageError] = useState("");

  const [imageUrls, setImageUrls] = useState({});

  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");

  const [uploadingImage, setUploadingImage] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadSuccess, setUploadSuccess] = useState("");

  const [dragActive, setDragActive] = useState(false);

  const [showUploadModal, setShowUploadModal] = useState(false);

  /* =========================
     ANALYSIS STATE
  ========================= */

  const [analyses, setAnalyses] = useState([]);
  const [assessmentRecords, setAssessmentRecords] = useState([]);
  const [loadingAnalyses, setLoadingAnalyses] = useState(false);

  /* =========================
     REPORT STATE
  ========================= */

  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const [showReportModal, setShowReportModal] = useState(false);

  // GRAD-CAM IMAGE STATE
const [explainabilityImageUrl, setExplainabilityImageUrl] =
  useState("");

const [explainabilityLoading, setExplainabilityLoading] =
  useState(false);

const [explainabilityError, setExplainabilityError] =
  useState("");

  /* =========================
     GLOBAL DASHBOARD COUNTS
  ========================= */

  const [allImagesCount, setAllImagesCount] = useState(0);
  const [allAssessmentsCount, setAllAssessmentsCount] =
    useState(0);

  /* =========================
     GENERAL
  ========================= */

  const [dashboardLoading, setDashboardLoading] =
    useState(false);

  /* =========================
     PROFILE
  ========================= */

  const [showProfile, setShowProfile] = useState(false);

  /* =========================
     LOAD CASES
  ========================= */

  const loadCases = async () => {
    try {
      setLoadingCases(true);
      setCaseError("");

      const response = await api.get("/cases");

      const loadedCases = response.data || [];

      setCases(loadedCases);

      return loadedCases;
    } catch (error) {
      console.error("Failed to load cases:", error);

      if (error.response?.data?.detail) {
        setCaseError(error.response.data.detail);
      } else {
        setCaseError("Unable to load your cases.");
      }

      setCases([]);

      return [];
    } finally {
      setLoadingCases(false);
    }
  };

  useEffect(() => {
    loadCases();
  }, []);

  /* =========================
     LOAD CASE DETAIL
  ========================= */

  const loadCaseDetail = async (caseId) => {
    if (!caseId) return;

    try {
      setLoadingCaseDetail(true);
      setCaseDetailError("");

      const response = await api.get(`/cases/${caseId}`);

      setSelectedCase(response.data);
    } catch (error) {
      console.error("Failed to load case:", error);

      if (error.response?.data?.detail) {
        setCaseDetailError(error.response.data.detail);
      } else {
        setCaseDetailError(
          "Unable to load case details."
        );
      }

      setSelectedCase(null);
    } finally {
      setLoadingCaseDetail(false);
    }
  };

  /* =========================
     LOAD IMAGE PREVIEW
  ========================= */

  const loadImagePreview = async (caseId, imageId) => {
  if (!caseId || !imageId) return;

  try {
    const response = await api.get(
      `/cases/${caseId}/images/${imageId}/file`,
      {
        responseType: "blob",
      }
    );

    if (!response.data || response.data.size === 0) {
      throw new Error("Empty image response");
    }

    const blobUrl = URL.createObjectURL(response.data);

    setImageUrls((previous) => {
      const oldUrl = previous[imageId];

      if (oldUrl) {
        URL.revokeObjectURL(oldUrl);
      }

      return {
        ...previous,
        [imageId]: blobUrl,
      };
    });
  } catch (error) {
    console.error(
      `Failed to load image preview ${imageId}:`,
      error.response?.status,
      error.response?.data || error.message
    );

    setImageUrls((previous) => ({
      ...previous,
      [imageId]: null,
    }));
  }
};

  /* =========================
     LOAD CASE IMAGES
  ========================= */

  const loadImages = async (caseId) => {
    if (!caseId) return [];

    try {
      setLoadingImages(true);
      setImageError("");

      const response = await api.get(
        `/cases/${caseId}/images`
      );

      const loadedImages = response.data || [];

      setImages(loadedImages);

      await Promise.all(
        loadedImages.map((image) => {
          const imageId = image.id || image._id;

          return loadImagePreview(caseId, imageId);
        })
      );

      return loadedImages;
    } catch (error) {
      console.error("Failed to load images:", error);

      if (error.response?.data?.detail) {
        setImageError(error.response.data.detail);
      } else {
        setImageError(
          "Unable to load case images."
        );
      }

      setImages([]);

      return [];
    } finally {
      setLoadingImages(false);
    }
  };

  /* =========================
     LOAD CASE ANALYSES
  ========================= */

  const loadAnalyses = async (caseId) => {
    if (!caseId) {
      setAnalyses([]);
      return [];
    }

    try {
      setLoadingAnalyses(true);

      const response = await api.get(
        `/cases/${caseId}/analyses`
      );

      const loadedAnalyses = response.data || [];

      setAnalyses(loadedAnalyses);

      // Keep the visible case status synchronized with the latest AI analysis.
      const hasCompletedAnalysis = loadedAnalyses.some((analysis) => {
        const status = getNormalizedStatus(analysis.analysis_status);
        return [
          "completed",
          "complete",
          "success",
          "successful",
        ].includes(status);
      });

      const nextCaseStatus = hasCompletedAnalysis
        ? "completed"
        : "pending";

      setSelectedCase((previous) =>
        previous
          ? { ...previous, status: nextCaseStatus }
          : previous
      );

      setCases((previousCases) =>
        previousCases.map((caseItem) =>
          String(caseItem.id || caseItem._id) === String(caseId)
            ? { ...caseItem, status: nextCaseStatus }
            : caseItem
        )
      );

      return loadedAnalyses;
    } catch (error) {
      console.error(
        "Failed to load analyses:",
        error
      );

      setAnalyses([]);

      return [];
    } finally {
      setLoadingAnalyses(false);
    }
  };

  /* =========================
     LOAD ALL DASHBOARD DATA
  ========================= */

  const loadGlobalDashboardData = async (
    loadedCases = cases
  ) => {
    if (!loadedCases.length) {
      setAllImagesCount(0);
      setAllAssessmentsCount(0);
      setAssessmentRecords([]);
      return;
    }

    try {
      const caseData = await Promise.all(
        loadedCases.map(async (caseItem) => {
          const caseId =
            caseItem.id || caseItem._id;

          try {
            const [imagesResponse, analysesResponse] =
              await Promise.all([
                api.get(
                  `/cases/${caseId}/images`
                ),
                api.get(
                  `/cases/${caseId}/analyses`
                ),
              ]);

            return {
              caseId,
              caseTitle:
                caseItem.title ||
                "Untitled Case",
              images:
                imagesResponse.data || [],
              analyses:
                analysesResponse.data || [],
            };
          } catch (error) {
            console.error(
              `Failed loading data for case ${caseId}:`,
              error
            );

            return {
              caseId,
              caseTitle:
                caseItem.title ||
                "Untitled Case",
              images: [],
              analyses: [],
            };
          }
        })
      );

      let imageCount = 0;
      let analysisCount = 0;
      const flattenedAssessments = [];

      caseData.forEach((item) => {
        imageCount += item.images.length;
        analysisCount += item.analyses.length;

        item.analyses.forEach((analysis) => {
          flattenedAssessments.push({
            ...analysis,
            caseId: item.caseId,
            caseTitle: item.caseTitle,
          });
        });
      });

      flattenedAssessments.sort((a, b) => {
        const dateA = new Date(
          a.completed_at ||
            a.started_at ||
            0
        ).getTime();

        const dateB = new Date(
          b.completed_at ||
            b.started_at ||
            0
        ).getTime();

        return dateB - dateA;
      });

      setAllImagesCount(imageCount);
      setAllAssessmentsCount(analysisCount);
      setAssessmentRecords(
        flattenedAssessments
      );
    } catch (error) {
      console.error(
        "Failed to load dashboard statistics:",
        error
      );
    }
  };

  /* =========================
     LOAD SELECTED CASE
  ========================= */

  useEffect(() => {
    if (
      currentSection === "case" &&
      selectedCaseId
    ) {
      loadCaseDetail(selectedCaseId);
      loadImages(selectedCaseId);
      loadAnalyses(selectedCaseId);
    }
  }, [currentSection, selectedCaseId]);

  /* =========================
     LOAD GLOBAL DATA
  ========================= */

  useEffect(() => {
    if (cases.length > 0) {
      loadGlobalDashboardData(cases);
    } else if (!loadingCases) {
      setAllImagesCount(0);
      setAllAssessmentsCount(0);
      setAssessmentRecords([]);
    }
  }, [cases, loadingCases]);

  /* =========================
     CASE FORM
  ========================= */

  const handleCreateCaseChange = (event) => {
    const { name, value } = event.target;

    setCaseForm((previous) => ({
      ...previous,
      [name]: value,
    }));
  };

  const handleCreateCase = async (event) => {
    event.preventDefault();

    setCreateCaseError("");

    if (!caseForm.title.trim()) {
      setCreateCaseError(
        "Please enter a case title."
      );
      return;
    }

    try {
      setCreatingCase(true);

      const response = await api.post("/cases", {
        title: caseForm.title.trim(),
        description:
          caseForm.description.trim(),
        status: "active",
      });

      const createdCase = response.data;

      setCases((previousCases) => [
        createdCase,
        ...previousCases,
      ]);

      setCaseForm({
        title: "",
        description: "",
      });

      setShowCreateCase(false);

      const createdCaseId =
        createdCase.id || createdCase._id;

      if (createdCaseId) {
        openCase(createdCaseId);
      }
    } catch (error) {
      console.error(
        "Failed to create case:",
        error
      );

      if (error.response?.data?.detail) {
        setCreateCaseError(
          error.response.data.detail
        );
      } else {
        setCreateCaseError(
          "Unable to create the case. Please try again."
        );
      }
    } finally {
      setCreatingCase(false);
    }
  };

  /* =========================
     FILE VALIDATION
  ========================= */

  const validateFile = (file) => {
    if (!file) {
      return "Please select an image.";
    }

    const allowedTypes = [
      "image/jpeg",
      "image/png",
      "image/webp",
    ];

    if (!allowedTypes.includes(file.type)) {
      return (
        "Only JPEG, PNG, and WEBP images are allowed."
      );
    }

    const maxSize = 10 * 1024 * 1024;

    if (file.size > maxSize) {
      return "Image size must be 10 MB or smaller.";
    }

    return "";
  };

  /* =========================
     SELECT FILE
  ========================= */

  const selectFile = (file) => {
    setUploadError("");
    setUploadSuccess("");

    const validationError =
      validateFile(file);

    if (validationError) {
      setSelectedFile(null);

      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }

      setPreviewUrl("");
      setUploadError(validationError);

      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(file);

    const newPreviewUrl =
      URL.createObjectURL(file);

    setPreviewUrl(newPreviewUrl);
  };

  const handleFileInput = (event) => {
    const file =
      event.target.files?.[0];

    if (file) {
      selectFile(file);
    }
  };

  /* =========================
     DRAG & DROP
  ========================= */

  const handleDragEnter = (event) => {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(false);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(true);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();

    setDragActive(false);

    const file =
      event.dataTransfer.files?.[0];

    if (file) {
      selectFile(file);
    }
  };

  /* =========================
     UPLOAD IMAGE
  ========================= */

  const uploadImage = async () => {
    if (!selectedCaseId) {
      setUploadError(
        "Please select a case first."
      );
      return;
    }

    if (!selectedFile) {
      setUploadError(
        "Please select an image first."
      );
      return;
    }

    const validationError =
      validateFile(selectedFile);

    if (validationError) {
      setUploadError(validationError);
      return;
    }

    try {
      setUploadingImage(true);
      setUploadError("");
      setUploadSuccess("");

      const formData = new FormData();

      formData.append("file", selectedFile);

      const response = await api.post(
        `/cases/${selectedCaseId}/images`,
        formData
      );

      const uploadedImage =
        response.data;

      setImages((previousImages) => [
        uploadedImage,
        ...previousImages,
      ]);

      const uploadedImageId =
        uploadedImage.id ||
        uploadedImage._id;

      if (uploadedImageId) {
        await loadImagePreview(
          selectedCaseId,
          uploadedImageId
        );
      }

      setSelectedFile(null);

      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }

      setPreviewUrl("");

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }

      setUploadSuccess(
        "Image uploaded successfully."
      );

      setShowUploadModal(false);

      const latestCases = await loadCases();

      await loadGlobalDashboardData(
        latestCases
      );
    } catch (error) {
      console.error(
        "Failed to upload image:",
        error
      );

      if (error.response?.data?.detail) {
        setUploadError(
          error.response.data.detail
        );
      } else {
        setUploadError(
          "Unable to upload the image. Please try again."
        );
      }
    } finally {
      setUploadingImage(false);
    }
  };

  /* =========================
     START ASSESSMENT
  ========================= */

  const handleStartAssessment = async () => {
    if (cases.length === 0) {
      setCreateCaseError("");
      setShowCreateCase(true);
      return;
    }

    let targetCase = null;

    if (selectedCaseId) {
      targetCase = cases.find(
        (caseItem) =>
          String(caseItem.id || caseItem._id) ===
          String(selectedCaseId)
      );
    }

    if (!targetCase) {
      targetCase = cases[0];
    }

    const targetCaseId =
      targetCase.id || targetCase._id;

    openCase(targetCaseId);

    try {
      const response = await api.get(
        `/cases/${targetCaseId}/images`
      );

      const caseImages =
        response.data || [];

      if (caseImages.length === 0) {
        setTimeout(() => {
          resetUpload();
          setShowUploadModal(true);
        }, 150);

        return;
      }

      const firstImage = caseImages[0];

      const imageId =
        firstImage.id ||
        firstImage._id;

      setTimeout(() => {
        preprocessImage(
          imageId,
          targetCaseId
        );
      }, 150);
    } catch (error) {
      console.error(
        "Unable to start assessment:",
        error
      );

      setImageError(
        error.response?.data?.detail ||
          "Unable to start the assessment."
      );
    }
  };

  /* =========================
     PREPROCESS IMAGE
  ========================= */

  const preprocessImage = async (
    imageId,
    caseIdOverride = null
  ) => {
    const caseId =
      caseIdOverride || selectedCaseId;

    if (!caseId || !imageId) {
      return;
    }

    try {
      setImageError("");

      setImages((previousImages) =>
        previousImages.map((image) =>
          (image.id || image._id) ===
          imageId
            ? {
                ...image,
                quality_status:
                  "processing",
              }
            : image
        )
      );

      await api.post(
        `/cases/${caseId}/images/${imageId}/preprocess`
      );

      const analysisResponse = await api.post(
         `/cases/${caseId}/images/${imageId}/analyze`
      );

      console.log("AI ANALYSIS RESULT:", analysisResponse.data);

      setImages((previousImages) =>
        previousImages.map((image) =>
          (image.id || image._id) ===
          imageId
            ? {
                ...image,
                quality_status:
                  "processed",
              }
            : image
        )
      );

      await loadImages(caseId);
      await loadAnalyses(caseId);

      const latestCases = await loadCases();

      await loadGlobalDashboardData(
        latestCases
      );
    } catch (error) {
      console.error(
        "Image preprocessing failed:",
        error
      );

      setImageError(
        error.response?.data?.detail ||
          "Image preprocessing failed."
      );

      await loadImages(caseId);
    }
  };

  /* =========================
     DELETE IMAGE
  ========================= */

  const deleteImage = async (imageId) => {
    if (!selectedCaseId || !imageId) {
      return;
    }

    const confirmed = window.confirm(
      "Are you sure you want to delete this image?"
    );

    if (!confirmed) {
      return;
    }

    try {
      await api.delete(
        `/cases/${selectedCaseId}/images/${imageId}`
      );

      setImages((previousImages) =>
        previousImages.filter(
          (image) =>
            (image.id || image._id) !==
            imageId
        )
      );

      setImageUrls((previous) => {
        const oldUrl =
          previous[imageId];

        if (oldUrl) {
          URL.revokeObjectURL(oldUrl);
        }

        const updated = {
          ...previous,
        };

        delete updated[imageId];

        return updated;
      });

      const latestCases = await loadCases();

      await loadGlobalDashboardData(
        latestCases
      );
    } catch (error) {
      console.error(
        "Failed to delete image:",
        error
      );

      setImageError(
        error.response?.data?.detail ||
          "Unable to delete the image."
      );
    }
  };

  /* =========================
     RESET UPLOAD
  ========================= */

  const resetUpload = () => {
    setSelectedFile(null);
    setUploadError("");
    setUploadSuccess("");

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setPreviewUrl("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  /* =========================
     OPEN UPLOAD
  ========================= */

  const openUpload = (
    caseId = selectedCaseId
  ) => {
    if (!caseId) {
      setSection("cases");
      return;
    }

    if (String(caseId) !== String(selectedCaseId)) {
      openCase(caseId);
    }

    resetUpload();
    setShowUploadModal(true);
  };

  /* =========================
     DATE
  ========================= */

  const formatDate = (dateValue) => {
    if (!dateValue) {
      return "Date unavailable";
    }

    const date = new Date(dateValue);

    if (Number.isNaN(date.getTime())) {
      return "Date unavailable";
    }

    return date.toLocaleDateString(
      "en-IN",
      {
        day: "2-digit",
        month: "short",
        year: "numeric",
      }
    );
  };

  const formatDateTime = (dateValue) => {
    if (!dateValue) {
      return "Date unavailable";
    }

    const date = new Date(dateValue);

    if (Number.isNaN(date.getTime())) {
      return "Date unavailable";
    }

    return date.toLocaleString(
      "en-IN",
      {
        day: "2-digit",
        month: "short",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      }
    );
  };

  const formatFileSize = (bytes) => {
    if (!bytes) {
      return "Unknown size";
    }

    if (bytes < 1024) {
      return `${bytes} B`;
    }

    if (bytes < 1024 * 1024) {
      return `${(
        bytes / 1024
      ).toFixed(1)} KB`;
    }

    return `${(
      bytes /
      (1024 * 1024)
    ).toFixed(2)} MB`;
  };

  /* =========================
     STATUS
  ========================= */

  const getNormalizedStatus = (status) => {
    return String(status || "")
      .trim()
      .toLowerCase()
      .replace(/_/g, "-")
      .replace(/\s+/g, "-");
  };

  const getStatusClass = (status) => {
    const normalizedStatus =
      getNormalizedStatus(status);

    if (
      normalizedStatus === "completed" ||
      normalizedStatus === "complete" ||
      normalizedStatus === "processed" ||
      normalizedStatus === "success" ||
      normalizedStatus === "successful"
    ) {
      return "status-completed";
    }

    if (
      normalizedStatus === "active" ||
      normalizedStatus === "in-progress" ||
      normalizedStatus === "processing" ||
      normalizedStatus === "running"
    ) {
      return "status-active";
    }

    if (
      normalizedStatus === "failed" ||
      normalizedStatus === "error"
    ) {
      return "status-failed";
    }

    return "status-pending";
  };

  const getStatusLabel = (status) => {
    if (!status) {
      return "Pending";
    }

    return String(status)
      .replace(/[_-]/g, " ")
      .replace(/\b\w/g, (letter) =>
        letter.toUpperCase()
      );
  };

  const isAnalysisCompleted = (analysis) => {
    if (!analysis) {
      return false;
    }

    const status =
      getNormalizedStatus(
        analysis.analysis_status
      );

    return (
      status === "completed" ||
      status === "complete" ||
      status === "success" ||
      status === "successful"
    );
  };

  /* =========================
     IMAGE URL
  ========================= */

  const getImageUrl = (image) => {
    if (!image) {
      return "";
    }

    const imageId =
      image.id || image._id;

    return imageUrls[imageId] || "";
  };

  /* =========================
     COUNTS
  ========================= */

  const totalImages = allImagesCount;

  const totalAssessments =
    allAssessmentsCount;

  const activeCases = useMemo(() => {
    return cases.filter((caseItem) => {
      const status = String(
        caseItem.status || ""
      ).toLowerCase();

      return (
        status === "active" ||
        status === "in-progress" ||
        status === "processing"
      );
    }).length;
  }, [cases]);

  /* =========================
     REPORT HELPERS
  ========================= */

  const getPrediction = (analysis) => {
    return (
      analysis?.predicted_condition ||
      analysis?.predicted_disease ||
      analysis?.prediction ||
      analysis?.predicted_class ||
      analysis?.diagnosis_prediction ||
      "Not available"
    );
  };

  const getConfidence = (analysis) => {
    const value =
      analysis?.confidence ??
      analysis?.confidence_score ??
      analysis?.prediction_confidence;

    if (
      value === null ||
      value === undefined ||
      value === ""
    ) {
      return "Not available";
    }

    const numericValue = Number(value);

    if (Number.isNaN(numericValue)) {
      return String(value);
    }

    const percentage =
      numericValue <= 1
        ? numericValue * 100
        : numericValue;

    return `${percentage.toFixed(2)}%`;
  };

  const getRiskCategory = (analysis) => {
    return (
      analysis?.risk_category ||
      analysis?.risk_assessment?.risk_category ||
      analysis?.risk_level ||
      "Not available"
    );
  };

  // Risk assessment fields may be returned flat or nested by the API.
  const getReviewPriority = (analysis) =>
    analysis?.review_priority ||
    analysis?.risk_assessment?.review_priority ||
    "Not available";

  const getRiskMessage = (analysis) =>
    analysis?.risk_message ||
    analysis?.risk_assessment?.message ||
    "Professional assessment is recommended.";

  const getRiskDisclaimer = (analysis) =>
    analysis?.medical_disclaimer ||
    analysis?.risk_assessment?.disclaimer ||
    "AI results are not a medical diagnosis. Consult a qualified dermatologist.";

  const getModelVersion = (analysis) => {
    return (
      analysis?.model_version ||
      analysis?.model ||
      analysis?.model_name ||
      "Not available"
    );
  };

  const getQualityResult = (
    analysis,
    relatedImage = null
  ) => {
    return (
      analysis?.quality_result ||
      analysis?.image_quality ||
      analysis?.quality_status ||
      relatedImage?.quality_status ||
      "Not available"
    );
  };

  const getLesionDetails = (analysis) => {
    return (
      analysis?.lesion_details ||
      analysis?.lesion_detection ||
      analysis?.detected_lesion ||
      analysis?.lesion ||
      "Not available"
    );
  };

  const getSummary = (analysis) => {
    return (
      analysis?.analysis_summary ||
      analysis?.summary ||
      analysis?.description ||
      "Analysis summary is not available."
    );
  };

  const getTopPredictions = (analysis) => {
    const predictions =
      analysis?.top_predictions ||
      analysis?.top_n_predictions ||
      analysis?.predictions;

    if (!Array.isArray(predictions)) {
      return [];
    }

    return predictions;
  };

  const getExplainabilityUrl = (analysis) => {
    return (
      analysis?.gradcam_url ||
      analysis?.grad_cam_url ||
      analysis?.attention_map_url ||
      analysis?.explainability_url ||
      ""
    );
  };

  /* =========================
   LOAD GRAD-CAM IMAGE
========================= */

useEffect(() => {
  let cancelled = false;
  let objectUrl = null;

  const loadGradCAM = async () => {
    const gradcamUrl =
      selectedAnalysis?.gradcam_url ||
      selectedAnalysis?.grad_cam_url ||
      selectedAnalysis?.attention_map_url ||
      selectedAnalysis?.explainability_url ||
      "";

    setExplainabilityImageUrl("");
    setExplainabilityError("");

    if (!showReportModal || !gradcamUrl) {
      setExplainabilityLoading(false);
      return;
    }

    try {
      setExplainabilityLoading(true);

      console.log(
        "[DermAI] Loading Grad-CAM:",
        gradcamUrl
      );

      const response = await api.get(
        gradcamUrl,
        {
          responseType: "blob",
        }
      );

      if (
        !response.data ||
        response.data.size === 0
      ) {
        throw new Error(
          "Grad-CAM image response is empty."
        );
      }

      objectUrl = URL.createObjectURL(
        response.data
      );

      if (!cancelled) {
        setExplainabilityImageUrl(
          objectUrl
        );
      } else {
        URL.revokeObjectURL(objectUrl);
      }

    } catch (error) {
      console.error(
        "[DermAI] Grad-CAM loading failed:",
        error.response?.status,
        error.response?.data || error.message
      );

      if (!cancelled) {
        setExplainabilityError(
          "Unable to load the Grad-CAM visualization."
        );
      }

    } finally {
      if (!cancelled) {
        setExplainabilityLoading(false);
      }
    }
  };

  loadGradCAM();

  return () => {
    cancelled = true;

    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
    }
  };

}, [selectedAnalysis, showReportModal]);

  const getAnalysisDate = (analysis) => {
    return (
      analysis?.completed_at ||
      analysis?.analysis_date ||
      analysis?.started_at ||
      null
    );
  };

  /* =========================
     FIND RELATED IMAGE
  ========================= */

  const getRelatedImage = (analysis) => {
    if (!analysis) {
      return null;
    }

    const analysisImageId =
      analysis.image_id ||
      analysis.imageId;

    if (!analysisImageId) {
      return images[0] || null;
    }

    return (
      images.find(
        (image) =>
          String(image.id || image._id) ===
          String(analysisImageId)
      ) || null
    );
  };

  /* =========================
     OPEN REPORT
  ========================= */

  const openReport = (analysis) => {
    if (!analysis) {
      return;
    }

    setSelectedAnalysis(analysis);
    setShowReportModal(true);
  };

  const closeReport = () => {
    setShowReportModal(false);
    setSelectedAnalysis(null);
  };

  /* =========================
     REPORT EXPORT
  ========================= */

  const buildReportData = (analysis) => {
    const relatedImage =
      getRelatedImage(analysis);

    return {
      case_information: {
        case_id:
          analysis?.caseId ||
          selectedCaseId ||
          "Not available",
        case_title:
          analysis?.caseTitle ||
          selectedCase?.title ||
          "Not available",
        image_id:
          analysis?.image_id ||
          relatedImage?.id ||
          relatedImage?._id ||
          "Not available",
        image_name:
          relatedImage?.original_filename ||
          "Not available",
        analysis_date:
          getAnalysisDate(analysis),
      },

      quality_and_detection: {
        image_quality:
          getQualityResult(
            analysis,
            relatedImage
          ),
        lesion_details:
          getLesionDetails(analysis),
      },

      prediction_results: {
        predicted_condition:
          getPrediction(analysis),
        confidence:
          getConfidence(analysis),
        top_predictions:
          getTopPredictions(analysis),
        risk_category:
          getRiskCategory(analysis),
      },

      visual_explanations: {
        explainability_available:
          Boolean(
            getExplainabilityUrl(analysis)
          ),
        explainability_url:
          getExplainabilityUrl(analysis) ||
          null,
      },

      metadata_and_summary: {
        model_version:
          getModelVersion(analysis),
        preprocessing_status:
          analysis?.preprocessing_status ||
          "Not available",
        analysis_status:
          analysis?.analysis_status ||
          "Not available",
        summary:
          getSummary(analysis),
      },

      disclaimer:
        "This output is an AI-assisted analysis and is not a standalone medical diagnosis. Results should be reviewed and interpreted appropriately by a qualified healthcare professional.",
    };
  };

  const downloadJsonReport = (analysis) => {
    const reportData =
      buildReportData(analysis);

    const blob = new Blob(
      [
        JSON.stringify(
          reportData,
          null,
          2
        ),
      ],
      {
        type: "application/json",
      }
    );

    const url =
      URL.createObjectURL(blob);

    const link =
      document.createElement("a");

    link.href = url;

    link.download = `dermai-report-${
      analysis?.caseId ||
      selectedCaseId ||
      "case"
    }.json`;

    document.body.appendChild(link);

    link.click();

    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  };

  const downloadCsvReport = (analysis) => {
    const reportData =
      buildReportData(analysis);

    const rows = [
      ["Section", "Field", "Value"],

      [
        "Case Information",
        "Case ID",
        reportData.case_information.case_id,
      ],

      [
        "Case Information",
        "Case Title",
        reportData.case_information.case_title,
      ],

      [
        "Case Information",
        "Image ID",
        reportData.case_information.image_id,
      ],

      [
        "Case Information",
        "Image Name",
        reportData.case_information.image_name,
      ],

      [
        "Case Information",
        "Analysis Date",
        reportData.case_information.analysis_date ||
          "Not available",
      ],

      [
        "Quality & Detection",
        "Image Quality",
        reportData.quality_and_detection.image_quality,
      ],

      [
        "Quality & Detection",
        "Lesion Details",
        reportData.quality_and_detection.lesion_details,
      ],

      [
        "Prediction Results",
        "Predicted Condition",
        reportData.prediction_results.predicted_condition,
      ],

      [
        "Prediction Results",
        "Confidence",
        reportData.prediction_results.confidence,
      ],

      [
        "Prediction Results",
        "Risk Category",
        reportData.prediction_results.risk_category,
      ],

      [
        "Metadata & Summary",
        "Model Version",
        reportData.metadata_and_summary.model_version,
      ],

      [
        "Metadata & Summary",
        "Preprocessing Status",
        reportData.metadata_and_summary.preprocessing_status,
      ],

      [
        "Metadata & Summary",
        "Analysis Status",
        reportData.metadata_and_summary.analysis_status,
      ],

      [
        "Metadata & Summary",
        "Summary",
        reportData.metadata_and_summary.summary,
      ],

      [
        "Disclaimer",
        "Statement",
        reportData.disclaimer,
      ],
    ];

    const escapeCsvValue = (value) => {
      const stringValue =
        value === null ||
        value === undefined
          ? ""
          : String(value);

      return `"${stringValue.replace(
        /"/g,
        '""'
      )}"`;
    };

    const csv = rows
      .map((row) =>
        row
          .map(escapeCsvValue)
          .join(",")
      )
      .join("\n");

    const blob = new Blob(
      [csv],
      {
        type: "text/csv;charset=utf-8;",
      }
    );

    const url =
      URL.createObjectURL(blob);

    const link =
      document.createElement("a");

    link.href = url;

    link.download = `dermai-report-${
      analysis?.caseId ||
      selectedCaseId ||
      "case"
    }.csv`;

    document.body.appendChild(link);

    link.click();

    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  };

  const exportPdfReport = () => {
    window.print();
  };

  /* =========================
     NOTIFICATIONS
  ========================= */

  const notifications = useMemo(() => {
    const generated = [];

    assessmentRecords.forEach(
      (analysis) => {
        if (
          isAnalysisCompleted(analysis)
        ) {
          generated.push({
            id: `completed-${
              analysis.id ||
              analysis._id
            }`,
            level: "informational",
            title: "Analysis completed",
            message:
              `DermAI analysis for ${
                analysis.caseTitle ||
                "your case"
              } has been completed.`,
            analysis,
          });

          generated.push({
            id: `report-${
              analysis.id ||
              analysis._id
            }`,
            level: "informational",
            title: "Report generated",
            message:
              "The analysis report is available for review and export.",
            analysis,
          });
        }

        const confidenceValue =
          Number(
            analysis.confidence ??
              analysis.confidence_score
          );

        if (
          !Number.isNaN(confidenceValue)
        ) {
          const percentage =
            confidenceValue <= 1
              ? confidenceValue * 100
              : confidenceValue;

          if (percentage < 60) {
            generated.push({
              id: `confidence-${
                analysis.id ||
                analysis._id
              }`,
              level: "medium",
              title:
                "Low-confidence prediction",
              message:
                "The reported prediction confidence is below the configured threshold.",
              analysis,
            });
          }
        }

        const risk =
          getNormalizedStatus(
            analysis.risk_category ||
              analysis.risk_level
          );

        if (
          risk === "high" ||
          risk === "critical"
        ) {
          generated.push({
            id: `risk-${
              analysis.id ||
              analysis._id
            }`,
            level:
              risk === "critical"
                ? "critical"
                : "high",
            title:
              "High-risk prediction",
            message:
              "A high-risk category was returned by the analysis record. Clinical review is recommended.",
            analysis,
          });

          generated.push({
            id: `clinical-review-${
              analysis.id ||
              analysis._id
            }`,
            level: "high",
            title:
              "Clinical review recommended",
            message:
              "Please consider appropriate professional review of the AI-assisted result.",
            analysis,
          });
        }
      }
    );

    images.forEach((image) => {
      const quality =
        getNormalizedStatus(
          image.quality_status
        );

      if (
        quality === "poor" ||
        quality === "low" ||
        quality === "failed"
      ) {
        generated.push({
          id: `quality-${
            image.id ||
            image._id
          }`,
          level: "medium",
          title:
            "Poor image quality",
          message:
            "The uploaded image may not meet the expected quality requirements for reliable analysis.",
          analysis: null,
        });
      }
    });

    return generated.slice(0, 12);
  }, [
    assessmentRecords,
    images,
  ]);

  /* =========================
     REFRESH DASHBOARD
  ========================= */

  const refreshDashboard = async () => {
    try {
      setDashboardLoading(true);

      const latestCases =
        await loadCases();

      await loadGlobalDashboardData(
        latestCases
      );

      if (selectedCaseId) {
        await loadCaseDetail(
          selectedCaseId
        );

        await loadImages(
          selectedCaseId
        );

        await loadAnalyses(
          selectedCaseId
        );
      }
    } finally {
      setDashboardLoading(false);
    }
  };

  /* =========================
     CLEANUP BLOB URLS
  ========================= */

  useEffect(() => {
    return () => {
      Object.values(imageUrls).forEach(
        (url) => {
          if (url) {
            URL.revokeObjectURL(url);
          }
        }
      );

      if (previewUrl) {
        URL.revokeObjectURL(
          previewUrl
        );
      }
    };
  }, []);

  /* =========================
     DASHBOARD VIEW
  ========================= */

  const renderDashboard = () => (
    <>
      <section className="welcome-section">
        <div>
          <p className="eyebrow">
            PATIENT DASHBOARD
          </p>

          <h1>
            Welcome back, {firstName}
            <span className="wave">
              👋
            </span>
          </h1>

          <p className="welcome-text">
            Manage your skin assessment
            cases and track your DermAI
            analysis activity.
          </p>
        </div>

        <button
          className="primary-button"
          onClick={() => {
            setCreateCaseError("");
            setShowCreateCase(true);
          }}
        >
          <span>+</span>
          New Case
        </button>
      </section>

      <section className="stats-grid">
        <div
          className="stat-card clickable-stat"
          onClick={() =>
            setSection("cases")
          }
        >
          <div className="stat-icon cases-icon">
            ▣
          </div>

          <div className="stat-content">
            <span className="stat-label">
              MY CASES
            </span>

            <strong>
              {loadingCases
                ? "..."
                : cases.length}
            </strong>

            <p>
              {activeCases} active
            </p>
          </div>
        </div>

        <div
          className="stat-card clickable-stat"
          onClick={() =>
            setSection("cases")
          }
        >
          <div className="stat-icon image-icon">
            ◉
          </div>

          <div className="stat-content">
            <span className="stat-label">
              IMAGES UPLOADED
            </span>

            <strong>
              {loadingCases
                ? "..."
                : totalImages}
            </strong>

            <p>
              Across all cases
            </p>
          </div>
        </div>

        <div
          className="stat-card clickable-stat"
          onClick={() =>
            setSection("assessments")
          }
        >
          <div className="stat-icon analysis-icon">
            ✦
          </div>

          <div className="stat-content">
            <span className="stat-label">
              ASSESSMENTS
            </span>

            <strong>
              {loadingCases
                ? "..."
                : totalAssessments}
            </strong>

            <p>
              AI analysis records
            </p>
          </div>
        </div>
      </section>

      <section className="dashboard-grid">
        <div className="cases-panel dashboard-card">
          <div className="card-header">
            <div>
              <h2>Your Cases</h2>

              <p>
                View and manage your skin
                assessment cases.
              </p>
            </div>

            <button
              className="text-button"
              onClick={() =>
                setSection("cases")
              }
            >
              View all →
            </button>
          </div>

          {loadingCases ? (
            <div className="loading-state">
              <div className="loading-spinner"></div>
              <p>
                Loading your cases...
              </p>
            </div>
          ) : caseError ? (
            <div className="error-state">
              <div className="error-state-icon">
                !
              </div>

              <h3>
                Unable to load cases
              </h3>

              <p>{caseError}</p>

              <button
                className="secondary-button"
                onClick={loadCases}
              >
                Try Again
              </button>
            </div>
          ) : cases.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">
                ▣
              </div>

              <h3>
                No cases yet
              </h3>

              <p>
                Create your first case to
                begin organizing your skin
                assessment.
              </p>

              <button
                className="secondary-button"
                onClick={() => {
                  setCreateCaseError("");
                  setShowCreateCase(
                    true
                  );
                }}
              >
                + Create Your First Case
              </button>
            </div>
          ) : (
            <div className="cases-list">
              {cases
                .slice(0, 4)
                .map((caseItem) => {
                  const caseId =
                    caseItem.id ||
                    caseItem._id;

                  return (
                    <div
                      className="case-row"
                      key={caseId}
                    >
                      <div className="case-main">
                        <div className="case-icon">
                          ▣
                        </div>

                        <div className="case-details">
                          <h3>
                            {caseItem.title ||
                              "Untitled Case"}
                          </h3>

                          <p>
                            Created{" "}
                            {formatDate(
                              caseItem.created_at
                            )}
                          </p>
                        </div>
                      </div>

                      <div className="case-right">
                        <span
                          className={`case-status ${getStatusClass(
                            caseItem.status
                          )}`}
                        >
                          <span className="status-dot"></span>

                          {getStatusLabel(
                            caseItem.status
                          )}
                        </span>

                        <button
                          className="case-view-button"
                          onClick={() =>
                            openCase(
                              caseId
                            )
                          }
                        >
                          View →
                        </button>
                      </div>
                    </div>
                  );
                })}
            </div>
          )}
        </div>

        <div className="quick-panel dashboard-card">
          <div className="card-header">
            <div>
              <h2>
                Quick Actions
              </h2>

              <p>
                Start your next DermAI
                workflow.
              </p>
            </div>
          </div>

          <div className="quick-actions">
            <button
              className="quick-action"
              onClick={() =>
                openUpload(
                  selectedCaseId ||
                    cases[0]?.id ||
                    cases[0]?._id
                )
              }
            >
              <div className="quick-icon upload-icon">
                ↑
              </div>

              <div>
                <strong>
                  Upload Image
                </strong>

                <span>
                  Add a skin image to a case
                </span>
              </div>

              <span className="arrow">
                →
              </span>
            </button>

            <button
              className="quick-action"
              onClick={
                handleStartAssessment
              }
            >
              <div className="quick-icon assessment-action-icon">
                ✦
              </div>

              <div>
                <strong>
                  Start Assessment
                </strong>

                <span>
                  Begin image preprocessing
                </span>
              </div>

              <span className="arrow">
                →
              </span>
            </button>

            <button
              className="quick-action"
              onClick={() => {
                setCreateCaseError("");
                setShowCreateCase(true);
              }}
            >
              <div className="quick-icon case-action-icon">
                +
              </div>

              <div>
                <strong>
                  Create Case
                </strong>

                <span>
                  Start a new assessment case
                </span>
              </div>

              <span className="arrow">
                →
              </span>
            </button>

            <button
              className="quick-action"
              onClick={() =>
                setSection("assessments")
              }
            >
              <div className="quick-icon history-icon">
                ◷
              </div>

              <div>
                <strong>
                  Assessment History
                </strong>

                <span>
                  Review previous analyses
                </span>
              </div>

              <span className="arrow">
                →
              </span>
            </button>
          </div>
        </div>
      </section>

      <section className="getting-started dashboard-card">
        <div className="getting-started-content">
          <div className="getting-started-icon">
            ✦
          </div>

          <div>
            <p className="eyebrow">
              GETTING STARTED
            </p>

            <h2>
              Begin your skin assessment
            </h2>

            <p>
              Create a case, upload a clear
              skin image, and prepare it for
              DermAI's image analysis
              pipeline.
            </p>
          </div>
        </div>

        <button
          className="primary-button"
          onClick={
            handleStartAssessment
          }
        >
          Start Assessment
          <span>→</span>
        </button>
      </section>
    </>
  );

  /* =========================
     CASES VIEW
  ========================= */

  const renderCases = () => (
    <section className="page-section">
      <div className="page-title-row">
        <div>
          <p className="eyebrow">
            CASE MANAGEMENT
          </p>

          <h1>My Cases</h1>

          <p>
            Manage your skin assessment cases
            and uploaded images.
          </p>
        </div>

        <button
          className="primary-button"
          onClick={() =>
            setShowCreateCase(true)
          }
        >
          <span>+</span>
          New Case
        </button>
      </div>

      {loadingCases ? (
        <div className="large-loading">
          <div className="loading-spinner"></div>
          <p>Loading cases...</p>
        </div>
      ) : caseError ? (
        <div className="full-error">
          <h3>
            Unable to load cases
          </h3>

          <p>{caseError}</p>

          <button
            className="secondary-button"
            onClick={loadCases}
          >
            Try Again
          </button>
        </div>
      ) : cases.length === 0 ? (
        <div className="large-empty">
          <div className="large-empty-icon">
            ▣
          </div>

          <h2>No cases yet</h2>

          <p>
            Create your first case to start
            using DermAI.
          </p>

          <button
            className="primary-button"
            onClick={() =>
              setShowCreateCase(true)
            }
          >
            Create Case
          </button>
        </div>
      ) : (
        <div className="all-cases-grid">
          {cases.map((caseItem) => {
            const caseId =
              caseItem.id ||
              caseItem._id;

            return (
              <div
                className="case-card"
                key={caseId}
              >
                <div className="case-card-top">
                  <div className="large-case-icon">
                    ▣
                  </div>

                  <span
                    className={`case-status ${getStatusClass(
                      caseItem.status
                    )}`}
                  >
                    <span className="status-dot"></span>

                    {getStatusLabel(
                      caseItem.status
                    )}
                  </span>
                </div>

                <h3>
                  {caseItem.title ||
                    "Untitled Case"}
                </h3>

                <p className="case-description">
                  {caseItem.description ||
                    "No description provided."}
                </p>

                <div className="case-card-meta">
                  <span>
                    Created
                  </span>

                  <strong>
                    {formatDate(
                      caseItem.created_at
                    )}
                  </strong>
                </div>

                <div className="case-card-actions">
                  <button
                    className="secondary-button"
                    onClick={() =>
                      openCase(caseId)
                    }
                  >
                    View Details
                  </button>

                  <button
                    className="outline-action"
                    onClick={() =>
                      openUpload(caseId)
                    }
                  >
                    Upload Image
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );

  /* =========================
     CASE DETAIL VIEW
  ========================= */

  const renderCaseDetail = () => {
    if (loadingCaseDetail) {
      return (
        <div className="large-loading">
          <div className="loading-spinner"></div>
          <p>
            Loading case details...
          </p>
        </div>
      );
    }

    if (
      caseDetailError ||
      !selectedCase
    ) {
      return (
        <div className="full-error">
          <h3>
            Unable to open case
          </h3>

          <p>
            {caseDetailError ||
              "Case could not be found."}
          </p>

          <button
            className="secondary-button"
            onClick={() =>
              setSection("cases")
            }
          >
            Back to My Cases
          </button>
        </div>
      );
    }

    return (
      <section className="page-section">
        <button
          className="back-button"
          onClick={() =>
            setSection("cases")
          }
        >
          ← Back to My Cases
        </button>

        <div className="case-detail-header">
          <div>
            <p className="eyebrow">
              CASE DETAILS
            </p>

            <h1>
              {selectedCase.title ||
                "Untitled Case"}
            </h1>

            <p>
              {selectedCase.description ||
                "No description provided."}
            </p>
          </div>

          <div className="case-detail-actions">
            <span
              className={`case-status ${getStatusClass(
                selectedCase.status
              )}`}
            >
              <span className="status-dot"></span>

              {getStatusLabel(
                selectedCase.status
              )}
            </span>

            <button
              className="primary-button"
              onClick={() =>
                openUpload(
                  selectedCaseId
                )
              }
            >
              ↑ Upload Image
            </button>
          </div>
        </div>

        <div className="case-detail-info">
          <div>
            <span>Created</span>

            <strong>
              {formatDate(
                selectedCase.created_at
              )}
            </strong>
          </div>

          <div>
            <span>Images</span>

            <strong>
              {images.length}
            </strong>
          </div>

          <div>
            <span>Assessments</span>

            <strong>
              {analyses.length}
            </strong>
          </div>
        </div>

        <section className="clinical-note-card">
          <div className="clinical-note-heading">
            <div>
              <h2>Clinical Note</h2>
              <p>Doctor's clinical observation or recommendation for this case.</p>
            </div>
            <span className="clinical-note-badge">Doctor Review</span>
          </div>

          <div className="clinical-note-content">
            {selectedCase.clinical_note ||
            selectedCase.doctor_note ||
            selectedCase.clinical_notes ? (
              <p>
                {selectedCase.clinical_note ||
                  selectedCase.doctor_note ||
                  selectedCase.clinical_notes}
              </p>
            ) : (
              <p className="clinical-note-empty">
                No clinical note has been added yet.
              </p>
            )}
          </div>
        </section>

        <div className="detail-section-header">
          <div>
            <h2>Case Images</h2>

            <p>
              Uploaded skin images associated
              with this case.
            </p>
          </div>

          <button
            className="secondary-button"
            onClick={() =>
              openUpload(
                selectedCaseId
              )
            }
          >
            + Add Image
          </button>
        </div>

        {loadingImages ? (
          <div className="large-loading compact">
            <div className="loading-spinner"></div>
            <p>
              Loading images...
            </p>
          </div>
        ) : imageError ? (
          <div className="full-error">
            <p>{imageError}</p>

            <button
              className="secondary-button"
              onClick={() =>
                loadImages(
                  selectedCaseId
                )
              }
            >
              Retry
            </button>
          </div>
        ) : images.length === 0 ? (
          <div className="large-empty compact">
            <div className="large-empty-icon">
              ◉
            </div>

            <h2>
              No images uploaded
            </h2>

            <p>
              Upload a clear skin image to
              begin the preprocessing
              workflow.
            </p>

            <button
              className="primary-button"
              onClick={() =>
                openUpload(
                  selectedCaseId
                )
              }
            >
              Upload Skin Image
            </button>
          </div>
        ) : (
          <div className="image-grid">
            {images.map((image) => {
              const imageId =
                image.id ||
                image._id;

              const imageUrl =
                getImageUrl(image);

              return (
                <div
                  className="image-card"
                  key={imageId}
                >
                  <div className="image-preview">
                    {imageUrl ? (
                      <img
                        src={imageUrl}
                        alt={
                          image.original_filename ||
                          "Uploaded skin image"
                        }
                      />
                    ) : (
                      <div className="image-placeholder">
                        <span>◉</span>

                        <p>
                          Loading image...
                        </p>
                      </div>
                    )}
                  </div>

                  <div className="image-card-body">
                    <h3>
                      {image.original_filename ||
                        "Uploaded Image"}
                    </h3>

                    <p>
                      {formatFileSize(
                        image.file_size
                      )}{" "}
                      · {image.width} ×{" "}
                      {image.height}
                    </p>

                    <div className="image-status-row">
                      <span
                        className={`case-status ${getStatusClass(
                          image.quality_status
                        )}`}
                      >
                        <span className="status-dot"></span>

                        {getStatusLabel(
                          image.quality_status
                        )}
                      </span>
                    </div>

                    <div className="image-actions">
                      <button
                        className="secondary-button small"
                        onClick={() =>
                          preprocessImage(
                            imageId
                          )
                        }
                        disabled={
                          image.quality_status ===
                          "processing"
                        }
                      >
                        {image.quality_status ===
                        "processing"
                          ? "Processing..."
                          : "Start Assessment"}
                      </button>

                      <button
                        className="delete-button"
                        onClick={() =>
                          deleteImage(
                            imageId
                          )
                        }
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        <div className="detail-section-header assessment-header">
          <div>
            <h2>
              Assessment Activity
            </h2>

            <p>
              Analysis and preprocessing
              activity for this case.
            </p>
          </div>
        </div>

        {loadingAnalyses ? (
          <div className="large-loading compact">
            <div className="loading-spinner"></div>
            <p>
              Loading assessments...
            </p>
          </div>
        ) : analyses.length === 0 ? (
          <div className="activity-empty">
            <span>✦</span>

            <div>
              <strong>
                No assessment activity yet
              </strong>

              <p>
                Start an assessment on an
                uploaded image to begin the
                preprocessing workflow.
              </p>
            </div>
          </div>
        ) : (
          <div className="activity-list">
            {analyses.map((analysis) => {
              const completed =
                isAnalysisCompleted(
                  analysis
                );

              return (
                <div
                  className={`activity-row ${
                    completed
                      ? "activity-completed"
                      : ""
                  }`}
                  key={
                    analysis.id ||
                    analysis._id
                  }
                >
                  <div className="activity-icon">
                    {completed
                      ? "✓"
                      : "✦"}
                  </div>

                  <div className="activity-main">
                    <strong>
                      {completed
                        ? "Analysis Completed"
                        : "Image Analysis"}
                    </strong>

                    <p>
                      Preprocessing:{" "}
                      {getStatusLabel(
                        analysis.preprocessing_status
                      )}
                      {" · "}
                      Analysis:{" "}
                      {getStatusLabel(
                        analysis.analysis_status
                      )}
                    </p>

                    {completed && (
                      <span className="completed-label">
                        AI-assisted analysis is ready
                        for review.
                      </span>
                    )}
                  </div>

                  <div className="activity-actions">
                    <span>
                      {formatDateTime(
                        analysis.started_at ||
                          analysis.completed_at
                      )}
                    </span>

                    {completed && (
                      <button
                        className="secondary-button small"
                        onClick={() =>
                          openReport(
                            {
                              ...analysis,
                              caseId:
                                selectedCaseId,
                              caseTitle:
                                selectedCase?.title ||
                                "Untitled Case",
                            }
                          )
                        }
                      >
                        View Report
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>
    );
  };

  /* =========================
     ASSESSMENTS VIEW
  ========================= */

  const renderAssessments = () => (
    <section className="page-section">
      <div className="page-title-row">
        <div>
          <p className="eyebrow">
            AI ANALYSIS
          </p>

          <h1>Assessments</h1>

          <p>
            Review your DermAI image-analysis
            activity.
          </p>
        </div>

        <button
          className="primary-button"
          onClick={
            handleStartAssessment
          }
        >
          Start Assessment
          <span>→</span>
        </button>
      </div>

      {loadingCases ? (
        <div className="large-loading">
          <div className="loading-spinner"></div>
          <p>
            Loading assessments...
          </p>
        </div>
      ) : assessmentRecords.length ===
        0 ? (
        <div className="large-empty">
          <div className="large-empty-icon">
            ✦
          </div>

          <h2>
            No assessments yet
          </h2>

          <p>
            Upload and preprocess a skin
            image from one of your cases
            to begin the analysis workflow.
          </p>

          <button
            className="primary-button"
            onClick={() =>
              setSection("cases")
            }
          >
            Go to My Cases →
          </button>
        </div>
      ) : (
        <div className="assessment-list">
          {assessmentRecords.map(
            (analysis) => {
              const completed =
                isAnalysisCompleted(
                  analysis
                );

              const reportAnalysis = {
                ...analysis,
                caseId:
                  analysis.caseId,
                caseTitle:
                  analysis.caseTitle,
              };

              return (
                <div
                  className={`assessment-card ${
                    completed
                      ? "assessment-completed"
                      : ""
                  }`}
                  key={`${analysis.caseId}-${
                    analysis.id ||
                    analysis._id
                  }`}
                >
                  <div className="assessment-icon">
                    {completed
                      ? "✓"
                      : "✦"}
                  </div>

                  <div className="assessment-main">
                    <h3>
                      {completed
                        ? "Analysis Completed"
                        : "DermAI Image Assessment"}
                    </h3>

                    <p className="assessment-case-name">
                      Case:{" "}
                      <strong>
                        {analysis.caseTitle}
                      </strong>
                    </p>

                    <p>
                      Preprocessing status:{" "}
                      <strong>
                        {getStatusLabel(
                          analysis.preprocessing_status
                        )}
                      </strong>
                    </p>

                    <p>
                      Analysis status:{" "}
                      <strong
                        className={
                          completed
                            ? "analysis-completed-text"
                            : ""
                        }
                      >
                        {completed
                          ? "Completed"
                          : getStatusLabel(
                              analysis.analysis_status
                            )}
                      </strong>
                    </p>

                    {completed && (
                      <span className="assessment-ready-text">
                        Report available for review
                      </span>
                    )}
                  </div>

                  <div className="assessment-card-right">
                    <div className="assessment-date">
                      {formatDateTime(
                        analysis.completed_at ||
                          analysis.started_at
                      )}
                    </div>

                    {completed && (
                      <button
                        className="secondary-button small"
                        onClick={() =>
                          openReport(
                            reportAnalysis
                          )
                        }
                      >
                        View Report
                      </button>
                    )}

                    <button
                      className="text-button"
                      onClick={() =>
                        openCase(
                          analysis.caseId
                        )
                      }
                    >
                      View Case →
                    </button>
                  </div>
                </div>
              );
            }
          )}
        </div>
      )}
    </section>
  );

  /* =========================
     PROFILE
  ========================= */

  const renderProfile = () => (
    <section className="page-section">
      <div className="page-title-row">
        <div>
          <p className="eyebrow">
            ACCOUNT
          </p>

          <h1>My Profile</h1>

          <p>
            Your DermAI account information.
          </p>
        </div>
      </div>

      <div className="profile-card">
        <div className="profile-large-avatar">
          {displayName
            .charAt(0)
            .toUpperCase()}
        </div>

        <div className="profile-fields">
          <div className="profile-field">
            <span>Full Name</span>

            <strong>
              {user?.full_name ||
                "Not available"}
            </strong>
          </div>

          <div className="profile-field">
            <span>Email</span>

            <strong>
              {user?.email ||
                "Not available"}
            </strong>
          </div>

          <div className="profile-field">
            <span>Account Role</span>

            <strong>
              {user?.role || "patient"}
            </strong>
          </div>

          <div className="profile-field">
            <span>Account Status</span>

            <strong className="profile-active">
              Active
            </strong>
          </div>
        </div>

        <div className="profile-note">
          <span>i</span>

          <p>
            DermAI is an AI-assisted
            research and educational
            platform. Results should not
            be treated as a medical
            diagnosis.
          </p>
        </div>
      </div>
    </section>
  );

  /* =========================
     UPLOAD MODAL
  ========================= */

  const renderUploadModal = () => {
    if (!showUploadModal) {
      return null;
    }

    return (
      <div
        className="modal-overlay"
        onClick={() => {
          if (!uploadingImage) {
            setShowUploadModal(false);
            resetUpload();
          }
        }}
      >
        <div
          className="upload-modal"
          onClick={(event) =>
            event.stopPropagation()
          }
        >
          <div className="modal-header">
            <div>
              <p className="eyebrow">
                IMAGE UPLOAD
              </p>

              <h2>
                Upload Skin Image
              </h2>

              <p>
                Add a clear image for this
                case.
              </p>
            </div>

            <button
              className="modal-close"
              onClick={() => {
                setShowUploadModal(false);
                resetUpload();
              }}
              disabled={uploadingImage}
            >
              ×
            </button>
          </div>

          <div className="upload-body">
            {!selectedFile ? (
              <div
                className={`drop-zone ${
                  dragActive
                    ? "drop-zone-active"
                    : ""
                }`}
                onDragEnter={
                  handleDragEnter
                }
                onDragLeave={
                  handleDragLeave
                }
                onDragOver={
                  handleDragOver
                }
                onDrop={handleDrop}
                onClick={() =>
                  fileInputRef.current?.click()
                }
              >
                <div className="drop-icon">
                  ↑
                </div>

                <h3>
                  Drag & drop your image here
                </h3>

                <p>
                  or click to browse from
                  your computer
                </p>

                <span>
                  JPEG, PNG or WEBP ·
                  Maximum 10 MB
                </span>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={
                    handleFileInput
                  }
                  hidden
                />
              </div>
            ) : (
              <div className="selected-file">
                <div className="preview-container">
                  {previewUrl && (
                    <img
                      src={previewUrl}
                      alt="Selected skin image"
                    />
                  )}
                </div>

                <div className="selected-file-info">
                  <h3>
                    {selectedFile.name}
                  </h3>

                  <p>
                    {formatFileSize(
                      selectedFile.size
                    )}
                  </p>

                  <button
                    className="change-file-button"
                    onClick={() =>
                      fileInputRef.current?.click()
                    }
                  >
                    Choose another image
                  </button>

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    onChange={
                      handleFileInput
                    }
                    hidden
                  />
                </div>
              </div>
            )}

            {uploadError && (
              <div className="modal-error">
                {uploadError}
              </div>
            )}

            {uploadSuccess && (
              <div className="modal-success">
                {uploadSuccess}
              </div>
            )}
          </div>

          <div className="modal-actions">
            <button
              className="modal-cancel"
              onClick={() => {
                setShowUploadModal(false);
                resetUpload();
              }}
              disabled={uploadingImage}
            >
              Cancel
            </button>

            <button
              className="modal-submit"
              onClick={uploadImage}
              disabled={
                uploadingImage ||
                !selectedFile
              }
            >
              {uploadingImage ? (
                <>
                  <span className="button-spinner"></span>
                  Uploading...
                </>
              ) : (
                <>
                  Upload Image
                  <span>↑</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    );
  };

  /* =========================
     CREATE CASE MODAL
  ========================= */

  const renderCreateCaseModal = () => {
    if (!showCreateCase) {
      return null;
    }

    return (
      <div
        className="modal-overlay"
        onClick={() => {
          if (!creatingCase) {
            setShowCreateCase(false);
          }
        }}
      >
        <div
          className="create-case-modal"
          onClick={(event) =>
            event.stopPropagation()
          }
        >
          <div className="modal-header">
            <div>
              <p className="eyebrow">
                NEW CASE
              </p>

              <h2>
                Create a Skin Assessment
                Case
              </h2>

              <p>
                Create a case before
                uploading your skin image.
              </p>
            </div>

            <button
              className="modal-close"
              onClick={() =>
                setShowCreateCase(false)
              }
              disabled={creatingCase}
            >
              ×
            </button>
          </div>

          <form
            onSubmit={
              handleCreateCase
            }
          >
            <div className="modal-form-group">
              <label htmlFor="case-title">
                Case Title
              </label>

              <input
                id="case-title"
                name="title"
                type="text"
                placeholder="e.g. Skin lesion assessment"
                value={
                  caseForm.title
                }
                onChange={
                  handleCreateCaseChange
                }
                maxLength={150}
                required
              />
            </div>

            <div className="modal-form-group">
              <label htmlFor="case-description">
                Description
              </label>

              <textarea
                id="case-description"
                name="description"
                placeholder="Briefly describe the reason for this assessment..."
                value={
                  caseForm.description
                }
                onChange={
                  handleCreateCaseChange
                }
                rows={4}
                maxLength={1000}
              />
            </div>

            {createCaseError && (
              <div className="modal-error">
                {createCaseError}
              </div>
            )}

            <div className="modal-actions">
              <button
                type="button"
                className="modal-cancel"
                onClick={() =>
                  setShowCreateCase(
                    false
                  )
                }
                disabled={creatingCase}
              >
                Cancel
              </button>

              <button
                type="submit"
                className="modal-submit"
                disabled={creatingCase}
              >
                {creatingCase ? (
                  <>
                    <span className="button-spinner"></span>
                    Creating...
                  </>
                ) : (
                  <>
                    Create Case
                    <span>→</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  };

  /* =========================
     REPORT MODAL
  ========================= */

  const renderReportModal = () => {
    if (
      !showReportModal ||
      !selectedAnalysis
    ) {
      return null;
    }

    const analysis =
      selectedAnalysis;

    const relatedImage =
      getRelatedImage(analysis);

    const topPredictions =
      getTopPredictions(analysis);

    const explainabilityUrl =
      getExplainabilityUrl(
        analysis
      );

    return (
      <div
        className="modal-overlay report-overlay"
        onClick={closeReport}
      >
        <div
          className="report-modal"
          onClick={(event) =>
            event.stopPropagation()
          }
        >
          <div className="report-modal-header">
            <div>
              <p className="eyebrow">
                DERM AI REPORT
              </p>

              <h2>
                Analysis Report
              </h2>

              <div className="report-completed-status">
                <span>✓</span>
                Analysis Completed
              </div>
            </div>

            <button
              className="modal-close"
              onClick={closeReport}
            >
              ×
            </button>
          </div>

          <div className="report-modal-body">

            {/* CASE INFORMATION */}

            <section className="report-content-section">
              <div className="report-section-heading">
                <span className="report-number">
                  01
                </span>

                <div>
                  <h3>
                    Case Information
                  </h3>

                  <p>
                    Case and image details
                  </p>
                </div>
              </div>

              <div className="report-info-grid">
                <div>
                  <span>Case ID</span>

                  <strong>
                    {analysis.caseId ||
                      selectedCaseId ||
                      "Not available"}
                  </strong>
                </div>

                <div>
                  <span>Case</span>

                  <strong>
                    {analysis.caseTitle ||
                      selectedCase?.title ||
                      "Not available"}
                  </strong>
                </div>

                <div>
                  <span>Image</span>

                  <strong>
                    {relatedImage?.original_filename ||
                      "Not available"}
                  </strong>
                </div>

                <div>
                  <span>Image Size</span>

                  <strong>
                    {relatedImage?.width &&
                    relatedImage?.height
                      ? `${relatedImage.width} × ${relatedImage.height}`
                      : "Not available"}
                  </strong>
                </div>

                <div>
                  <span>Analysis Date</span>

                  <strong>
                    {formatDateTime(
                      getAnalysisDate(
                        analysis
                      )
                    )}
                  </strong>
                </div>
              </div>
            </section>

            {/* QUALITY AND DETECTION */}

            <section className="report-content-section">
              <div className="report-section-heading">
                <span className="report-number">
                  02
                </span>

                <div>
                  <h3>
                    Quality & Detection
                  </h3>

                  <p>
                    Image quality and lesion
                    detection information
                  </p>
                </div>
              </div>

              <div className="report-info-grid">
                <div>
                  <span>
                    Image Quality
                  </span>

                  <strong>
                    {getQualityResult(
                      analysis,
                      relatedImage
                    )}
                  </strong>
                </div>

                <div>
                  <span>
                    Lesion Details
                  </span>

                  <strong>
                    {String(
                      getLesionDetails(
                        analysis
                      )
                    )}
                  </strong>
                </div>
              </div>
            </section>

            {/* PREDICTION RESULTS */}

            <section className="report-content-section">
              <div className="report-section-heading">
                <span className="report-number">
                  03
                </span>

                <div>
                  <h3>
                    Prediction Results
                  </h3>

                  <p>
                    AI prediction information
                    where supported
                  </p>
                </div>
              </div>

              <div className="prediction-result-card">
                <div>
                  <span>
                    Predicted Condition
                  </span>

                  <strong>
                    {getPrediction(
                      analysis
                    )}
                  </strong>
                </div>

                <div>
                  <span>
                    Confidence Score
                  </span>

                  <strong>
                    {getConfidence(
                      analysis
                    )}
                  </strong>
                </div>

                <div>
                  <span>
                    Risk Category
                  </span>

                  <strong>
                    {getRiskCategory(
                      analysis
                    )
                    }
                  </strong>
                </div>

                <div>
                  <span>Review Priority</span>
                  <strong>{getReviewPriority(analysis)}</strong>
                </div>
              </div>

              <div className="report-risk-guidance" style={{ marginTop: 16, padding: 14, borderRadius: 10, background: "#f8fafc", border: "1px solid #e2e8f0" }}>
                <p style={{ margin: "0 0 8px" }}>{getRiskMessage(analysis)}</p>
                <small style={{ color: "#64748b", lineHeight: 1.5 }}>{getRiskDisclaimer(analysis)}</small>
              </div>

              {topPredictions.length >
                0 && (
                <div className="top-predictions">
                  <h4>
                    Top-N Predictions
                  </h4>

                  {topPredictions.map(
                    (prediction, index) => {
                      const label =
                        typeof prediction ===
                        "string"
                          ? prediction
                          : prediction.label ||
                            prediction.condition ||
                            prediction.class ||
                            "Prediction";

                      const score =
                        typeof prediction ===
                        "object"
                          ? prediction.confidence ??
                            prediction.score ??
                            prediction.probability
                          : null;

                      return (
                        <div
                          className="prediction-row"
                          key={`${label}-${index}`}
                        >
                          <span>
                            {index + 1}.{" "}
                            {label}
                          </span>

                          <strong>
                            {score !==
                            null
                              ? `${
                                  Number(
                                    score
                                  ) <=
                                  1
                                    ? (
                                        Number(
                                          score
                                        ) *
                                        100
                                      ).toFixed(
                                        2
                                      )
                                    : Number(
                                        score
                                      ).toFixed(
                                        2
                                      )
                                }%`
                              : "N/A"}
                          </strong>
                        </div>
                      );
                    }
                  )}
                </div>
              )}
            </section>

            {/* VISUAL EXPLANATIONS */}

            <section className="report-content-section">
              <div className="report-section-heading">
                <span className="report-number">
                  04
                </span>

                <div>
                  <h3>
                    Visual Explanations
                  </h3>

                  <p>
                    Explainability visualization
                    where available
                  </p>
                </div>
              </div>

              {explainabilityUrl ? (

  <div className="explanation-preview">

    {explainabilityLoading ? (

      <p>
        Loading Grad-CAM visualization...
      </p>

    ) : explainabilityImageUrl ? (

      <img
        src={explainabilityImageUrl}
        alt="DermAI Grad-CAM Visualization"
        style={{
          width: "100%",
          height: "auto",
          display: "block",
          objectFit: "contain",
          borderRadius: "10px"
        }}
      />

    ) : (

      <div className="report-not-available">

        <span>◉</span>

        <p>
          {explainabilityError ||
            "Grad-CAM visualization is unavailable."}
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="report-not-available">
                <span>◉</span>
                <p>
                  No Grad-CAM visualization is available
                  for this analysis.
                </p>
              </div>
            )}
            </section>

            {/* METADATA AND SUMMARY */}

            <section className="report-content-section">
              <div className="report-section-heading">
                <span className="report-number">
                  05
                </span>

                <div>
                  <h3>
                    Metadata & Summary
                  </h3>

                  <p>
                    Model and analysis information
                  </p>
                </div>
              </div>

              <div className="report-info-grid">
                <div>
                  <span>
                    Model Version
                  </span>

                  <strong>
                    {getModelVersion(
                      analysis
                    )}
                  </strong>
                </div>

                <div>
                  <span>
                    Preprocessing
                  </span>

                  <strong>
                    {getStatusLabel(
                      analysis.preprocessing_status
                    )}
                  </strong>
                </div>

                <div>
                  <span>
                    Analysis Status
                  </span>

                  <strong className="report-complete-text">
                    {isAnalysisCompleted(
                      analysis
                    )
                      ? "Completed"
                      : getStatusLabel(
                          analysis.analysis_status
                        )}
                  </strong>
                </div>
              </div>

              <div className="report-summary">
                <span>
                  Analysis Summary
                </span>

                <p>
                  {getSummary(
                    analysis
                  )}
                </p>
              </div>
            </section>

            {/* DISCLAIMER */}

            <section className="report-disclaimer-box">
              <div className="report-disclaimer-icon">
                !
              </div>

              <div>
                <h3>
                  Important Disclaimer
                </h3>

                <p>
                  This output is an
                  <strong>
                    {" "}
                    AI-assisted analysis
                  </strong>{" "}
                  and is not a standalone
                  medical diagnosis. AI
                  predictions may be uncertain
                  and should not be used as a
                  substitute for professional
                  medical assessment.
                </p>
              </div>
            </section>
          </div>

          <div className="report-modal-footer">
            <div>
              <span>
                Available Export Formats
              </span>

              <strong>
                PDF · JSON · CSV
              </strong>
            </div>

            <div className="report-export-buttons">
              <button
                className="report-export-button"
                onClick={
                  exportPdfReport
                }
              >
                PDF
              </button>

              <button
                className="report-export-button"
                onClick={() =>
                  downloadJsonReport(
                    analysis
                  )
                }
              >
                JSON
              </button>

              <button
                className="report-export-button"
                onClick={() =>
                  downloadCsvReport(
                    analysis
                  )
                }
              >
                CSV
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  /* =========================
     MAIN
  ========================= */

  return (
    <div className="patient-dashboard">
      <aside className="dashboard-sidebar">
        <div className="brand">
          <div className="brand-icon">
            D
          </div>

          <div>
            <h2>DermAI</h2>

            <span>
              Skin Health Platform
            </span>
          </div>
        </div>

        <nav className="sidebar-nav">
          <button
            className={`nav-item ${
              currentSection ===
              "dashboard"
                ? "active"
                : ""
            }`}
            onClick={() =>
              setSection("dashboard")
            }
          >
            <span className="nav-icon">
              ⌂
            </span>

            Dashboard
          </button>

          <button
            className={`nav-item ${
              currentSection ===
                "cases" ||
              currentSection === "case"
                ? "active"
                : ""
            }`}
            onClick={() =>
              setSection("cases")
            }
          >
            <span className="nav-icon">
              ▣
            </span>

            My Cases
          </button>

          <button
            className={`nav-item ${
              currentSection ===
              "assessments"
                ? "active"
                : ""
            }`}
            onClick={() =>
              setSection("assessments")
            }
          >
            <span className="nav-icon">
              ✦
            </span>

            Assessments
          </button>

          <button
            className={`nav-item ${
              currentSection ===
              "profile"
                ? "active"
                : ""
            }`}
            onClick={() =>
              setSection("profile")
            }
          >
            <span className="nav-icon">
              ◌
            </span>

            Profile
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="research-note">
            <div className="research-icon">
              i
            </div>

            <div>
              <strong>
                Research Tool
              </strong>

              <p>
                DermAI provides AI-assisted
                analysis for research and
                educational purposes.
              </p>
            </div>
          </div>

          <button
            className="logout-button"
            onClick={logout}
          >
            <span>↪</span>
            Sign Out
          </button>
        </div>
      </aside>

      <main className="dashboard-main">
        <header className="dashboard-header">
          <div className="mobile-brand">
            <div className="brand-icon">
              D
            </div>

            <strong>DermAI</strong>
          </div>

          <div className="header-actions">
            <button
              className="refresh-button"
              onClick={
                refreshDashboard
              }
              disabled={
                dashboardLoading
              }
              title="Refresh"
            >
              ↻
            </button>

            <NotificationBell variant="patient" />

            <button
              className="profile-menu"
              onClick={() =>
                setSection("profile")
              }
            >
              <div className="profile-avatar">
                {displayName
                  .charAt(0)
                  .toUpperCase()}
              </div>

              <div className="profile-info">
                <strong>
                  {displayName}
                </strong>

                <span>
                  Patient
                </span>
              </div>
            </button>
          </div>
        </header>

        {currentSection ===
          "dashboard" &&
          renderDashboard()}

        {currentSection === "cases" &&
          renderCases()}

        {currentSection === "case" &&
          renderCaseDetail()}

        {currentSection ===
          "assessments" &&
          renderAssessments()}

        {currentSection ===
          "profile" &&
          renderProfile()}

        <footer className="dashboard-footer">
          <span>
            DermAI © 2026
          </span>

          <span>
            AI-assisted skin image
            analysis · Research use only
          </span>
        </footer>
      </main>

      {renderCreateCaseModal()}
      {renderUploadModal()}
      {renderReportModal()}
    </div>
  );
}

export default PatientDashboard;