import { useEffect, useMemo, useState } from "react";
import { useAuth } from "../context/AuthContext";
import api from "../services/api";
import NotificationBell from "../components/NotificationBell";
import "./DoctorDashboard.css";

function DoctorDashboard() {
  const { user, logout } = useAuth();

  const [cases, setCases] = useState([]);
  const [loadingCases, setLoadingCases] = useState(true);
  const [caseError, setCaseError] = useState("");

  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");

  const [selectedCase, setSelectedCase] = useState(null);
  const [showCaseReview, setShowCaseReview] = useState(false);

  // ADDED
  const [showProfile, setShowProfile] = useState(false);

  const [loadingCaseReview, setLoadingCaseReview] = useState(false);
  const [savingClinicalNote, setSavingClinicalNote] = useState(false);
  const [clinicalNote, setClinicalNote] = useState("");
  const [reviewError, setReviewError] = useState("");
  const [reviewSuccess, setReviewSuccess] = useState("");

  const displayName =
    user?.full_name ||
    user?.name ||
    "Doctor";

  const firstName = displayName.split(" ")[0];

  /*
   * ============================================================
   * LOAD DOCTOR CASES
   * ============================================================
   */

  useEffect(() => {
    let mounted = true;

    const loadCases = async () => {
      try {
        setLoadingCases(true);
        setCaseError("");

        let response;

        try {
          response = await api.get("/doctor/cases");
        } catch (doctorEndpointError) {
          response = await api.get("/cases");
        }

        if (mounted) {
          const receivedCases = Array.isArray(response.data)
            ? response.data
            : response.data?.cases || [];

          setCases(receivedCases);
        }
      } catch (error) {
        console.error("Failed to load doctor cases:", error);

        if (mounted) {
          if (error.response?.data?.detail) {
            setCaseError(error.response.data.detail);
          } else {
            setCaseError(
              "Unable to load patient cases. Please try again."
            );
          }
        }
      } finally {
        if (mounted) {
          setLoadingCases(false);
        }
      }
    };

    loadCases();

    return () => {
      mounted = false;
    };
  }, []);

  /*
   * ============================================================
   * FORMAT DATE
   * ============================================================
   */

  const formatDate = (dateValue) => {
    if (!dateValue) {
      return "Date unavailable";
    }

    const date = new Date(dateValue);

    if (Number.isNaN(date.getTime())) {
      return "Date unavailable";
    }

    return date.toLocaleDateString("en-IN", {
      day: "2-digit",
      month: "short",
      year: "numeric",
    });
  };

  /*
   * ============================================================
   * STATUS HELPERS
   * ============================================================
   */

  const normalizeStatus = (status) => {
    return String(status || "")
      .toLowerCase()
      .replace(/\s+/g, "-");
  };

  const getStatusClass = (status) => {
    const normalizedStatus = normalizeStatus(status);

    if (
      normalizedStatus === "completed" ||
      normalizedStatus === "complete" ||
      normalizedStatus === "reviewed"
    ) {
      return "doctor-status-completed";
    }

    if (
      normalizedStatus === "active" ||
      normalizedStatus === "in-progress" ||
      normalizedStatus === "processing"
    ) {
      return "doctor-status-active";
    }

    if (
      normalizedStatus === "failed" ||
      normalizedStatus === "rejected"
    ) {
      return "doctor-status-danger";
    }

    return "doctor-status-pending";
  };

  const getStatusLabel = (status) => {
    if (!status) {
      return "Pending";
    }

    return String(status)
      .replace(/-/g, " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  };

  /*
   * ============================================================
   * GET PATIENT NAME
   * ============================================================
   */

  const getPatientName = (caseItem) => {
    return (
      caseItem.patient_name ||
      caseItem.patient?.full_name ||
      caseItem.patient?.name ||
      caseItem.user_name ||
      caseItem.user?.full_name ||
      caseItem.user?.name ||
      "Unknown Patient"
    );
  };

  /*
   * ============================================================
   * GET CASE TITLE
   * ============================================================
   */

  const getCaseTitle = (caseItem) => {
    return caseItem.title || "Untitled Assessment";
  };

  /*
   * ============================================================
   * GET CASE ID
   * ============================================================
   */

  const getCaseId = (caseItem) => {
    return caseItem.id || caseItem._id;
  };

  const getAiStatus = (caseItem) => {
    const rawStatus =
      caseItem?.analysis_status ||
      caseItem?.analysis?.analysis_status ||
      caseItem?.analysis?.status ||
      caseItem?.ai_status ||
      caseItem?.latest_analysis?.analysis_status ||
      caseItem?.latest_analysis?.status ||
      caseItem?.status;

    const status = normalizeStatus(rawStatus);

    return ["completed", "complete", "success", "successful"].includes(status)
      ? "completed"
      : "pending";
  };

  /*
   * ============================================================
   * FILTER CASES
   * ============================================================
   */

  const filteredCases = useMemo(() => {
    const search = searchTerm.trim().toLowerCase();

    return cases.filter((caseItem) => {
      const patientName = getPatientName(caseItem).toLowerCase();
      const title = getCaseTitle(caseItem).toLowerCase();
      const status = getAiStatus(caseItem);

      const matchesSearch =
        !search ||
        patientName.includes(search) ||
        title.includes(search);

      const matchesStatus =
        statusFilter === "all" ||
        status === statusFilter;

      return matchesSearch && matchesStatus;
    });
  }, [cases, searchTerm, statusFilter]);

  /*
   * ============================================================
   * STATISTICS
   * ============================================================
   */

  const statistics = useMemo(() => {
    let pending = 0;
    let completed = 0;

    cases.forEach((caseItem) => {
      if (getAiStatus(caseItem) === "completed") completed += 1;
      else pending += 1;
    });

    return {
      total: cases.length,
      pending,
      processing: 0,
      completed,
    };
  }, [cases]);

  /*
   * ============================================================
   * VIEW CASE
   * ============================================================
   */

  const handleViewCase = async (caseItem) => {
    const caseId = getCaseId(caseItem);
    if (!caseId) return;

    try {
      setLoadingCaseReview(true);
      setReviewError("");
      setReviewSuccess("");
      setShowCaseReview(true);

      const response = await api.get(`/doctor/cases/${caseId}`);
      const detail = response.data || caseItem;

      setSelectedCase(detail);

      setClinicalNote(
        detail.clinical_note ||
          detail.doctor_note ||
          detail.clinical_notes ||
          ""
      );
    } catch (error) {
      console.error("Failed to load case details:", error);

      setReviewError(
        error.response?.data?.detail ||
          "Unable to load case details."
      );
    } finally {
      setLoadingCaseReview(false);
    }
  };

  const handleSaveClinicalNote = async () => {
    const caseId = getCaseId(selectedCase);
    if (!caseId) return;

    try {
      setSavingClinicalNote(true);
      setReviewError("");
      setReviewSuccess("");

      const response = await api.put(
        `/doctor/cases/${caseId}/notes`,
        {
          clinical_note: clinicalNote.trim(),
        }
      );

      const updatedCase = response.data || {
        ...selectedCase,
        clinical_note: clinicalNote.trim(),
      };

      setSelectedCase(updatedCase);

      setCases((previous) =>
        previous.map((item) =>
          String(getCaseId(item)) === String(caseId)
            ? {
                ...item,
                clinical_note: clinicalNote.trim(),
              }
            : item
        )
      );

      setReviewSuccess(
        "Clinical note saved successfully."
      );
    } catch (error) {
      console.error(
        "Failed to save clinical note:",
        error
      );

      setReviewError(
        error.response?.data?.detail ||
          "Unable to save the clinical note."
      );
    } finally {
      setSavingClinicalNote(false);
    }
  };

  const closeCaseReview = () => {
    if (savingClinicalNote) return;

    setShowCaseReview(false);
    setSelectedCase(null);
    setClinicalNote("");
    setReviewError("");
    setReviewSuccess("");
  };

  /*
   * ============================================================
   * REFRESH CASES
   * ============================================================
   */

  const handleRefresh = async () => {
    try {
      setLoadingCases(true);
      setCaseError("");

      let response;

      try {
        response = await api.get("/doctor/cases");
      } catch (doctorEndpointError) {
        response = await api.get("/cases");
      }

      const receivedCases = Array.isArray(response.data)
        ? response.data
        : response.data?.cases || [];

      setCases(receivedCases);
    } catch (error) {
      console.error(
        "Failed to refresh doctor cases:",
        error
      );

      if (error.response?.data?.detail) {
        setCaseError(error.response.data.detail);
      } else {
        setCaseError("Unable to refresh cases.");
      }
    } finally {
      setLoadingCases(false);
    }
  };

  return (
    <div className="doctor-dashboard">

      {/* ======================================================
          SIDEBAR
      ====================================================== */}

      <aside className="doctor-sidebar">

        <div className="doctor-brand">
          <div className="doctor-brand-icon">
            D
          </div>

          <div>
            <h2>DermAI</h2>
            <span>Clinical Workspace</span>
          </div>
        </div>

        <nav className="doctor-sidebar-nav">

          <button className="doctor-nav-item active">
            <span className="doctor-nav-icon">⌂</span>
            Dashboard
          </button>

          <button
            className="doctor-nav-item"
            onClick={() => {
              setStatusFilter("all");

              window.scrollTo({
                top: 0,
                behavior: "smooth",
              });
            }}
          >
            <span className="doctor-nav-icon">▣</span>
            Patient Cases
          </button>

          <button
            className="doctor-nav-item"
            onClick={() =>
              setStatusFilter("pending")
            }
          >
            <span className="doctor-nav-icon">◷</span>
            Pending Review

            {statistics.pending > 0 && (
              <span className="doctor-nav-badge">
                {statistics.pending}
              </span>
            )}
          </button>

          <button
            className="doctor-nav-item"
            onClick={() =>
              setStatusFilter("completed")
            }
          >
            <span className="doctor-nav-icon">✓</span>
            Completed
          </button>

          {/* FIXED PROFILE BUTTON */}
          <button
            className="doctor-nav-item"
            onClick={() => setShowProfile(true)}
          >
            <span className="doctor-nav-icon">◌</span>
            Profile
          </button>

        </nav>

        <div className="doctor-sidebar-bottom">

          <div className="doctor-research-note">

            <div className="doctor-research-icon">
              i
            </div>

            <div>
              <strong>
                Clinical Support Tool
              </strong>

              <p>
                DermAI provides AI-assisted image
                analysis for research and educational
                support.
              </p>
            </div>

          </div>

          <button
            className="doctor-logout-button"
            onClick={logout}
          >
            <span>↪</span>
            Sign Out
          </button>

        </div>

      </aside>

      {/* ======================================================
          MAIN
      ====================================================== */}

      <main className="doctor-dashboard-main">

        {/* HEADER */}

        <header className="doctor-header">

          <div className="doctor-mobile-brand">
            <div className="doctor-brand-icon">
              D
            </div>

            <strong>DermAI</strong>
          </div>

          <div className="doctor-header-actions">

            <NotificationBell variant="doctor" />

            <div className="doctor-profile">

              <div className="doctor-profile-avatar">
                {displayName.charAt(0).toUpperCase()}
              </div>

              <div className="doctor-profile-info">
                <strong>{displayName}</strong>
                <span>Doctor</span>
              </div>

            </div>

          </div>

        </header>

        {/* WELCOME */}

        <section className="doctor-welcome">

          <div>

            <p className="doctor-eyebrow">
              DOCTOR DASHBOARD
            </p>

            <h1>
              Welcome back, Dr. {firstName}
              <span className="doctor-wave">👋</span>
            </h1>

            <p className="doctor-welcome-text">
              Review patient cases, monitor AI-assisted
              assessments, and manage dermatological
              analysis activity.
            </p>

          </div>

          <button
            className="doctor-refresh-button"
            onClick={handleRefresh}
            disabled={loadingCases}
          >
            <span>↻</span>
            {loadingCases
              ? "Refreshing..."
              : "Refresh"}
          </button>

        </section>

        {/* STATISTICS */}

        <section className="doctor-stats-grid">

          <div className="doctor-stat-card">

            <div className="doctor-stat-icon doctor-cases-icon">
              ▣
            </div>

            <div className="doctor-stat-content">

              <span className="doctor-stat-label">
                PATIENT CASES
              </span>

              <strong>
                {loadingCases
                  ? "..."
                  : statistics.total}
              </strong>

              <p>
                Total assigned cases
              </p>

            </div>

          </div>

          <div className="doctor-stat-card">

            <div className="doctor-stat-icon doctor-pending-icon">
              ◷
            </div>

            <div className="doctor-stat-content">

              <span className="doctor-stat-label">
                PENDING REVIEW
              </span>

              <strong>
                {loadingCases
                  ? "..."
                  : statistics.pending}
              </strong>

              <p>
                Cases awaiting review
              </p>

            </div>

          </div>

          <div className="doctor-stat-card">

            <div className="doctor-stat-icon doctor-processing-icon">
              ✦
            </div>

            <div className="doctor-stat-content">

              <span className="doctor-stat-label">
                PROCESSING
              </span>

              <strong>
                {loadingCases
                  ? "..."
                  : statistics.processing}
              </strong>

              <p>
                Active AI assessments
              </p>

            </div>

          </div>

          <div className="doctor-stat-card">

            <div className="doctor-stat-icon doctor-completed-icon">
              ✓
            </div>

            <div className="doctor-stat-content">

              <span className="doctor-stat-label">
                COMPLETED
              </span>

              <strong>
                {loadingCases
                  ? "..."
                  : statistics.completed}
              </strong>

              <p>
                Completed assessments
              </p>

            </div>

          </div>

        </section>

        {/* MAIN CASE SECTION */}

        <section className="doctor-dashboard-card doctor-cases-card">

          <div className="doctor-card-header">

            <div>
              <h2>Patient Cases</h2>

              <p>
                Review and manage patient skin assessment
                cases.
              </p>
            </div>

            <div className="doctor-card-header-count">
              {filteredCases.length} cases
            </div>

          </div>

          {/* SEARCH + FILTER */}

          <div className="doctor-case-controls">

            <div className="doctor-search-wrapper">

              <span className="doctor-search-icon">
                ⌕
              </span>

              <input
                type="text"
                placeholder="Search patient or case..."
                value={searchTerm}
                onChange={(event) =>
                  setSearchTerm(event.target.value)
                }
                className="doctor-search-input"
              />

              {searchTerm && (
                <button
                  className="doctor-clear-search"
                  onClick={() =>
                    setSearchTerm("")
                  }
                  aria-label="Clear search"
                >
                  ×
                </button>
              )}

            </div>

            <select
              className="doctor-status-filter"
              value={statusFilter}
              onChange={(event) =>
                setStatusFilter(event.target.value)
              }
            >
              <option value="all">
                All Status
              </option>

              <option value="pending">
                Pending
              </option>

              <option value="completed">
                Completed
              </option>
            </select>

          </div>

          {/* ERROR */}

          {caseError && (
            <div className="doctor-error-state">

              <div className="doctor-error-icon">
                !
              </div>

              <div>
                <strong>
                  Unable to load patient cases
                </strong>

                <p>{caseError}</p>
              </div>

              <button
                className="doctor-secondary-button"
                onClick={handleRefresh}
              >
                Try Again
              </button>

            </div>
          )}

          {/* LOADING */}

          {loadingCases && !caseError && (
            <div className="doctor-loading-state">

              <div className="doctor-loading-spinner"></div>

              <p>
                Loading patient cases...
              </p>

            </div>
          )}

          {/* EMPTY */}

          {!loadingCases &&
            !caseError &&
            filteredCases.length === 0 && (

              <div className="doctor-empty-state">

                <div className="doctor-empty-icon">
                  ▣
                </div>

                <h3>
                  {cases.length === 0
                    ? "No patient cases yet"
                    : "No matching cases"}
                </h3>

                <p>
                  {cases.length === 0
                    ? "Patient assessment cases will appear here when they are assigned to you."
                    : "Try changing your search term or status filter."}
                </p>

                {searchTerm && (
                  <button
                    className="doctor-secondary-button"
                    onClick={() => {
                      setSearchTerm("");
                      setStatusFilter("all");
                    }}
                  >
                    Clear Filters
                  </button>
                )}

              </div>
            )}

          {/* TABLE */}

          {!loadingCases &&
            !caseError &&
            filteredCases.length > 0 && (

              <div className="doctor-table-wrapper">

                <table className="doctor-cases-table">

                  <thead>
                    <tr>

                      <th>Patient</th>

                      <th>Case</th>

                      <th>Created</th>

                      <th>Status</th>

                      <th>AI Analysis</th>

                      <th></th>

                    </tr>
                  </thead>

                  <tbody>

                    {filteredCases.map((caseItem) => {

                      const caseId =
                        getCaseId(caseItem);

                      const patientName =
                        getPatientName(caseItem);

                      const status =
                        getAiStatus(caseItem);

                      const analysisStatus =
                        getAiStatus(caseItem);

                      return (
                        <tr
                          key={
                            caseId ||
                            `${patientName}-${caseItem.created_at}`
                          }
                        >

                          <td>

                            <div className="doctor-patient-cell">

                              <div className="doctor-patient-avatar">
                                {patientName
                                  .charAt(0)
                                  .toUpperCase()}
                              </div>

                              <div>
                                <strong>
                                  {patientName}
                                </strong>

                                <span>
                                  Patient
                                </span>
                              </div>

                            </div>

                          </td>

                          <td>

                            <div className="doctor-case-cell">

                              <strong>
                                {getCaseTitle(caseItem)}
                              </strong>

                              <span>
                                {caseItem.description
                                  ? caseItem.description.slice(
                                      0,
                                      55
                                    ) +
                                    (caseItem.description
                                      .length > 55
                                      ? "..."
                                      : "")
                                  : "Skin assessment case"}
                              </span>

                            </div>

                          </td>

                          <td>

                            <span className="doctor-date">
                              {formatDate(
                                caseItem.created_at
                              )}
                            </span>

                          </td>

                          <td>

                            <span
                              className={`doctor-case-status ${getStatusClass(
                                status
                              )}`}
                            >

                              <span className="doctor-status-dot"></span>

                              {getStatusLabel(status)}

                            </span>

                          </td>

                          <td>

                            <span className="doctor-analysis-status">
                              {getStatusLabel(
                                analysisStatus
                              )}
                            </span>

                          </td>

                          <td>

                            <button
                              className="doctor-view-button"
                              onClick={() =>
                                handleViewCase(caseItem)
                              }
                              disabled={!caseId}
                            >
                              Review →
                            </button>

                          </td>

                        </tr>
                      );
                    })}

                  </tbody>

                </table>

              </div>
            )}

        </section>

        {/* QUICK ACTIONS */}

        <section className="doctor-dashboard-grid">

          <div className="doctor-dashboard-card doctor-quick-card">

            <div className="doctor-card-header">

              <div>

                <h2>Quick Actions</h2>

                <p>
                  Quickly access your most-used
                  clinical workflows.
                </p>

              </div>

            </div>

            <div className="doctor-quick-actions">

              <button
                className="doctor-quick-action"
                onClick={() =>
                  setStatusFilter("pending")
                }
              >

                <div className="doctor-quick-icon pending">
                  ◷
                </div>

                <div>
                  <strong>
                    Pending Reviews
                  </strong>

                  <span>
                    Review cases awaiting attention
                  </span>
                </div>

                <span className="doctor-quick-arrow">
                  →
                </span>

              </button>

              <button
                className="doctor-quick-action"
                onClick={() =>
                  setStatusFilter("pending")
                }
              >

                <div className="doctor-quick-icon processing">
                  ✦
                </div>

                <div>
                  <strong>
                    Pending AI Analysis
                  </strong>

                  <span>
                    Waiting for AI analysis to complete
                  </span>
                </div>

                <span className="doctor-quick-arrow">
                  →
                </span>

              </button>

              <button
                className="doctor-quick-action"
                onClick={() => {
                  setStatusFilter("all");
                  setSearchTerm("");
                }}
              >

                <div className="doctor-quick-icon patients">
                  ▣
                </div>

                <div>
                  <strong>
                    All Patient Cases
                  </strong>

                  <span>
                    View your complete case list
                  </span>
                </div>

                <span className="doctor-quick-arrow">
                  →
                </span>

              </button>

            </div>

          </div>

          {/* CLINICAL NOTICE */}

          <div className="doctor-dashboard-card doctor-notice-card">

            <div className="doctor-notice-icon">
              i
            </div>

            <div>

              <p className="doctor-eyebrow">
                IMPORTANT
              </p>

              <h2>
                AI-assisted analysis
              </h2>

              <p>
                DermAI results are intended to support
                clinical review and research. AI predictions
                should not be treated as a standalone
                medical diagnosis.
              </p>

              <div className="doctor-notice-badge">
                Research / Educational Use
              </div>

            </div>

          </div>

        </section>

        {/* FOOTER */}

        <footer className="doctor-dashboard-footer">

          <span>
            DermAI © 2026
          </span>

          <span>
            AI-assisted dermatological image analysis
            · Research use only
          </span>

        </footer>

      </main>

      {/* ======================================================
          PROFILE MODAL - ADDED
      ====================================================== */}

      {showProfile && (
        <div
          className="doctor-profile-overlay"
          onClick={() => setShowProfile(false)}
        >
          <div
            className="doctor-profile-modal"
            onClick={(event) =>
              event.stopPropagation()
            }
          >

            <div className="doctor-profile-header">

              <div>
                <p className="doctor-eyebrow">
                  ACCOUNT
                </p>

                <h2>Doctor Profile</h2>

                <p>
                  Your DermAI account information.
                </p>
              </div>

              <button
                className="doctor-profile-close"
                onClick={() =>
                  setShowProfile(false)
                }
              >
                ×
              </button>

            </div>

            <div className="doctor-profile-body">

              <div className="doctor-profile-large-avatar">
                {displayName
                  .charAt(0)
                  .toUpperCase()}
              </div>

              <div className="doctor-profile-details">

                <div className="doctor-profile-field">
                  <span>Full Name</span>

                  <strong>
                    {user?.full_name ||
                      user?.name ||
                      "Not available"}
                  </strong>
                </div>

                <div className="doctor-profile-field">
                  <span>Email</span>

                  <strong>
                    {user?.email ||
                      "Not available"}
                  </strong>
                </div>

                <div className="doctor-profile-field">
                  <span>Account Role</span>

                  <strong>
                    {user?.role || "doctor"}
                  </strong>
                </div>

                <div className="doctor-profile-field">
                  <span>Account Status</span>

                  <strong className="doctor-profile-active">
                    Active
                  </strong>
                </div>

              </div>

            </div>

            <div className="doctor-profile-footer">

              <button
                className="doctor-secondary-button"
                onClick={() =>
                  setShowProfile(false)
                }
              >
                Close
              </button>

            </div>

          </div>
        </div>
      )}

      {/* ======================================================
          CASE REVIEW MODAL
      ====================================================== */}

      {showCaseReview && (
        <div
          className="doctor-review-overlay"
          onClick={closeCaseReview}
        >
          <div
            className="doctor-review-modal"
            onClick={(event) =>
              event.stopPropagation()
            }
          >

            <div className="doctor-review-header">

              <div>
                <p className="doctor-eyebrow">
                  CASE REVIEW
                </p>

                <h2>
                  Patient Case Details
                </h2>

                <p>
                  Review the AI-assisted result and add
                  a clinical note.
                </p>
              </div>

              <button
                className="doctor-review-close"
                onClick={closeCaseReview}
              >
                ×
              </button>

            </div>

            {loadingCaseReview ? (
              <div className="doctor-review-loading">

                <div className="doctor-loading-spinner"></div>

                <p>
                  Loading case details...
                </p>

              </div>
            ) : (
              <div className="doctor-review-body">

                {reviewError && (
                  <div className="doctor-review-error">
                    {reviewError}
                  </div>
                )}

                {reviewSuccess && (
                  <div className="doctor-review-success">
                    {reviewSuccess}
                  </div>
                )}

                <div className="doctor-review-grid">

                  <div>
                    <span>Patient</span>
                    <strong>
                      {getPatientName(
                        selectedCase || {}
                      )}
                    </strong>
                  </div>

                  <div>
                    <span>Case</span>
                    <strong>
                      {getCaseTitle(
                        selectedCase || {}
                      )}
                    </strong>
                  </div>

                  <div>
                    <span>AI Status</span>

                    <strong
                      className={
                        getAiStatus(
                          selectedCase || {}
                        ) === "completed"
                          ? "doctor-review-completed"
                          : "doctor-review-pending"
                      }
                    >
                      {getAiStatus(
                        selectedCase || {}
                      ) === "completed"
                        ? "Completed"
                        : "Pending"}
                    </strong>
                  </div>

                  <div>
                    <span>Created</span>

                    <strong>
                      {formatDate(
                        selectedCase?.created_at
                      )}
                    </strong>
                  </div>

                </div>

                <div className="doctor-review-section">

                  <h3>
                    Case Description
                  </h3>

                  <p>
                    {selectedCase?.description ||
                      "No description provided."}
                  </p>

                </div>

                <div className="doctor-review-section">

                  <h3>
                    AI Prediction
                  </h3>

                  <p>
                    {selectedCase?.analysis
                      ?.predicted_condition ||
                      selectedCase?.analysis
                        ?.condition ||
                      selectedCase?.predicted_condition ||
                      "AI analysis pending"}
                  </p>

                </div>

                {/* AI-assisted review priority; not a clinical diagnosis. */}
                <div className="doctor-review-section">
                  <h3>AI Risk Assessment</h3>
                  {(() => {
                    const analysis = selectedCase?.analysis || selectedCase?.latest_analysis || {};
                    const risk = analysis.risk_assessment || analysis;
                    const confidence = analysis.confidence ?? analysis.probability;
                    const confidenceValue = confidence == null
                      ? null
                      : Number(confidence) <= 1
                        ? Number(confidence) * 100
                        : Number(confidence);

                    return (
                      <>
                        <p><strong>Risk category: </strong>{risk.risk_category || "Not available"}</p>
                        <p><strong>Review priority: </strong>{risk.review_priority || "Not available"}</p>
                        <p><strong>Model confidence: </strong>{confidenceValue == null || Number.isNaN(confidenceValue) ? "Not available" : `${confidenceValue.toFixed(2)}%`}</p>
                        <p>{risk.message || analysis.risk_message || "Risk assessment is not available for this record."}</p>
                        <small style={{ display: "block", color: "#64748b", lineHeight: 1.5 }}>
                          {risk.disclaimer || analysis.medical_disclaimer || "AI results are not a medical diagnosis. Consult a qualified dermatologist."}
                        </small>
                      </>
                    );
                  })()}
                </div>

                <div className="doctor-review-section">

                  <h3>
                    Clinical Note
                  </h3>

                  <textarea
                    value={clinicalNote}
                    onChange={(event) =>
                      setClinicalNote(
                        event.target.value
                      )
                    }
                    placeholder="Enter clinical observations, recommendations, or follow-up notes..."
                    rows={7}
                  />

                </div>

              </div>
            )}

            {!loadingCaseReview && (
              <div className="doctor-review-footer">

                <button
                  className="doctor-secondary-button"
                  onClick={closeCaseReview}
                >
                  Close
                </button>

                <button
                  className="doctor-primary-review-button"
                  onClick={handleSaveClinicalNote}
                  disabled={savingClinicalNote}
                >
                  {savingClinicalNote
                    ? "Saving..."
                    : "Submit Clinical Note"}
                </button>

              </div>
            )}

          </div>
        </div>
      )}

    </div>
  );
}

export default DoctorDashboard;