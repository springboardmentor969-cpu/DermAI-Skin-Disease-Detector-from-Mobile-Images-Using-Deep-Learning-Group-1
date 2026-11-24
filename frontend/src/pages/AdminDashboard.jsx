import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import api from "../services/api";
import NotificationBell from "../components/NotificationBell";
import "./AdminDashboard.css";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const DASHBOARD_ENDPOINT = `${API_BASE_URL}/api/admin/dashboard`;

const Icon = ({ name, size = 20 }) => {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: "1.8",
    strokeLinecap: "round",
    strokeLinejoin: "round",
  };

  const icons = {
    dashboard: (
      <>
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </>
    ),

    users: (
      <>
        <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
        <path d="M16 3.13a4 4 0 0 1 0 7.75" />
      </>
    ),

    doctor: (
      <>
        <circle cx="12" cy="7" r="4" />
        <path d="M5.5 21a6.5 6.5 0 0 1 13 0" />
        <path d="M18 8h3" />
        <path d="M19.5 6.5v3" />
      </>
    ),

    analysis: (
      <>
        <path d="M4 19V5" />
        <path d="M4 19h16" />
        <path d="M7 16v-5" />
        <path d="M11 16V7" />
        <path d="M15 16v-3" />
        <path d="M19 16V5" />
      </>
    ),

    activity: (
      <>
        <polyline points="3 12 7 12 10 5 14 19 17 12 21 12" />
      </>
    ),

    model: (
      <>
        <rect x="4" y="4" width="16" height="16" rx="2" />
        <rect x="8" y="8" width="8" height="8" rx="1" />
        <path d="M8 2v2" />
        <path d="M12 2v2" />
        <path d="M16 2v2" />
        <path d="M8 20v2" />
        <path d="M12 20v2" />
        <path d="M16 20v2" />
        <path d="M2 8h2" />
        <path d="M2 12h2" />
        <path d="M2 16h2" />
        <path d="M20 8h2" />
        <path d="M20 12h2" />
        <path d="M20 16h2" />
      </>
    ),

    settings: (
      <>
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.7 1.7 0 0 0 .34 1.88l.06.06-1.7 1.7-.06-.06a1.7 1.7 0 0 0-1.88-.34 1.7 1.7 0 0 0-1.03 1.56V20h-2.4v-.2a1.7 1.7 0 0 0-1.03-1.56 1.7 1.7 0 0 0-1.88.34l-.06.06-1.7-1.7.06-.06A1.7 1.7 0 0 0 8.4 15a1.7 1.7 0 0 0-1.56-1.03H6v-2.4h.2A1.7 1.7 0 0 0 7.76 10a1.7 1.7 0 0 0-.34-1.88l-.06-.06 1.7-1.7.06.06A1.7 1.7 0 0 0 11 6.4a1.7 1.7 0 0 0 1.03-1.56V4h2.4v.2a1.7 1.7 0 0 0 1.03 1.56 1.7 1.7 0 0 0 1.88-.34l.06-.06 1.7 1.7-.06.06A1.7 1.7 0 0 0 18.6 10a1.7 1.7 0 0 0 1.56 1.03h.2v2.4h-.2A1.7 1.7 0 0 0 19.4 15Z" />
      </>
    ),

    bell: (
      <>
        <path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9" />
        <path d="M10 21h4" />
      </>
    ),

    search: (
      <>
        <circle cx="11" cy="11" r="7" />
        <path d="m20 20-4-4" />
      </>
    ),

    refresh: (
      <>
        <path d="M20 11a8.1 8.1 0 0 0-14.9-4L3 10" />
        <path d="M3 4v6h6" />
        <path d="M4 13a8.1 8.1 0 0 0 14.9 4L21 14" />
        <path d="M21 20v-6h-6" />
      </>
    ),

    logout: (
      <>
        <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
        <polyline points="16 17 21 12 16 7" />
        <line x1="21" y1="12" x2="9" y2="12" />
      </>
    ),

    check: (
      <>
        <polyline points="20 6 9 17 4 12" />
      </>
    ),

    warning: (
      <>
        <path d="M10.3 3.4 2.2 18a2 2 0 0 0 1.7 3h16.2a2 2 0 0 0 1.7-3L13.7 3.4a2 2 0 0 0-3.4 0Z" />
        <line x1="12" y1="9" x2="12" y2="13" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
      </>
    ),

    eye: (
      <>
        <path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12Z" />
        <circle cx="12" cy="12" r="2.5" />
      </>
    ),

    arrow: (
      <>
        <line x1="5" y1="12" x2="19" y2="12" />
        <polyline points="12 5 19 12 12 19" />
      </>
    ),

    shield: (
      <>
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" />
        <polyline points="9 12 11 14 15 10" />
      </>
    ),

    menu: (
      <>
        <line x1="4" y1="6" x2="20" y2="6" />
        <line x1="4" y1="12" x2="20" y2="12" />
        <line x1="4" y1="18" x2="20" y2="18" />
      </>
    ),
  };

  return <svg {...common}>{icons[name]}</svg>;
};

const DEFAULT_DATA = {
  admin: {
    full_name: "Administrator",
    email: "",
  },

  stats: {
    total_users: 0,
    total_doctors: 0,
    total_patients: 0,
    total_analyses: 0,
    pending_reviews: 0,
    high_risk_cases: 0,
    active_users: 0,
    system_uptime: "—",
    resnet50_usage: 0,
    densenet121_usage: 0,
  },

  recent_activity: [],
};

function getToken() {
  return (
    localStorage.getItem("access_token") ||
    localStorage.getItem("token") ||
    localStorage.getItem("authToken") ||
    ""
  );
}

function getInitials(name = "Admin") {
  return name
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0].toUpperCase())
    .join("");
}

function formatNumber(value) {
  if (value === null || value === undefined || value === "") return "0";
  return Number(value).toLocaleString();
}

function formatDate(value) {
  if (!value) return "—";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleDateString("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

function normaliseData(payload) {
  const source = payload?.data || payload || {};

  return {
    admin: {
      full_name:
        source.admin?.full_name ||
        source.user?.full_name ||
        source.full_name ||
        "Administrator",

      email:
        source.admin?.email ||
        source.user?.email ||
        source.email ||
        "",
    },

    stats: {
      total_users:
        source.stats?.total_users ??
        source.total_users ??
        source.users_count ??
        0,

      total_doctors:
        source.stats?.total_doctors ??
        source.total_doctors ??
        source.doctors_count ??
        0,

      total_patients:
        source.stats?.total_patients ??
        source.total_patients ??
        source.patients_count ??
        0,

      total_analyses:
        source.stats?.total_analyses ??
        source.total_analyses ??
        source.analyses_count ??
        0,

      pending_reviews:
        source.stats?.pending_reviews ??
        source.pending_reviews ??
        0,

      high_risk_cases:
        source.stats?.high_risk_cases ??
        source.high_risk_cases ??
        0,

      active_users:
        source.stats?.active_users ??
        source.active_users ??
        0,

      system_uptime:
        source.stats?.system_uptime ??
        source.system_uptime ??
        "—",

      resnet50_usage:
        source.stats?.resnet50_usage ??
        source.resnet50_usage ??
        0,

      densenet121_usage:
        source.stats?.densenet121_usage ??
        source.densenet121_usage ??
        0,
    },

    recent_activity:
      source.recent_activity ||
      source.activities ||
      source.activity ||
      [],
  };
}

export default function AdminDashboard({
  apiBaseUrl = API_BASE_URL,
  dashboardEndpoint = `${apiBaseUrl}/api/admin/dashboard`,
  onLogout,
}) {
  const [dashboard, setDashboard] = useState(DEFAULT_DATA);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");

  const [search, setSearch] = useState("");
  const [activityFilter, setActivityFilter] = useState("all");

  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [activeMenu, setActiveMenu] = useState("dashboard");

  const [managementItems, setManagementItems] = useState([]);
  const [managementCount, setManagementCount] = useState(0);
  const [managementLoading, setManagementLoading] = useState(false);
  const [managementError, setManagementError] = useState("");
  const [managementSearch, setManagementSearch] = useState("");
  const [editTarget, setEditTarget] = useState(null);
  const [editForm, setEditForm] = useState({
    full_name: "",
    email: "",
    password: "",
  });
  const [managementSaving, setManagementSaving] = useState(false);
  const [managementActionId, setManagementActionId] = useState("");


  const adminName = dashboard.admin.full_name || "Administrator";
  const initials = getInitials(adminName);

  const loadDashboard = async ({ showRefresh = false } = {}) => {
    try {
      setError("");

      if (showRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }

      const token = getToken();

      const response = await fetch(dashboardEndpoint, {
        method: "GET",
        headers: {
          Accept: "application/json",
          ...(token
            ? {
                Authorization: `Bearer ${token}`,
              }
            : {}),
        },
      });

      if (!response.ok) {
        throw new Error(
          `Unable to load admin dashboard (${response.status}).`
        );
      }

      const payload = await response.json();
      setDashboard(normaliseData(payload));
    } catch (err) {
      console.error("Admin dashboard error:", err);

      setError(
        err?.message ||
          "Unable to connect to the admin dashboard service."
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadDashboard();
  }, [dashboardEndpoint]);


  const managementRequest = async (path, options = {}) => {
    const token = getToken();
    const response = await fetch(`${apiBaseUrl}/api/admin${path}`, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(options.headers || {}),
      },
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || payload.message || `Request failed (${response.status}).`);
    }
    return payload;
  };

  const loadManagement = async (section = activeMenu) => {
    const endpoints = {
      users: "/patients",
      doctors: "/doctors",
      analyses: "/analyses",
    };
    if (!endpoints[section]) return;

    setManagementLoading(true);
    setManagementError("");
    try {
      const payload = await managementRequest(endpoints[section]);
      setManagementItems(payload.items || []);
      setManagementCount(payload.count ?? (payload.items || []).length);
    } catch (err) {
      setManagementError(err.message || "Unable to load management data.");
      setManagementItems([]);
      setManagementCount(0);
    } finally {
      setManagementLoading(false);
    }
  };

  useEffect(() => {
    if (["users", "doctors", "analyses"].includes(activeMenu)) {
      setManagementSearch("");
      loadManagement(activeMenu);
    }
  }, [activeMenu, apiBaseUrl]);

  const openEditUser = (user) => {
    setEditTarget(user);
    setEditForm({
      full_name: user.full_name || "",
      email: user.email || "",
      password: "",
    });
  };

  const saveManagedUser = async (event) => {
    event.preventDefault();
    if (!editTarget) return;

    const payload = {
      full_name: editForm.full_name.trim(),
      email: editForm.email.trim(),
    };
    if (editForm.password.trim()) payload.password = editForm.password;

    setManagementSaving(true);
    setManagementError("");
    try {
      await managementRequest(`/users/${editTarget.id}`, {
        method: "PATCH",
        body: JSON.stringify(payload),
      });
      setEditTarget(null);
      await loadManagement(activeMenu);
      await loadDashboard({ showRefresh: true });
    } catch (err) {
      setManagementError(err.message || "Unable to update account.");
    } finally {
      setManagementSaving(false);
    }
  };

  const deleteManagedRecord = async (item, type) => {
    const label = type === "analysis"
      ? "this analysis record"
      : `${item.full_name || "this account"} (${item.email || "no email"})`;
    if (!window.confirm(`Are you sure you want to permanently delete ${label}?`)) return;

    setManagementActionId(item.id);
    setManagementError("");
    try {
      const path = type === "analysis"
        ? `/analyses/${item.id}`
        : `/users/${item.id}`;
      await managementRequest(path, { method: "DELETE" });
      await loadManagement(activeMenu);
      await loadDashboard({ showRefresh: true });
    } catch (err) {
      setManagementError(err.message || "Unable to delete record.");
    } finally {
      setManagementActionId("");
    }
  };

  const filteredManagementItems = useMemo(() => {
    const query = managementSearch.trim().toLowerCase();
    if (!query) return managementItems;
    return managementItems.filter((item) =>
      [
        item.full_name, item.email, item.patient_name, item.patient_email,
        item.predicted_condition, item.analysis_status, item.model,
        item.risk_category, item.review_priority,
      ].filter(Boolean).join(" ").toLowerCase().includes(query)
    );
  }, [managementItems, managementSearch]);

  const renderManagementView = () => {
    const isPatients = activeMenu === "users";
    const isDoctors = activeMenu === "doctors";
    const isAnalyses = activeMenu === "analyses";
    const heading = isPatients ? "Patient Management"
      : isDoctors ? "Doctor Management" : "Analysis Management";
    const description = isPatients
      ? "View registered patients and update or remove their accounts."
      : isDoctors
      ? "View registered doctors and update or remove their accounts."
      : "Review DermAI image-analysis records, predictions, processing status, and remove unwanted records.";

    return (
      <>
        <div className="admin-welcome">
          <div>
            <span className="admin-page-tag">ADMINISTRATION</span>
            <h2>{heading}</h2>
            <p>{description}</p>
          </div>
          <div className="admin-system-status">
            <span className="admin-status-pulse" />
            <span>Live Database Records</span>
          </div>
        </div>

        {managementError && (
          <div className="admin-error-banner">
            <div className="admin-error-icon"><Icon name="warning" size={19} /></div>
            <div><strong>Management request failed</strong><p>{managementError}</p></div>
            <button onClick={() => loadManagement(activeMenu)}>Retry</button>
          </div>
        )}

        <section className="admin-panel admin-management-panel">
          <div className="admin-panel-header">
            <div>
              <span className="admin-panel-kicker">DATABASE MANAGEMENT</span>
              <h3>{isAnalyses ? "Analysis Records" : isPatients ? "Registered Patients" : "Registered Doctors"}</h3>
            </div>
            <div className="admin-management-count">
              <strong>{managementLoading ? "…" : formatNumber(managementCount)}</strong>
              <span>{isAnalyses ? "Total analyses" : isPatients ? "Total patients" : "Total doctors"}</span>
            </div>
          </div>

          <div className="admin-toolbar">
            <div className="admin-search">
              <Icon name="search" size={18} />
              <input
                type="text"
                placeholder={isAnalyses ? "Search patient, prediction, model, status..." : "Search name or email..."}
                value={managementSearch}
                onChange={(event) => setManagementSearch(event.target.value)}
              />
            </div>
            <button
              className={`admin-refresh-button ${managementLoading ? "spinning" : ""}`}
              onClick={() => loadManagement(activeMenu)}
              title="Refresh records"
              disabled={managementLoading}
            >
              <span>↻</span>
            </button>
          </div>

          <div className="admin-management-table-wrap">
            <table className="admin-management-table">
              <thead>
                {isAnalyses ? (
                  <tr>
                    <th>Patient</th><th>Prediction</th><th>Confidence</th>
                    <th>Model</th><th>Status</th><th>Date</th><th>Action</th>
                  </tr>
                ) : (
                  <tr><th>{isPatients ? "Patient" : "Doctor"}</th><th>Email</th><th>Registered</th><th>Actions</th></tr>
                )}
              </thead>
              <tbody>
                {managementLoading ? (
                  <tr><td colSpan={isAnalyses ? 7 : 4} className="admin-table-message">Loading records…</td></tr>
                ) : filteredManagementItems.length === 0 ? (
                  <tr><td colSpan={isAnalyses ? 7 : 4} className="admin-table-message">
                    {managementItems.length === 0 ? "No records found in the database." : "No records match your search."}
                  </td></tr>
                ) : filteredManagementItems.map((item) => (
                  isAnalyses ? (
                    <tr key={item.id}>
                      <td>
                        <strong>{item.patient_name || "Unknown patient"}</strong>
                        <small>{item.patient_email || "Patient account unavailable"}</small>
                      </td>
                      <td>
                        <strong>{item.predicted_condition || "Not available"}</strong>
                        <small>{item.risk_category || item.review_priority || "—"}</small>
                      </td>
                      <td>{item.confidence == null ? "—" : `${(Number(item.confidence) <= 1 ? Number(item.confidence) * 100 : Number(item.confidence)).toFixed(1)}%`}</td>
                      <td>{item.model || "—"}</td>
                      <td><span className={`admin-status-badge ${String(item.analysis_status || "pending").toLowerCase()}`}>{item.analysis_status || "pending"}</span></td>
                      <td>{formatDate(item.completed_at || item.started_at)}</td>
                      <td>
                        <button
                          className="admin-action-button delete"
                          onClick={() => deleteManagedRecord(item, "analysis")}
                          disabled={managementActionId === item.id}
                        >{managementActionId === item.id ? "Deleting…" : "Delete"}</button>
                      </td>
                    </tr>
                  ) : (
                    <tr key={item.id}>
                      <td><div className="admin-managed-person"><span>{getInitials(item.full_name || "U")}</span><strong>{item.full_name || "Unnamed account"}</strong></div></td>
                      <td>{item.email || "—"}</td>
                      <td>{formatDate(item.created_at)}</td>
                      <td>
                        <div className="admin-management-actions">
                          <button className="admin-action-button edit" onClick={() => openEditUser(item)}>Edit</button>
                          <button
                            className="admin-action-button delete"
                            onClick={() => deleteManagedRecord(item, "user")}
                            disabled={managementActionId === item.id}
                          >{managementActionId === item.id ? "Deleting…" : "Delete"}</button>
                        </div>
                      </td>
                    </tr>
                  )
                ))}
              </tbody>
            </table>
          </div>
          <div className="admin-management-footnote">
            Showing {formatNumber(filteredManagementItems.length)} of {formatNumber(managementCount)} records
            {isAnalyses ? " • Analysis deletion removes the selected analysis document only." : " • Account deletion revokes active sessions and prevents future sign-in."}
          </div>
        </section>

        {editTarget && (
          <div className="admin-modal-backdrop" onMouseDown={(event) => {
            if (event.target === event.currentTarget) setEditTarget(null);
          }}>
            <form className="admin-edit-modal" onSubmit={saveManagedUser}>
              <div className="admin-modal-heading">
                <div><span className="admin-panel-kicker">ACCOUNT SETTINGS</span><h3>Edit {isPatients ? "Patient" : "Doctor"}</h3></div>
                <button type="button" className="admin-modal-close" onClick={() => setEditTarget(null)}>×</button>
              </div>
              <label>Full name</label>
              <input value={editForm.full_name} onChange={(event) => setEditForm({ ...editForm, full_name: event.target.value })} required />
              <label>Email address</label>
              <input type="email" value={editForm.email} onChange={(event) => setEditForm({ ...editForm, email: event.target.value })} required />
              <label>New password <span>(leave blank to keep current password)</span></label>
              <input type="password" minLength={8} autoComplete="new-password" placeholder="Enter a new password (optional)" value={editForm.password} onChange={(event) => setEditForm({ ...editForm, password: event.target.value })} />
              <p className="admin-modal-note">Changing the email or password signs the account out of existing sessions.</p>
              <div className="admin-modal-actions">
                <button type="button" className="admin-action-button cancel" onClick={() => setEditTarget(null)}>Cancel</button>
                <button type="submit" className="admin-action-button save" disabled={managementSaving}>{managementSaving ? "Saving…" : "Save changes"}</button>
              </div>
            </form>
          </div>
        )}
      </>
    );
  };

  const filteredActivity = useMemo(() => {
    const activities = dashboard.recent_activity || [];

    return activities.filter((item) => {
      const searchableText = [
        item.title,
        item.description,
        item.user_name,
        item.user,
        item.type,
        item.action,
        item.status,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();

      const matchesSearch = searchableText.includes(
        search.trim().toLowerCase()
      );

      const type = String(
        item.type || item.action || item.status || ""
      ).toLowerCase();

      const matchesFilter =
        activityFilter === "all" ||
        type.includes(activityFilter.toLowerCase());

      return matchesSearch && matchesFilter;
    });
  }, [dashboard.recent_activity, search, activityFilter]);

  const handleLogout = () => {
    if (onLogout) {
      onLogout();
      return;
    }

    localStorage.removeItem("access_token");
    localStorage.removeItem("token");
    localStorage.removeItem("authToken");

    window.location.href = "/login";
  };

  const handleNavigation = (menu) => {
    setActiveMenu(menu);
    setMobileSidebarOpen(false);
  };

  const statCards = [
    {
      label: "Total Users",
      value: dashboard.stats.total_users,
      icon: "users",
      className: "blue",
      description: "Registered platform users",
    },

    {
      label: "Doctors",
      value: dashboard.stats.total_doctors,
      icon: "doctor",
      className: "green",
      description: "Registered medical professionals",
    },

    {
      label: "Patients",
      value: dashboard.stats.total_patients,
      icon: "users",
      className: "purple",
      description: "Patients using DermAI",
    },

    {
      label: "Total Analyses",
      value: dashboard.stats.total_analyses,
      icon: "analysis",
      className: "orange",
      description: "Skin image analyses",
    },
  ];

  const renderActivityView = () => {
    return (
      <>
        <div className="admin-welcome">
          <div>
            <span className="admin-page-tag">SYSTEM ACTIVITY</span>

            <h2>Activity Logs</h2>

            <p>
              Monitor recent users, doctors, analyses, and system
              activity.
            </p>
          </div>

          <div className="admin-system-status">
            <span className="admin-status-pulse" />
            <span>Live Data</span>
          </div>
        </div>

        <section className="admin-panel">
          <div className="admin-panel-header">
            <div>
              <span className="admin-panel-kicker">
                PLATFORM MONITORING
              </span>

              <h3>Recent Activity</h3>
            </div>

            <button
              className={`admin-refresh-button ${
                refreshing ? "spinning" : ""
              }`}
              onClick={() => loadDashboard({ showRefresh: true })}
              title="Refresh activity"
            >
              <span style={{ fontSize: "22px", lineHeight: 1 }}>
                ↻
              </span>
            </button>
          </div>

          <div className="admin-toolbar">
            <div className="admin-search">
              <Icon name="search" size={18} />

              <input
                type="text"
                placeholder="Search activity..."
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
            </div>

            <select
              className="admin-filter"
              value={activityFilter}
              onChange={(event) =>
                setActivityFilter(event.target.value)
              }
            >
              <option value="all">All Activity</option>
              <option value="user">Users</option>
              <option value="doctor">Doctors</option>
              <option value="analysis">Analyses</option>
              <option value="system">System</option>
            </select>
          </div>

          <div className="admin-activity-list">
            {loading ? (
              <>
                <div className="admin-skeleton-row" />
                <div className="admin-skeleton-row" />
                <div className="admin-skeleton-row" />
              </>
            ) : filteredActivity.length === 0 ? (
              <div className="admin-empty-state">
                <div className="admin-empty-icon">
                  <Icon name="activity" size={25} />
                </div>

                <h4>No activity found</h4>

                <p>
                  Recent platform activity will appear here once
                  available.
                </p>
              </div>
            ) : (
              filteredActivity.map((item, index) => {
                const type = String(
                  item.type || item.action || "system"
                ).toLowerCase();

                const isWarning =
                  type.includes("warning") ||
                  type.includes("risk") ||
                  type.includes("failed");

                return (
                  <div
                    className="admin-activity-row"
                    key={item.id || index}
                  >
                    <div
                      className={`admin-activity-icon ${
                        isWarning ? "warning" : ""
                      }`}
                    >
                      <Icon
                        name={
                          type.includes("doctor")
                            ? "doctor"
                            : type.includes("user")
                            ? "users"
                            : type.includes("analysis")
                            ? "analysis"
                            : "activity"
                        }
                        size={18}
                      />
                    </div>

                    <div className="admin-activity-info">
                      <strong>
                        {item.title ||
                          item.action ||
                          "System activity"}
                      </strong>

                      <p>
                        {item.description ||
                          item.user_name ||
                          item.user ||
                          "Platform event recorded"}
                      </p>
                    </div>

                    <div className="admin-activity-status">
                      {item.status && (
                        <span
                          className={`admin-status-badge ${String(
                            item.status
                          ).toLowerCase()}`}
                        >
                          {item.status}
                        </span>
                      )}

                      <span className="admin-activity-date">
                        {formatDate(
                          item.created_at ||
                            item.timestamp ||
                            item.date
                        )}
                      </span>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </section>
      </>
    );
  };

  const renderModelView = () => {
    return (
      <>
        <div className="admin-welcome">
          <div>
            <span className="admin-page-tag">AI INFRASTRUCTURE</span>

            <h2>AI Model</h2>

            <p>
              Monitor the AI models currently used by DermAI.
            </p>
          </div>

          <div className="admin-system-status">
            <span className="admin-status-pulse" />
            <span>System Operational</span>
          </div>
        </div>

        <section className="admin-panel">
          <div className="admin-panel-header">
            <div>
              <span className="admin-panel-kicker">
                MODEL MONITORING
              </span>

              <h3>DermAI Models</h3>
            </div>

            <button
              className={`admin-refresh-button ${
                refreshing ? "spinning" : ""
              }`}
              onClick={() => loadDashboard({ showRefresh: true })}
              title="Refresh model data"
            >
              <span style={{ fontSize: "22px", lineHeight: 1 }}>
                ↻
              </span>
            </button>
          </div>

          <div className="admin-bottom-grid">
            <div className="admin-panel admin-health-panel">
              <div className="admin-panel-header">
                <div>
                  <span className="admin-panel-kicker">
                    PRIMARY CLASSIFIER
                  </span>

                  <h3>ResNet-50</h3>
                </div>

                <div className="admin-health-badge">
                  <span />
                  Active
                </div>
              </div>

              <div className="admin-health-list">
                <div className="admin-health-item">
                  <div>
                    <span>Total Analyses</span>

                    <strong>
                      {formatNumber(
                        dashboard.stats.resnet50_usage ||
                          dashboard.stats.total_analyses
                      )}
                    </strong>
                  </div>
                </div>

                <div className="admin-health-item">
                  <div>
                    <span>Model Status</span>

                    <strong>Operational</strong>
                  </div>
                </div>

                <div className="admin-health-item">
                  <div>
                    <span>Architecture</span>

                    <strong>ResNet-50</strong>
                  </div>
                </div>
              </div>
            </div>

            <div className="admin-panel admin-health-panel">
              <div className="admin-panel-header">
                <div>
                  <span className="admin-panel-kicker">
                    COMPARISON MODEL
                  </span>

                  <h3>DenseNet-121</h3>
                </div>

                <div className="admin-health-badge">
                  <span />
                  Available
                </div>
              </div>

              <div className="admin-health-list">
                <div className="admin-health-item">
                  <div>
                    <span>Model Usage</span>

                    <strong>
                      {formatNumber(
                        dashboard.stats.densenet121_usage
                      )}
                    </strong>
                  </div>
                </div>

                <div className="admin-health-item">
                  <div>
                    <span>Model Status</span>

                    <strong>Operational</strong>
                  </div>
                </div>

                <div className="admin-health-item">
                  <div>
                    <span>Architecture</span>

                    <strong>DenseNet-121</strong>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </>
    );
  };

  const renderSettingsView = () => {
    return (
      <>
        <div className="admin-welcome">
          <div>
            <span className="admin-page-tag">
              SYSTEM CONFIGURATION
            </span>

            <h2>System Settings</h2>

            <p>
              View the current DermAI administration and system
              configuration.
            </p>
          </div>

          <div className="admin-system-status">
            <span className="admin-status-pulse" />
            <span>System Operational</span>
          </div>
        </div>

        <section className="admin-panel">
          <div className="admin-panel-header">
            <div>
              <span className="admin-panel-kicker">
                CONFIGURATION
              </span>

              <h3
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                }}
              >
                <Icon name="settings" size={19} />
                System Settings
              </h3>
            </div>

            <button
              className={`admin-refresh-button ${
                refreshing ? "spinning" : ""
              }`}
              onClick={() => loadDashboard({ showRefresh: true })}
              title="Refresh settings"
            >
              <span style={{ fontSize: "22px", lineHeight: 1 }}>
                ↻
              </span>
            </button>
          </div>

          <div className="admin-secondary-grid">
            <div className="admin-mini-card">
              <div className="admin-mini-icon active">
                <Icon name="shield" size={19} />
              </div>

              <div>
                <span>System Status</span>
                <strong>Operational</strong>
              </div>
            </div>

            <div className="admin-mini-card">
              <div className="admin-mini-icon active">
                <Icon name="users" size={19} />
              </div>

              <div>
                <span>Active Users</span>

                <strong>
                  {formatNumber(dashboard.stats.active_users)}
                </strong>
              </div>
            </div>

            <div className="admin-mini-card">
              <div className="admin-mini-icon pending">
                <Icon name="analysis" size={19} />
              </div>

              <div>
                <span>Total Analyses</span>

                <strong>
                  {formatNumber(dashboard.stats.total_analyses)}
                </strong>
              </div>
            </div>

            <div className="admin-mini-card">
              <div className="admin-mini-icon pending">
                <Icon name="activity" size={19} />
              </div>

              <div>
                <span>Pending Reviews</span>

                <strong>
                  {formatNumber(dashboard.stats.pending_reviews)}
                </strong>
              </div>
            </div>
          </div>

          <div
            className="admin-panel"
            style={{ marginTop: "20px" }}
          >
            <div className="admin-health-list">
              <div className="admin-health-item">
                <div>
                  <span>Administrator</span>

                  <strong>{adminName}</strong>
                </div>
              </div>

              <div className="admin-health-item">
                <div>
                  <span>Admin Email</span>

                  <strong>
                    {dashboard.admin.email || "Not available"}
                  </strong>
                </div>
              </div>

              <div className="admin-health-item">
                <div>
                  <span>System Uptime</span>

                  <strong>
                    {dashboard.stats.system_uptime || "—"}
                  </strong>
                </div>
              </div>

              <div className="admin-health-item">
                <div>
                  <span>API Endpoint</span>

                  <strong>{dashboardEndpoint}</strong>
                </div>
              </div>
            </div>
          </div>
        </section>
      </>
    );
  };

  const renderSystemView = () => {
    if (activeMenu === "activity") {
      return renderActivityView();
    }

    if (activeMenu === "model") {
      return renderModelView();
    }

    if (activeMenu === "settings") {
      return renderSettingsView();
    }

    return null;
  };

  return (
    <div className="admin-dashboard">
      <aside
        className={`admin-sidebar ${
          mobileSidebarOpen ? "admin-sidebar-open" : ""
        }`}
      >
        <div className="admin-brand">
          <div className="admin-brand-mark">
            <span>D</span>
          </div>

          <div>
            <h2>DermAI</h2>
            <span>Administration</span>
          </div>
        </div>

        <div className="admin-sidebar-section">
          <p className="admin-sidebar-label">MAIN</p>

          <button
            className={`admin-nav-item ${
              activeMenu === "dashboard" ? "active" : ""
            }`}
            onClick={() => handleNavigation("dashboard")}
          >
            <Icon name="dashboard" size={19} />
            <span>Dashboard</span>
          </button>

          <button
            className={`admin-nav-item ${
              activeMenu === "users" ? "active" : ""
            }`}
            onClick={() => handleNavigation("users")}
          >
            <Icon name="users" size={19} />
            <span>Patients</span>
          </button>

          <button
            className={`admin-nav-item ${
              activeMenu === "doctors" ? "active" : ""
            }`}
            onClick={() => handleNavigation("doctors")}
          >
            <Icon name="doctor" size={19} />
            <span>Doctors</span>
          </button>

          <button
            className={`admin-nav-item ${
              activeMenu === "analyses" ? "active" : ""
            }`}
            onClick={() => handleNavigation("analyses")}
          >
            <Icon name="analysis" size={19} />
            <span>Analyses</span>
          </button>
        </div>

        <div className="admin-sidebar-section">
          <p className="admin-sidebar-label">SYSTEM</p>

          <button
            className={`admin-nav-item ${
              activeMenu === "activity" ? "active" : ""
            }`}
            onClick={() => handleNavigation("activity")}
          >
            <Icon name="activity" size={19} />
            <span>Activity Logs</span>
          </button>

          <button
            className={`admin-nav-item ${
              activeMenu === "model" ? "active" : ""
            }`}
            onClick={() => handleNavigation("model")}
          >
            <Icon name="model" size={19} />
            <span>AI Model</span>
          </button>

          <button
            className={`admin-nav-item ${
              activeMenu === "settings" ? "active" : ""
            }`}
            onClick={() => handleNavigation("settings")}
          >
            <Icon name="settings" size={19} />
            <span>Settings</span>
          </button>
        </div>

        <div className="admin-sidebar-bottom">
          <div className="admin-security-card">
            <div className="admin-security-icon">
              <Icon name="shield" size={20} />
            </div>

            <div>
              <strong>System Secure</strong>
              <span>DermAI services protected</span>
            </div>
          </div>

          <button className="admin-logout" onClick={handleLogout}>
            <Icon name="logout" size={18} />
            <span>Logout</span>
          </button>
        </div>
      </aside>

      {mobileSidebarOpen && (
        <button
          className="admin-sidebar-overlay"
          onClick={() => setMobileSidebarOpen(false)}
          aria-label="Close navigation"
        />
      )}

      <main className="admin-main">
        <header className="admin-topbar">
          <div className="admin-topbar-left">
            <button
              className="admin-mobile-menu"
              onClick={() => setMobileSidebarOpen(true)}
              aria-label="Open navigation"
            >
              <Icon name="menu" size={22} />
            </button>

            <div>
              <p className="admin-breadcrumb">Administration</p>

              <h1>
                {activeMenu === "activity"
                  ? "Activity Logs"
                  : activeMenu === "model"
                  ? "AI Model"
                  : activeMenu === "settings"
                  ? "System Settings"
                  : activeMenu === "users"
                  ? "Patient Management"
                  : activeMenu === "doctors"
                  ? "Doctor Management"
                  : activeMenu === "analyses"
                  ? "Analysis Management"
                  : "Admin Dashboard"}
              </h1>
            </div>
          </div>

          <div className="admin-topbar-right">
            <button
              className={`admin-refresh-button ${
                refreshing ? "spinning" : ""
              }`}
              onClick={() => loadDashboard({ showRefresh: true })}
              title="Refresh dashboard"
            >
              <span style={{ fontSize: "22px", lineHeight: 1 }}>
                ↻
              </span>
            </button>

            <NotificationBell variant="admin" />

            <div className="admin-profile">
              <div className="admin-avatar">{initials}</div>

              <div className="admin-profile-text">
                <strong>{adminName}</strong>
                <span>Administrator</span>
              </div>
            </div>
          </div>
        </header>

        <section className="admin-content">
          {activeMenu === "dashboard" ? (
            <>
              <div className="admin-welcome">
                <div>
                  <span className="admin-page-tag">
                    SYSTEM OVERVIEW
                  </span>

                  <h2>
                    Welcome back, {adminName.split(" ")[0]}.
                  </h2>

                  <p>
                    Monitor DermAI users, analyses, doctors, and AI
                    system activity from one place.
                  </p>
                </div>

                <div className="admin-system-status">
                  <span className="admin-status-pulse" />
                  <span>System Operational</span>
                </div>
              </div>

              {error && (
                <div className="admin-error-banner">
                  <div className="admin-error-icon">
                    <Icon name="warning" size={19} />
                  </div>

                  <div>
                    <strong>Dashboard connection issue</strong>

                    <p>{error}</p>
                  </div>

                  <button onClick={() => loadDashboard()}>
                    Retry
                  </button>
                </div>
              )}

              <div className="admin-stat-grid">
                {statCards.map((card) => (
                  <div
                    className="admin-stat-card"
                    key={card.label}
                  >
                    <div className="admin-stat-top">
                      <div
                        className={`admin-stat-icon ${card.className}`}
                      >
                        <Icon name={card.icon} size={21} />
                      </div>

                      <span className="admin-stat-label">
                        {card.label}
                      </span>
                    </div>

                    <div className="admin-stat-number">
                      {loading ? (
                        <span className="admin-loading-number">
                          ...
                        </span>
                      ) : (
                        formatNumber(card.value)
                      )}
                    </div>

                    <p>{card.description}</p>
                  </div>
                ))}
              </div>

              <div className="admin-secondary-grid">
                <div className="admin-mini-card">
                  <div className="admin-mini-icon pending">
                    <Icon name="activity" size={19} />
                  </div>

                  <div>
                    <span>Pending Reviews</span>

                    <strong>
                      {formatNumber(
                        dashboard.stats.pending_reviews
                      )}
                    </strong>
                  </div>
                </div>

                <div className="admin-mini-card">
                  <div className="admin-mini-icon danger">
                    <Icon name="warning" size={19} />
                  </div>

                  <div>
                    <span>High Risk Cases</span>

                    <strong>
                      {formatNumber(
                        dashboard.stats.high_risk_cases
                      )}
                    </strong>
                    <small style={{ display: "block", marginTop: 4, color: "#64748b" }}>
                      Based on backend risk-assessment records
                    </small>
                  </div>
                </div>

                <div className="admin-mini-card">
                  <div className="admin-mini-icon active">
                    <Icon name="users" size={19} />
                  </div>

                  <div>
                    <span>Active Users</span>

                    <strong>
                      {formatNumber(
                        dashboard.stats.active_users
                      )}
                    </strong>
                  </div>
                </div>

                <div className="admin-mini-card">
                  <div className="admin-mini-icon uptime">
                    <Icon name="shield" size={19} />
                  </div>

                  <div>
                    <span>System Uptime</span>

                    <strong>
                      {dashboard.stats.system_uptime || "—"}
                    </strong>
                  </div>
                </div>
              </div>

              <section className="admin-panel">
                <div className="admin-panel-header">
                  <div>
                    <span className="admin-panel-kicker">
                      PLATFORM MONITORING
                    </span>

                    <h3>Recent Activity</h3>
                  </div>

                  <button
                    className="admin-view-all"
                    onClick={() => handleNavigation("activity")}
                  >
                    View all

                    <Icon name="arrow" size={16} />
                  </button>
                </div>

                <div className="admin-toolbar">
                  <div className="admin-search">
                    <Icon name="search" size={18} />

                    <input
                      type="text"
                      placeholder="Search activity..."
                      value={search}
                      onChange={(event) =>
                        setSearch(event.target.value)
                      }
                    />
                  </div>

                  <select
                    className="admin-filter"
                    value={activityFilter}
                    onChange={(event) =>
                      setActivityFilter(event.target.value)
                    }
                  >
                    <option value="all">All Activity</option>
                    <option value="user">Users</option>
                    <option value="doctor">Doctors</option>
                    <option value="analysis">Analyses</option>
                    <option value="system">System</option>
                  </select>
                </div>

                <div className="admin-activity-list">
                  {loading ? (
                    <>
                      <div className="admin-skeleton-row" />
                      <div className="admin-skeleton-row" />
                      <div className="admin-skeleton-row" />
                    </>
                  ) : filteredActivity.length === 0 ? (
                    <div className="admin-empty-state">
                      <div className="admin-empty-icon">
                        <Icon name="activity" size={25} />
                      </div>

                      <h4>No activity found</h4>

                      <p>
                        Recent platform activity will appear here
                        once available.
                      </p>
                    </div>
                  ) : (
                    filteredActivity
                      .slice(0, 8)
                      .map((item, index) => {
                        const type = String(
                          item.type ||
                            item.action ||
                            "system"
                        ).toLowerCase();

                        const isWarning =
                          type.includes("warning") ||
                          type.includes("risk") ||
                          type.includes("failed");

                        return (
                          <div
                            className="admin-activity-row"
                            key={item.id || index}
                          >
                            <div
                              className={`admin-activity-icon ${
                                isWarning ? "warning" : ""
                              }`}
                            >
                              <Icon
                                name={
                                  type.includes("doctor")
                                    ? "doctor"
                                    : type.includes("user")
                                    ? "users"
                                    : type.includes("analysis")
                                    ? "analysis"
                                    : "activity"
                                }
                                size={18}
                              />
                            </div>

                            <div className="admin-activity-info">
                              <strong>
                                {item.title ||
                                  item.action ||
                                  "System activity"}
                              </strong>

                              <p>
                                {item.description ||
                                  item.user_name ||
                                  item.user ||
                                  "Platform event recorded"}
                              </p>
                            </div>

                            <div className="admin-activity-status">
                              {item.status && (
                                <span
                                  className={`admin-status-badge ${String(
                                    item.status
                                  ).toLowerCase()}`}
                                >
                                  {item.status}
                                </span>
                              )}

                              <span className="admin-activity-date">
                                {formatDate(
                                  item.created_at ||
                                    item.timestamp ||
                                    item.date
                                )}
                              </span>
                            </div>
                          </div>
                        );
                      })
                  )}
                </div>
              </section>

              <section className="admin-bottom-grid">
                <div className="admin-panel admin-health-panel">
                  <div className="admin-panel-header">
                    <div>
                      <span className="admin-panel-kicker">
                        AI INFRASTRUCTURE
                      </span>

                      <h3>DermAI System Health</h3>
                    </div>

                    <div className="admin-health-badge">
                      <span />
                      Healthy
                    </div>
                  </div>

                  <div className="admin-health-list">
                    <div className="admin-health-item">
                      <div>
                        <span>API Service</span>
                        <strong>Operational</strong>
                      </div>

                      <div className="admin-health-bar">
                        <span style={{ width: "96%" }} />
                      </div>

                      <b>96%</b>
                    </div>

                    <div className="admin-health-item">
                      <div>
                        <span>AI Inference</span>
                        <strong>Operational</strong>
                      </div>

                      <div className="admin-health-bar">
                        <span style={{ width: "93%" }} />
                      </div>

                      <b>93%</b>
                    </div>

                    <div className="admin-health-item">
                      <div>
                        <span>Database</span>
                        <strong>Operational</strong>
                      </div>

                      <div className="admin-health-bar">
                        <span style={{ width: "98%" }} />
                      </div>

                      <b>98%</b>
                    </div>
                  </div>
                </div>

                <div className="admin-panel admin-quick-panel">
                  <div className="admin-panel-header">
                    <div>
                      <span className="admin-panel-kicker">
                        ADMINISTRATION
                      </span>

                      <h3>Quick Actions</h3>
                    </div>
                  </div>

                  <div className="admin-quick-actions">
                    <button
                      onClick={() => handleNavigation("users")}
                    >
                      <span className="quick-action-icon blue">
                        <Icon name="users" size={18} />
                      </span>

                      <span>
                        <strong>Manage Patients</strong>
                        <small>
                          View and manage accounts
                        </small>
                      </span>

                      <Icon name="arrow" size={16} />
                    </button>

                    <button
                      onClick={() =>
                        handleNavigation("doctors")
                      }
                    >
                      <span className="quick-action-icon green">
                        <Icon name="doctor" size={18} />
                      </span>

                      <span>
                        <strong>Manage Doctors</strong>
                        <small>
                          Review medical professionals
                        </small>
                      </span>

                      <Icon name="arrow" size={16} />
                    </button>

                    <button
                      onClick={() => handleNavigation("model")}
                    >
                      <span className="quick-action-icon purple">
                        <Icon name="model" size={18} />
                      </span>

                      <span>
                        <strong>AI Model</strong>
                        <small>
                          Monitor model performance
                        </small>
                      </span>

                      <Icon name="arrow" size={16} />
                    </button>

                    <button
                      onClick={() =>
                        handleNavigation("settings")
                      }
                    >
                      <span className="quick-action-icon orange">
                        <Icon name="settings" size={18} />
                      </span>

                      <span>
                        <strong>System Settings</strong>
                        <small>
                          Configure DermAI
                        </small>
                      </span>

                      <Icon name="arrow" size={16} />
                    </button>
                  </div>
                </div>
              </section>

              <footer className="admin-footer">
                <span>DermAI Administration Panel</span>

                <span>
                  Medical AI system • Admin access
                </span>
              </footer>
            </>
          ) : activeMenu === "activity" ? (
            renderActivityView()
          ) : activeMenu === "model" ? (
            renderModelView()
          ) : activeMenu === "settings" ? (
            renderSettingsView()
          ) : (
            renderManagementView()
          )}
        </section>
      </main>
    </div>
  );
}
