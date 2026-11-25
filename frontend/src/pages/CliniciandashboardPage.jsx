import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { reviewsApi } from '../api/client';
import {
  Activity,
  CheckCircle2,
  Clock,
  AlertTriangle,
  ShieldAlert,
  ArrowRight,
  Filter,
  RefreshCw,
  Eye,
  FileCheck,
  TrendingUp,
  AlertCircle,
  ImageIcon,
} from 'lucide-react';

export const CliniciandashboardPage = () => {
  const { user, isAuthenticated, loading: authLoading } = useAuth();
  const navigate = useNavigate();

  // State
  const [stats, setStats] = useState({
    total_cases_reviewed: 0,
    pending_review: 0,
    ai_accuracy: 100.0,
    overrides_this_week: 0,
  });
  const [cases, setCases] = useState([]);
  const [activeTab, setActiveTab] = useState('all'); // 'all', 'pending', 'reviewed', 'overridden'
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  // Role validation
  useEffect(() => {
    if (!authLoading && isAuthenticated) {
      const role = (user?.role || '').toLowerCase();
      if (role !== 'clinician' && role !== 'admin') {
        navigate('/dashboard', { replace: true });
      }
    }
  }, [user, isAuthenticated, authLoading, navigate]);

  const fetchData = async () => {
    setError(null);
    try {
      // 1. Fetch stats
      const statsRes = await reviewsApi.getStats();
      setStats(statsRes);

      // 2. Fetch cases by active filter tab
      const casesRes = await reviewsApi.getCases(activeTab);
      const items = Array.isArray(casesRes) ? casesRes : casesRes.items || [];
      setCases(items);
    } catch (err) {
      console.error('Failed to load clinician dashboard data:', err);
      setError(err.response?.data?.detail || 'Failed to fetch clinician dashboard data.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    document.title = 'Clinician Dashboard | DermAI';
    if (isAuthenticated) {
      fetchData();
    }
  }, [activeTab, isAuthenticated]);

  const handleRefresh = () => {
    setRefreshing(true);
    fetchData();
  };

  // Helper for risk badge
  const renderRiskBadge = (level) => {
    const norm = (level || 'LOW').toUpperCase();
    if (norm === 'HIGH') {
      return (
        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-bold bg-red-100 text-red-800 border border-red-200">
          <ShieldAlert className="w-3 h-3 text-red-600" />
          HIGH
        </span>
      );
    }
    if (norm === 'MEDIUM') {
      return (
        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-bold bg-amber-100 text-amber-800 border border-amber-200">
          <AlertTriangle className="w-3 h-3 text-amber-600" />
          MED
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-100 text-emerald-800 border border-emerald-200">
        <CheckCircle2 className="w-3 h-3 text-emerald-600" />
        LOW
      </span>
    );
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-sky-100 text-sky-800">
              Clinician Portal
            </span>
            <span className="text-xs text-slate-400">•</span>
            <span className="text-xs text-slate-500 font-medium">Diagnostic Triage Queue</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 mt-1">
            Clinical Review & Decision Support
          </h1>
          <p className="text-sm text-slate-500">
            Review deep learning skin lesion predictions, inspect Grad-CAM heatmaps, and record diagnostic sign-offs.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl border border-slate-300 bg-white hover:bg-slate-50 text-slate-700 text-sm font-medium transition-all shadow-sm"
          >
            <RefreshCw className={`w-4 h-4 text-slate-500 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-800 flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 shrink-0" />
          <div className="text-sm font-medium">{error}</div>
        </div>
      )}

      {/* 4 Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {/* Card 1: Total Cases Reviewed */}
        <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
              Total Cases Reviewed
            </span>
            <div className="w-9 h-9 rounded-xl bg-sky-50 text-sky-600 flex items-center justify-center">
              <FileCheck className="w-5 h-5" />
            </div>
          </div>
          <div className="text-3xl font-extrabold text-slate-900">
            {stats.total_cases_reviewed}
          </div>
          <div className="text-xs text-slate-500 flex items-center gap-1">
            <span>Verified expert evaluations</span>
          </div>
        </div>

        {/* Card 2: Pending Review */}
        <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
              Pending Review
            </span>
            <div className="w-9 h-9 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center">
              <Clock className="w-5 h-5" />
            </div>
          </div>
          <div className="text-3xl font-extrabold text-amber-600">
            {stats.pending_review}
          </div>
          <div className="text-xs text-slate-500 flex items-center gap-1">
            <span>Awaiting clinician confirmation</span>
          </div>
        </div>

        {/* Card 3: AI Accuracy */}
        <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
              AI Accuracy
            </span>
            <div className="w-9 h-9 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center">
              <TrendingUp className="w-5 h-5" />
            </div>
          </div>
          <div className="text-3xl font-extrabold text-emerald-600">
            {stats.ai_accuracy}%
          </div>
          <div className="text-xs text-slate-500 flex items-center gap-1">
            <span>Clinician agreement with AI pick</span>
          </div>
        </div>

        {/* Card 4: Overrides This Week */}
        <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
              Overrides This Week
            </span>
            <div className="w-9 h-9 rounded-xl bg-purple-50 text-purple-600 flex items-center justify-center">
              <Activity className="w-5 h-5" />
            </div>
          </div>
          <div className="text-3xl font-extrabold text-purple-600">
            {stats.overrides_this_week}
          </div>
          <div className="text-xs text-slate-500 flex items-center gap-1">
            <span>Diagnoses updated by doctor</span>
          </div>
        </div>
      </div>

      {/* Case Table Section with Filter Tabs */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden space-y-0">
        {/* Filter Tabs Header */}
        <div className="px-6 py-4 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-slate-50/50">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-slate-500" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-600">
              Filter Queue:
            </span>
          </div>

          <div className="flex items-center gap-1.5 flex-wrap">
            {[
              { id: 'all', label: 'All Cases' },
              { id: 'pending', label: 'Pending' },
              { id: 'reviewed', label: 'Reviewed' },
              { id: 'overridden', label: 'Overridden' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all ${
                  activeTab === tab.id
                    ? 'bg-sky-600 text-white shadow-sm'
                    : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Table Content */}
        {loading ? (
          <div className="py-20 flex flex-col items-center justify-center">
            <div className="w-8 h-8 border-4 border-sky-200 border-t-sky-600 rounded-full animate-spin"></div>
            <p className="mt-3 text-xs font-medium text-slate-500">Loading triage cases...</p>
          </div>
        ) : cases.length === 0 ? (
          <div className="py-16 text-center space-y-2">
            <ImageIcon className="w-10 h-10 mx-auto text-slate-300" />
            <h3 className="text-sm font-bold text-slate-700">No cases found in this view</h3>
            <p className="text-xs text-slate-400 max-w-sm mx-auto">
              {activeTab === 'pending'
                ? 'All submitted skin lesion cases have been reviewed by a clinician!'
                : 'No cases match the selected filter category.'}
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-slate-600">
              <thead className="bg-slate-50 border-b border-slate-200 text-xs font-bold uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="py-3 px-4">Image</th>
                  <th className="py-3 px-4">Patient / Case</th>
                  <th className="py-3 px-4">AI Prediction</th>
                  <th className="py-3 px-4">Clinician Confirmation</th>
                  <th className="py-3 px-4">Risk Level</th>
                  <th className="py-3 px-4">Date</th>
                  <th className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {cases.map((c) => (
                  <tr key={c.image_id} className="hover:bg-slate-50/70 transition-colors">
                    {/* Thumbnail */}
                    <td className="py-3 px-4">
                      <div className="w-12 h-12 rounded-lg bg-slate-100 border border-slate-200 overflow-hidden flex items-center justify-center shrink-0">
                        <img
                          src={`http://localhost:8000/images/${c.image_id}/content`}
                          alt={`Lesion #${c.image_id}`}
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            e.target.onerror = null;
                            e.target.style.display = 'none';
                            e.target.parentNode.innerHTML = '<span class="text-[10px] text-slate-400 font-mono">#' + c.image_id + '</span>';
                          }}
                        />
                      </div>
                    </td>

                    {/* Patient Name */}
                    <td className="py-3 px-4">
                      <div className="font-semibold text-slate-800">{c.patient_name}</div>
                      <div className="text-xs text-slate-400">Case #{c.case_id}</div>
                    </td>

                    {/* AI Prediction */}
                    <td className="py-3 px-4">
                      <div className="font-medium text-slate-900">
                        {c.predicted_class_name} <span className="text-xs text-slate-400">({c.predicted_class?.toUpperCase()})</span>
                      </div>
                      <div className="text-xs text-slate-500">
                        Confidence: {(c.confidence * 100).toFixed(1)}%
                      </div>
                    </td>

                    {/* Clinician Confirmation */}
                    <td className="py-3 px-4">
                      {c.reviewed ? (
                        <div className="space-y-0.5">
                          <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-bold ${
                            c.overridden
                              ? 'bg-purple-100 text-purple-800 border border-purple-200'
                              : 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                          }`}>
                            <CheckCircle2 className="w-3 h-3" />
                            {c.confirmed_class_name || c.confirmed_class?.toUpperCase()}
                            {c.overridden && ' (Overridden)'}
                          </span>
                          {c.notes && (
                            <div className="text-[11px] text-slate-500 truncate max-w-xs" title={c.notes}>
                              "{c.notes}"
                            </div>
                          )}
                        </div>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-50 text-amber-700 border border-amber-200">
                          <Clock className="w-3 h-3 text-amber-500" />
                          Pending Review
                        </span>
                      )}
                    </td>

                    {/* Risk Level */}
                    <td className="py-3 px-4">
                      {renderRiskBadge(c.risk_level)}
                    </td>

                    {/* Date */}
                    <td className="py-3 px-4 text-xs text-slate-500">
                      {c.date ? new Date(c.date).toLocaleDateString() : 'N/A'}
                    </td>

                    {/* Action Button */}
                    <td className="py-3 px-4 text-right">
                      <Link
                        to={`/review/${c.image_id}`}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold bg-sky-50 text-sky-700 hover:bg-sky-600 hover:text-white transition-all border border-sky-200 hover:border-transparent shadow-sm"
                      >
                        <Eye className="w-3.5 h-3.5" />
                        {c.reviewed ? 'Edit Review' : 'Review'}
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export const ClinicianDashboardPage = CliniciandashboardPage;
export default CliniciandashboardPage;
