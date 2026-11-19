import React, { useState, useEffect, useCallback } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { dashboardApi } from '../api/client';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import {
  Activity,
  AlertTriangle,
  Calendar,
  CheckCircle,
  Clock,
  ExternalLink,
  FolderOpen,
  Percent,
  RefreshCw,
  TrendingUp,
  ShieldAlert,
  Sparkles,
} from 'lucide-react';

export const DashboardPage = () => {
  const { user } = useAuth();
  const navigate = useNavigate();

  // Role guard: redirect patients away to their history page
  useEffect(() => {
    if (user && user.role === 'patient') {
      navigate('/history', { replace: true });
    }
  }, [user, navigate]);

  const [stats, setStats] = useState({
    total_cases: 0,
    cases_this_week: 0,
    cases_this_month: 0,
    avg_confidence: 0,
    high_risk_count: 0,
    pending_review_count: 0,
  });
  const [classDist, setClassDist] = useState([]);
  const [confidenceDist, setConfidenceDist] = useState([]);
  const [casesOverTime, setCasesOverTime] = useState([]);
  const [recentCases, setRecentCases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const [lastRefreshed, setLastRefreshed] = useState(null);

  const loadDashboardData = useCallback(async (isManual = false) => {
    if (isManual) setRefreshing(true);
    setError('');

    try {
      const [statsData, classData, confData, timeData, recentData] = await Promise.all([
        dashboardApi.getStats(),
        dashboardApi.getClassDistribution(),
        dashboardApi.getConfidenceDistribution(),
        dashboardApi.getCasesOverTime(),
        dashboardApi.getRecentCases(),
      ]);

      setStats(statsData);
      setClassDist(classData);
      setConfidenceDist(confData);
      setCasesOverTime(timeData);
      setRecentCases(recentData);
      setLastRefreshed(new Date());
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
      setError(
        err.response?.data?.detail ||
          'Failed to load clinician dashboard. Please verify connection and privileges.'
      );
    } finally {
      setLoading(false);
      if (isManual) setRefreshing(false);
    }
  }, []);

  // Initial load and 60-second polling interval
  useEffect(() => {
    if (user?.role === 'patient') return;

    loadDashboardData();
    const interval = setInterval(() => {
      loadDashboardData();
    }, 60000);

    return () => clearInterval(interval);
  }, [user, loadDashboardData]);

  // Format date helper for line chart axis
  const formatChartDate = (dateStr) => {
    if (!dateStr) return '';
    const parts = dateStr.split('-');
    if (parts.length === 3) {
      const d = new Date(parts[0], parts[1] - 1, parts[2]);
      return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    }
    return dateStr;
  };

  const getRiskBadge = (risk) => {
    const level = (risk || 'LOW').toUpperCase();
    if (level === 'HIGH') {
      return (
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold bg-rose-100 text-rose-800 border border-rose-200">
          <ShieldAlert className="w-3 h-3 mr-1 text-rose-600" />
          HIGH
        </span>
      );
    }
    if (level === 'MEDIUM') {
      return (
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold bg-amber-100 text-amber-800 border border-amber-200">
          <AlertTriangle className="w-3 h-3 mr-1 text-amber-600" />
          MEDIUM
        </span>
      );
    }
    return (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-100 text-emerald-800 border border-emerald-200">
        <CheckCircle className="w-3 h-3 mr-1 text-emerald-600" />
        LOW
      </span>
    );
  };

  if (user?.role === 'patient') {
    return null; // Will redirect via useEffect
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10 space-y-8">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-sky-950 to-indigo-950 rounded-2xl p-6 sm:p-8 text-white shadow-xl shadow-slate-900/10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-sky-500/20 text-sky-200 text-xs font-semibold backdrop-blur-sm mb-3 border border-sky-400/30">
            <Sparkles className="w-3.5 h-3.5 text-cyan-300" />
            <span>Clinician Analytics & Triage Intelligence</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight">
            Clinical Screening Dashboard
          </h1>
          <p className="text-slate-300 text-sm mt-1 max-w-2xl">
            Real-time multi-class screening metrics, automated risk stratification distributions,
            and longitudinal case trends.
          </p>
        </div>

        <div className="flex items-center space-x-3 self-end sm:self-center">
          {lastRefreshed && (
            <span className="text-xs text-slate-400 hidden sm:inline-flex items-center space-x-1">
              <Clock className="w-3.5 h-3.5" />
              <span>Updated {lastRefreshed.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
            </span>
          )}
          <button
            onClick={() => loadDashboardData(true)}
            disabled={refreshing || loading}
            className="inline-flex items-center space-x-1.5 px-4 py-2.5 rounded-xl bg-white/10 hover:bg-white/20 text-white text-xs font-semibold backdrop-blur-sm border border-white/15 transition-all disabled:opacity-50"
            title="Auto-refreshes every 60 seconds"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
            <span>{refreshing ? 'Refreshing...' : 'Refresh'}</span>
          </button>
        </div>
      </div>

      {/* Error notification */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-sm flex items-start space-x-3">
          <AlertTriangle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="font-semibold">Unable to load dashboard data</p>
            <p className="text-xs text-rose-700 mt-0.5">{error}</p>
          </div>
          <button
            onClick={() => loadDashboardData(true)}
            className="text-xs font-semibold underline text-rose-800 hover:text-rose-950"
          >
            Retry
          </button>
        </div>
      )}

      {/* 4 Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        {/* Total Cases */}
        <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-xs hover:border-slate-300 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Total Cases
            </span>
            <div className="w-10 h-10 rounded-xl bg-sky-50 text-sky-600 flex items-center justify-center">
              <FolderOpen className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-3xl font-extrabold text-slate-900">
              {loading ? '—' : stats.total_cases.toLocaleString()}
            </div>
            <p className="text-xs text-slate-500 mt-1 flex items-center space-x-1">
              <span>Repository cumulative count</span>
            </p>
          </div>
        </div>

        {/* Cases This Week */}
        <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-xs hover:border-slate-300 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Cases This Week
            </span>
            <div className="w-10 h-10 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center">
              <Calendar className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-3xl font-extrabold text-indigo-900">
              {loading ? '—' : stats.cases_this_week.toLocaleString()}
            </div>
            <p className="text-xs text-slate-500 mt-1 flex items-center space-x-1">
              <span>Past 7 days intake</span>
            </p>
          </div>
        </div>

        {/* High Risk Cases */}
        <div className="bg-white rounded-2xl p-5 border border-rose-200/80 shadow-xs hover:border-rose-300 transition-all bg-gradient-to-br from-white to-rose-50/30">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-rose-700">
              High Risk Cases
            </span>
            <div className="w-10 h-10 rounded-xl bg-rose-100 text-rose-600 flex items-center justify-center">
              <ShieldAlert className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-3xl font-extrabold text-rose-700">
              {loading ? '—' : stats.high_risk_count.toLocaleString()}
            </div>
            <p className="text-xs text-rose-600 font-medium mt-1">
              Urgent review recommendation
            </p>
          </div>
        </div>

        {/* Average Confidence */}
        <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-xs hover:border-slate-300 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Avg Confidence
            </span>
            <div className="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center">
              <Percent className="w-5 h-5" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-3xl font-extrabold text-slate-900">
              {loading ? '—' : `${(stats.avg_confidence * 100).toFixed(1)}%`}
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Across all evaluated images
            </p>
          </div>
        </div>
      </div>

      {/* Analytics Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Bar Chart: Class Distribution (7 classes) */}
        <div className="bg-white rounded-2xl p-6 border border-slate-200/80 shadow-xs">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-base font-bold text-slate-900 flex items-center space-x-2">
                <Activity className="w-4 h-4 text-sky-600" />
                <span>Condition Distribution (7 Classes)</span>
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Frequency across all AI model predictions
              </p>
            </div>
          </div>

          <div className="h-72 w-full">
            {loading ? (
              <div className="h-full flex items-center justify-center">
                <div className="w-8 h-8 border-3 border-sky-200 border-t-sky-600 rounded-full animate-spin"></div>
              </div>
            ) : classDist.length === 0 ? (
              <div className="h-full flex items-center justify-center text-xs text-slate-400">
                No predictions recorded yet.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={classDist}
                  margin={{ top: 15, right: 10, left: -20, bottom: 25 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                  <XAxis
                    dataKey="class_code"
                    tick={{ fill: '#64748b', fontSize: 11 }}
                    tickLine={false}
                    axisLine={{ stroke: '#e2e8f0' }}
                  />
                  <YAxis
                    allowDecimals={false}
                    tick={{ fill: '#64748b', fontSize: 11 }}
                    tickLine={false}
                    axisLine={{ stroke: '#e2e8f0' }}
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const item = payload[0].payload;
                        return (
                          <div className="bg-slate-900 text-white text-xs rounded-lg px-3 py-2 shadow-lg">
                            <p className="font-bold">{item.class_name}</p>
                            <p className="text-sky-300 mt-0.5">
                              {item.count} cases ({item.percentage}%)
                            </p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar
                    dataKey="count"
                    fill="#0284c7"
                    radius={[6, 6, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* Quick legend for class codes */}
          <div className="mt-3 pt-3 border-t border-slate-100 flex flex-wrap gap-2 text-[10px] text-slate-500">
            {classDist.map((item) => (
              <span key={item.class_code || item.class_name} className="bg-slate-50 px-2 py-0.5 rounded border border-slate-200">
                <span className="font-bold text-slate-700">{item.class_code?.toUpperCase()}:</span> {item.class_name}
              </span>
            ))}
          </div>
        </div>

        {/* Line Chart: Cases Over Time (last 30 days) */}
        <div className="bg-white rounded-2xl p-6 border border-slate-200/80 shadow-xs">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-base font-bold text-slate-900 flex items-center space-x-2">
                <TrendingUp className="w-4 h-4 text-indigo-600" />
                <span>Screening Volume (Last 30 Days)</span>
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Daily submission velocity across clinical workflows
              </p>
            </div>
          </div>

          <div className="h-72 w-full">
            {loading ? (
              <div className="h-full flex items-center justify-center">
                <div className="w-8 h-8 border-3 border-indigo-200 border-t-indigo-600 rounded-full animate-spin"></div>
              </div>
            ) : casesOverTime.length === 0 ? (
              <div className="h-full flex items-center justify-center text-xs text-slate-400">
                No time-series data available.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={casesOverTime}
                  margin={{ top: 15, right: 15, left: -20, bottom: 25 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                  <XAxis
                    dataKey="date"
                    tickFormatter={formatChartDate}
                    tick={{ fill: '#64748b', fontSize: 10 }}
                    tickLine={false}
                    axisLine={{ stroke: '#e2e8f0' }}
                    interval={4}
                  />
                  <YAxis
                    allowDecimals={false}
                    tick={{ fill: '#64748b', fontSize: 11 }}
                    tickLine={false}
                    axisLine={{ stroke: '#e2e8f0' }}
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const item = payload[0].payload;
                        return (
                          <div className="bg-slate-900 text-white text-xs rounded-lg px-3 py-2 shadow-lg">
                            <p className="font-semibold text-slate-300">{item.date}</p>
                            <p className="text-indigo-300 font-bold mt-0.5">
                              {item.count} {item.count === 1 ? 'case' : 'cases'}
                            </p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="count"
                    stroke="#4f46e5"
                    strokeWidth={2.5}
                    dot={{ r: 2.5, fill: '#4f46e5' }}
                    activeDot={{ r: 5, stroke: '#818cf8', strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>Aggregated by submission timestamp (UTC)</span>
            <span className="font-medium text-slate-700">30-day timeline</span>
          </div>
        </div>
      </div>

      {/* Recent Cases Table */}
      <div className="bg-white rounded-2xl border border-slate-200/80 shadow-xs overflow-hidden">
        <div className="px-6 py-5 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-bold text-slate-900">Recent Cases for Clinical Triage</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Last 20 cases with latest automated risk and condition classifications
            </p>
          </div>
          <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-slate-100 text-slate-600 self-start sm:self-auto">
            {recentCases.length} Recent Cases
          </span>
        </div>

        {loading ? (
          <div className="py-16 text-center">
            <div className="w-8 h-8 border-3 border-sky-200 border-t-sky-600 rounded-full animate-spin mx-auto mb-3"></div>
            <p className="text-sm text-slate-500">Loading cases for triage...</p>
          </div>
        ) : recentCases.length === 0 ? (
          <div className="py-12 text-center text-sm text-slate-500">
            No recent cases registered in the system.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50/80 border-b border-slate-200/80 text-[11px] font-bold uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="py-3.5 px-6">Case ID</th>
                  <th className="py-3.5 px-6">Patient</th>
                  <th className="py-3.5 px-6">Predicted Condition</th>
                  <th className="py-3.5 px-6">Confidence</th>
                  <th className="py-3.5 px-6">Risk Stratification</th>
                  <th className="py-3.5 px-6">Created Date</th>
                  <th className="py-3.5 px-6 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {recentCases.map((rc) => {
                  const dateStr = rc.created_at
                    ? new Date(rc.created_at).toLocaleDateString(undefined, {
                        month: 'short',
                        day: 'numeric',
                        year: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                      })
                    : 'N/A';

                  return (
                    <tr
                      key={rc.case_id}
                      className="hover:bg-slate-50/80 transition-colors group"
                    >
                      <td className="py-4 px-6 font-bold text-slate-900">
                        #{rc.case_id}
                      </td>
                      <td className="py-4 px-6 text-slate-700 font-medium">
                        {rc.patient_name}
                      </td>
                      <td className="py-4 px-6">
                        <div className="font-semibold text-slate-900">
                          {rc.predicted_class_name || rc.predicted_class}
                        </div>
                        {rc.predicted_class_name && (
                          <div className="text-[11px] text-slate-400 font-mono">
                            {rc.predicted_class?.toUpperCase()}
                          </div>
                        )}
                      </td>
                      <td className="py-4 px-6">
                        <div className="flex items-center space-x-2">
                          <span className="font-semibold text-slate-800 text-xs">
                            {(rc.confidence * 100).toFixed(1)}%
                          </span>
                          <div className="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-sky-600 rounded-full"
                              style={{ width: `${Math.min(100, Math.max(0, rc.confidence * 100))}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td className="py-4 px-6">
                        {getRiskBadge(rc.risk_level)}
                      </td>
                      <td className="py-4 px-6 text-xs text-slate-500 whitespace-nowrap">
                        {dateStr}
                      </td>
                      <td className="py-4 px-6 text-right">
                        <Link
                          to={`/cases/${rc.case_id}`}
                          className="inline-flex items-center space-x-1 px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-sky-50 text-slate-700 hover:text-sky-700 text-xs font-semibold transition-colors"
                        >
                          <span>Review</span>
                          <ExternalLink className="w-3.5 h-3.5" />
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};