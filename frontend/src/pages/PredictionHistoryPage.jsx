import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Clock,
  Calendar,
  AlertTriangle,
  ChevronRight,
  Sparkles,
  ArrowRight,
  Layers,
  Activity,
  PlusCircle,
  RefreshCw,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { casesApi } from '../api/client';

export const DISEASE_NAMES = {
  akiec: 'Actinic Keratosis',
  bcc: 'Basal Cell Carcinoma',
  bkl: 'Benign Keratosis',
  df: 'Dermatofibroma',
  mel: 'Melanoma',
  nv: 'Melanocytic Nevus',
  vasc: 'Vascular Lesion',
};

export const getConfidenceBarColor = (confidencePct) => {
  if (confidencePct >= 70) return 'bg-emerald-500';
  if (confidencePct >= 40) return 'bg-amber-500';
  return 'bg-rose-500';
};

export const computeRiskLevel = (predictedClass, confidence) => {
  const cls = (predictedClass || '').toLowerCase().trim();
  const conf = Number(confidence || 0);

  // mel (Melanoma) → always HIGH RISK (red badge)
  if (cls === 'mel') {
    return 'HIGH';
  }

  // anything with confidence < 45% (0.45) → MEDIUM RISK
  if (conf < 0.45) {
    return 'MEDIUM';
  }

  // bcc, akiec → HIGH RISK if confidence >= 50% (0.50)
  if ((cls === 'bcc' || cls === 'akiec') && conf >= 0.50) {
    return 'HIGH';
  }

  // nv, df, vasc → LOW RISK if confidence >= 50% (0.50)
  if ((cls === 'nv' || cls === 'df' || cls === 'vasc') && conf >= 0.50) {
    return 'LOW';
  }

  // bkl at confidence >= 45%
  if (cls === 'bkl' && conf >= 0.45) {
    return 'LOW';
  }

  return 'MEDIUM';
};

export const extractTopPrediction = (entry) => {
  if (!entry) {
    return {
      classCode: 'nv',
      diseaseName: 'Melanocytic Nevus',
      confidence: 0,
      confidencePct: 0,
    };
  }

  const rawPredictions = entry.predictions || entry.top_n || entry.all_predictions;

  if (Array.isArray(rawPredictions) && rawPredictions.length > 0) {
    // Array of numbers e.g. [0.12, 0.05, ...]
    if (typeof rawPredictions[0] === 'number') {
      const CLASS_ORDER = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'];
      let maxProb = -1;
      let maxIdx = 0;
      rawPredictions.forEach((prob, idx) => {
        const val = Number(prob);
        if (val > maxProb) {
          maxProb = val;
          maxIdx = idx;
        }
      });
      const topCode = (CLASS_ORDER[maxIdx] || 'nv').toLowerCase();
      const confPct = Math.round(maxProb * 100);
      return {
        classCode: topCode,
        diseaseName: DISEASE_NAMES[topCode] || topCode,
        confidence: maxProb,
        confidencePct: confPct,
      };
    }

    // Array of prediction objects
    let topItem = null;
    let maxProb = -1;

    for (const item of rawPredictions) {
      if (!item) continue;
      const prob = Number(
        item.probability ??
        item.confidence ??
        item.score ??
        item.prob ??
        item.value ??
        0
      );
      if (prob > maxProb) {
        maxProb = prob;
        topItem = item;
      }
    }

    if (topItem && maxProb >= 0) {
      const code = (
        topItem.class ||
        topItem.predicted_class ||
        topItem.code ||
        topItem.disease ||
        topItem.label ||
        topItem.name ||
        entry.predicted_class ||
        'nv'
      ).toLowerCase().trim();

      const confPct = Math.round(maxProb * 100);
      return {
        classCode: code,
        diseaseName: DISEASE_NAMES[code] || entry.predicted_class_name || code,
        confidence: maxProb,
        confidencePct: confPct,
      };
    }
  }

  // Object mapping e.g. { mel: 0.68, nv: 0.16 }
  if (rawPredictions && typeof rawPredictions === 'object' && !Array.isArray(rawPredictions)) {
    let topCode = '';
    let maxProb = -1;
    for (const [key, val] of Object.entries(rawPredictions)) {
      const p = Number(val);
      if (p > maxProb) {
        maxProb = p;
        topCode = key;
      }
    }
    if (topCode && maxProb >= 0) {
      const code = topCode.toLowerCase().trim();
      const confPct = Math.round(maxProb * 100);
      return {
        classCode: code,
        diseaseName: DISEASE_NAMES[code] || code,
        confidence: maxProb,
        confidencePct: confPct,
      };
    }
  }

  // Fallback to entry.predicted_class and entry.confidence
  const fallbackCode = (entry.predicted_class || 'nv').toLowerCase().trim();
  const fallbackConf = Number(entry.confidence || 0);
  const confPct = Math.round(fallbackConf * 100);

  return {
    classCode: fallbackCode,
    diseaseName: DISEASE_NAMES[fallbackCode] || entry.predicted_class_name || fallbackCode,
    confidence: fallbackConf,
    confidencePct: confPct,
  };
};

// Module-level in-memory cache to preserve loaded history across page navigation
let cachedHistoryState = {
  items: null,
  page: 1,
  hasMore: true,
};

export const PredictionHistoryPage = () => {
  const { user } = useAuth();
  const navigate = useNavigate();

  const [history, setHistory] = useState(() => cachedHistoryState.items || []);
  const [loading, setLoading] = useState(() => cachedHistoryState.items === null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState('');
  const [page, setPage] = useState(() => cachedHistoryState.page || 1);
  const [hasMore, setHasMore] = useState(() => cachedHistoryState.hasMore ?? true);

  const observerTarget = React.useRef(null);
  const limit = 10;

  useEffect(() => {
    document.title = 'My History | DermAI';
    if (cachedHistoryState.items === null) {
      loadInitial();
    }
  }, []);

  const loadInitial = async () => {
    setLoading(true);
    setError('');
    try {
      const data = await casesApi.getPatientHistory('me', 1, limit);
      // Support both empty array [] and envelope { items: [], total: 0 }
      const items = Array.isArray(data) ? data : (data?.items || []);
      const more = items.length === limit;
      setHistory(items);
      setPage(1);
      setHasMore(more);
      cachedHistoryState = {
        items,
        page: 1,
        hasMore: more,
      };
    } catch (err) {
      console.error('Failed to load prediction history:', err);
      // If 404 (e.g. patient has no profile/records yet), treat as empty history without error banner
      if (err.response?.status === 404) {
        setHistory([]);
        setHasMore(false);
        cachedHistoryState = { items: [], page: 1, hasMore: false };
      } else {
        const msg = !err.response
          ? 'Failed to load screening history. Please check your connection.'
          : err.response?.data?.detail || 'Failed to load screening history. Please try again.';
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  const loadMore = async () => {
    if (loading || loadingMore || !hasMore) return;
    setLoadingMore(true);
    const nextPage = page + 1;
    try {
      const data = await casesApi.getPatientHistory('me', nextPage, limit);
      const newItems = Array.isArray(data) ? data : (data?.items || []);
      const more = newItems.length === limit;

      setHistory((prev) => {
        const existingIds = new Set(prev.map((i) => i.id));
        const filtered = newItems.filter((i) => !existingIds.has(i.id));
        const combined = [...prev, ...filtered];
        cachedHistoryState.items = combined;
        return combined;
      });

      setPage(nextPage);
      setHasMore(more);
      cachedHistoryState.page = nextPage;
      cachedHistoryState.hasMore = more;
    } catch (err) {
      console.error('Failed to load more prediction history:', err);
    } finally {
      setLoadingMore(false);
    }
  };

  // IntersectionObserver for infinite scrolling
  useEffect(() => {
    const target = observerTarget.current;
    if (!target) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore && !loading && !loadingMore) {
          loadMore();
        }
      },
      { threshold: 0.1, rootMargin: '120px' }
    );

    observer.observe(target);
    return () => observer.disconnect();
  }, [hasMore, loading, loadingMore, page]);

  const handleRefresh = () => {
    cachedHistoryState = { items: null, page: 1, hasMore: true };
    loadInitial();
  };


  const formatDate = (dateString) => {
    if (!dateString) return 'Date unknown';
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getRiskBadge = (riskLevel) => {
    const lvl = (riskLevel || '').toUpperCase();
    if (lvl === 'HIGH') {
      return (
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-bold tracking-wide bg-rose-100 text-rose-800 border border-rose-300">
          HIGH RISK
        </span>
      );
    }
    if (lvl === 'MEDIUM') {
      return (
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-bold tracking-wide bg-amber-100 text-amber-800 border border-amber-300">
          MEDIUM RISK
        </span>
      );
    }
    return (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-bold tracking-wide bg-emerald-100 text-emerald-800 border border-emerald-300">
        LOW RISK
      </span>
    );
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
        <div>
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-lg bg-sky-100 text-sky-700 flex items-center justify-center">
              <Clock className="w-4 h-4" />
            </div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">My Screening History</h1>
          </div>
          <p className="text-sm text-slate-500 mt-1">
            Timeline of all past AI skin lesion analyses and risk assessments
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            type="button"
            onClick={handleRefresh}
            disabled={loading || loadingMore}
            className="inline-flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold transition-all cursor-pointer disabled:opacity-50"
            title="Refresh history"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-sky-600' : ''}`} />
            <span>Refresh</span>
          </button>
          <Link
            to="/cases/new"
            className="inline-flex items-center justify-center space-x-2 px-4 py-2.5 rounded-xl bg-sky-600 hover:bg-sky-700 text-white text-sm font-semibold shadow-xs hover:shadow-sky-500/20 transition-all active:scale-[0.99]"
          >
            <PlusCircle className="w-4 h-4" />
            <span>New Analysis</span>
          </Link>
        </div>
      </div>

      {/* Error State */}
      {error && (
        <div className="mb-6 p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-sm flex items-start space-x-3">
          <AlertTriangle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold block">Notice:</span>
            {error}
          </div>
        </div>
      )}

      {/* Loading Skeleton */}
      {loading ? (
        <div className="space-y-4 w-full">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="bg-white rounded-2xl border border-slate-200/90 p-4 sm:p-5 shadow-xs flex flex-col sm:flex-row items-start sm:items-center space-y-3 sm:space-y-0 sm:space-x-4 animate-pulse w-full"
            >
              <div className="w-16 sm:w-20 h-16 sm:h-20 rounded-xl bg-slate-200 shrink-0" />
              <div className="flex-1 space-y-2.5 w-full">
                <div className="w-1/3 sm:w-1/4 h-3.5 bg-slate-200 rounded" />
                <div className="w-2/3 sm:w-1/2 h-5 bg-slate-200 rounded" />
                <div className="w-1/2 sm:w-1/3 h-2.5 bg-slate-200 rounded" />
              </div>
            </div>
          ))}
        </div>
      ) : history.length === 0 ? (
        /* Empty State */
        <div className="rounded-2xl border-2 border-dashed border-slate-200 bg-white/70 p-12 text-center max-w-lg mx-auto my-8">
          <div className="w-14 h-14 mx-auto rounded-2xl bg-sky-50 text-sky-600 flex items-center justify-center mb-4 shadow-xs">
            <Sparkles className="w-7 h-7" />
          </div>
          <h3 className="text-base font-bold text-slate-800">No analyses yet. Upload an image to get started.</h3>
          <p className="text-xs text-slate-500 mt-1.5 max-w-sm mx-auto">
            Take or upload a photo of a skin lesion to receive real-time AI decision-support and risk scoring.
          </p>
          <Link
            to="/cases/new"
            className="inline-flex items-center space-x-2 mt-6 px-4 py-2.5 rounded-xl bg-sky-600 hover:bg-sky-700 text-white text-xs font-semibold shadow-xs transition-all"
          >
            <PlusCircle className="w-4 h-4" />
            <span>Upload Image</span>
          </Link>
        </div>
      ) : (
        /* Timeline List */
        <div className="relative border-l-2 border-slate-200 ml-4 sm:ml-6 pl-6 sm:pl-8 space-y-6">
          {history.map((entry) => {
            const top = extractTopPrediction(entry);
            const diseaseName = top.diseaseName;
            const classCode = top.classCode;
            const confPct = top.confidencePct;
            const riskLevel = computeRiskLevel(classCode, top.confidence);
            const barColorClass = getConfidenceBarColor(confPct);

            return (
              <div
                key={entry.id}
                onClick={() => navigate(`/cases/${entry.case_id}`)}
                className="relative group cursor-pointer bg-white rounded-2xl border border-slate-200 hover:border-sky-300 p-4 sm:p-5 shadow-xs hover:shadow-md transition-all duration-200 w-full"
              >
                {/* Timeline node */}
                <div className="absolute -left-[31px] sm:-left-[39px] top-6 w-3.5 h-3.5 rounded-full border-2 border-white bg-sky-600 shadow-xs ring-4 ring-slate-100 group-hover:scale-125 transition-transform" />

                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  {/* Left: Thumbnail & Details */}
                  <div className="flex items-start sm:items-center space-x-4 min-w-0">
                    {/* Thumbnail Image */}
                    <div className="w-20 h-20 sm:w-24 sm:h-24 rounded-xl overflow-hidden bg-slate-100 border border-slate-200 shrink-0">
                      <img
                        src={entry.thumbnail_url}
                        alt={`${diseaseName} scan`}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                        onError={(e) => {
                          // Fallback to internal API content endpoint if presigned URL fails
                          if (entry.image_url && e.target.src !== entry.image_url) {
                            e.target.src = entry.image_url;
                          } else {
                            e.target.style.display = 'none';
                          }
                        }}
                      />
                    </div>

                    {/* Metadata & Prediction */}
                    <div className="min-w-0 space-y-1.5">
                      <div className="flex items-center space-x-2 text-xs text-slate-500">
                        <Calendar className="w-3.5 h-3.5 text-slate-400" />
                        <span>{formatDate(entry.created_at)}</span>
                        <span>&bull;</span>
                        <span className="font-mono text-slate-600">Case #{entry.case_id}</span>
                      </div>

                      {/* Disease Name & Subtitle */}
                      <div>
                        <div className="text-base font-bold text-slate-900 group-hover:text-sky-700 transition-colors">
                          {diseaseName}
                        </div>
                        <div className="text-xs text-slate-500 font-medium mt-0.5">
                          {diseaseName} ({classCode})
                        </div>
                      </div>

                      {/* Confidence Progress Bar */}
                      <div className="pt-1 max-w-xs space-y-1">
                        <div className="flex justify-between items-center text-[11px]">
                          <span className="text-slate-500 font-medium">Confidence</span>
                          <span className="font-bold text-slate-800">{confPct}%</span>
                        </div>
                        <div className="w-48 bg-slate-100 rounded-full h-2 overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-300 ${barColorClass}`}
                            style={{ width: `${Math.min(100, Math.max(5, confPct))}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Right: Risk Badge & Arrow */}
                  <div className="flex sm:flex-col items-center sm:items-end justify-between sm:justify-center gap-2 pt-2 sm:pt-0 border-t sm:border-t-0 border-slate-100">
                    <div>{getRiskBadge(riskLevel)}</div>
                    <div className="flex items-center text-xs font-semibold text-sky-600 group-hover:text-sky-700">
                      <span>View Case</span>
                      <ChevronRight className="w-4 h-4 ml-0.5 group-hover:translate-x-1 transition-transform" />
                    </div>
                  </div>
                </div>
              </div>
            );
          })}

          {/* Loading more skeleton placeholders for infinite scroll */}
          {loadingMore && (
            <div className="space-y-4 pt-2">
              {[1, 2].map((i) => (
                <div
                  key={`more-skeleton-${i}`}
                  className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs flex items-center space-x-4 animate-pulse"
                >
                  <div className="w-20 h-20 rounded-xl bg-slate-200 shrink-0" />
                  <div className="flex-1 space-y-2.5">
                    <div className="w-1/4 h-3.5 bg-slate-200 rounded" />
                    <div className="w-1/2 h-5 bg-slate-200 rounded" />
                    <div className="w-1/3 h-2.5 bg-slate-200 rounded" />
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Infinite Scroll Intersection Observer Sentinel */}
          <div ref={observerTarget} className="h-4 w-full" />
        </div>
      )}

      {/* End of History Indicator */}
      {!loading && !loadingMore && history.length > 0 && !hasMore && (
        <div className="text-center pt-8 border-t border-slate-200 mt-8">
          <p className="text-xs font-medium text-slate-400">
            All screening history loaded ({history.length} cases)
          </p>
        </div>
      )}
    </div>
  );
};

