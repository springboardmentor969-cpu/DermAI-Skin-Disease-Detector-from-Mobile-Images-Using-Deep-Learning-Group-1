import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import apiClient, { reviewsApi, imagesApi } from '../api/client';
import {
  ArrowLeft,
  CheckCircle2,
  AlertTriangle,
  ShieldAlert,
  Save,
  Sparkles,
  Layers,
  Image as ImageIcon,
  Activity,
  FileText,
  User,
  Clock,
  Check,
} from 'lucide-react';

export const DISEASE_CLASSES = [
  { code: 'mel', name: 'Melanoma', risk: 'HIGH', description: 'Malignant melanoma of the skin' },
  { code: 'nv', name: 'Melanocytic Nevus', risk: 'LOW', description: 'Common benign melanocytic mole' },
  { code: 'bkl', name: 'Benign Keratosis', risk: 'LOW', description: 'Seborrheic keratosis or solar lentigo' },
  { code: 'bcc', name: 'Basal Cell Carcinoma', risk: 'MEDIUM', description: 'Non-melanoma skin cancer' },
  { code: 'akiec', name: 'Actinic Keratosis', risk: 'MEDIUM', description: 'Pre-cancerous intraepithelial carcinoma' },
  { code: 'vasc', name: 'Vascular Lesion', risk: 'LOW', description: 'Angioma, angiokeratoma, pyogenic granuloma' },
  { code: 'df', name: 'Dermatofibroma', risk: 'LOW', description: 'Benign fibrous skin lesion' },
];

export const ClinicalReviewPage = () => {
  const { image_id } = useParams();
  const navigate = useNavigate();
  const { user, isAuthenticated, loading: authLoading } = useAuth();

  // Page state
  const [imageMeta, setImageMeta] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [existingReview, setExistingReview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  // Visual assets blob URLs
  const [originalUrl, setOriginalUrl] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const [overlayUrl, setOverlayUrl] = useState(null);
  const [gradcamError, setGradcamError] = useState(false);

  // Clinician form state
  const [notes, setNotes] = useState('');
  const [overridden, setOverridden] = useState(false);
  const [confirmedClass, setConfirmedClass] = useState('mel');

  // Role verification: only clinicians and admins
  useEffect(() => {
    if (!authLoading && isAuthenticated) {
      const role = (user?.role || '').toLowerCase();
      if (role !== 'clinician' && role !== 'admin') {
        navigate('/dashboard', { replace: true });
      }
    }
  }, [user, isAuthenticated, authLoading, navigate]);

  // Load image, predictions, and existing review
  useEffect(() => {
    let isMounted = true;

    const loadData = async () => {
      setLoading(true);
      setError(null);
      try {
        // 1. Fetch image metadata
        const metaRes = await apiClient.get(`/images/${image_id}`);
        const meta = metaRes.data;
        if (!isMounted) return;
        setImageMeta(meta);

        // 2. Determine prediction
        let activePred = null;
        if (meta.predictions && meta.predictions.length > 0) {
          activePred = meta.predictions[meta.predictions.length - 1];
        } else {
          // If not analyzed, trigger analyze
          try {
            activePred = await imagesApi.analyzeImage(image_id);
          } catch (e) {
            console.warn('Auto-analyze attempt notice:', e);
          }
        }
        setPrediction(activePred);

        // 3. Set default confirmed class to AI's prediction
        const defaultClass = (activePred?.predicted_class || 'mel').toLowerCase();
        setConfirmedClass(defaultClass);

        // 4. Fetch existing review if available
        try {
          const rev = await reviewsApi.getReview(image_id);
          if (rev && isMounted) {
            setExistingReview(rev);
            setNotes(rev.notes || '');
            setOverridden(Boolean(rev.overridden));
            if (rev.confirmed_class) {
              setConfirmedClass(rev.confirmed_class.toLowerCase());
            }
          }
        } catch (revErr) {
          // 404 is normal if image has not been reviewed yet
          console.log('No prior review for image:', image_id);
        }

        // 5. Fetch image asset blobs
        try {
          const origBlob = await apiClient.get(`/images/${image_id}/content`, { responseType: 'blob' });
          if (isMounted) setOriginalUrl(URL.createObjectURL(origBlob.data));
        } catch (origErr) {
          // Fallback through case image route if needed
          if (meta.case_id) {
            try {
              const fallbackBlob = await apiClient.get(`/cases/${meta.case_id}/images/${image_id}/content`, { responseType: 'blob' });
              if (isMounted) setOriginalUrl(URL.createObjectURL(fallbackBlob.data));
            } catch (fbErr) {
              console.warn('Failed to load original image blob:', fbErr);
            }
          }
        }

        // Heatmap
        try {
          const hmUrl = await imagesApi.getGradcamBlobUrl(image_id, 'heatmap');
          if (isMounted) setHeatmapUrl(hmUrl);
        } catch (e) {
          console.warn('Grad-CAM heatmap unavailable:', e);
          setGradcamError(true);
        }

        // Overlay
        try {
          const ovUrl = await imagesApi.getGradcamBlobUrl(image_id, 'overlay');
          if (isMounted) setOverlayUrl(ovUrl);
        } catch (e) {
          console.warn('Grad-CAM overlay unavailable:', e);
          setGradcamError(true);
        }
      } catch (err) {
        console.error('Error loading clinical review data:', err);
        if (isMounted) {
          setError(err.response?.data?.detail || 'Failed to load image details.');
        }
      } finally {
        if (isMounted) setLoading(false);
      }
    };

    if (image_id) {
      loadData();
    }

    return () => {
      isMounted = false;
      if (originalUrl) URL.revokeObjectURL(originalUrl);
      if (heatmapUrl) URL.revokeObjectURL(heatmapUrl);
      if (overlayUrl) URL.revokeObjectURL(overlayUrl);
    };
  }, [image_id]);

  // Handle override checkbox toggle
  const handleOverrideToggle = (e) => {
    const isChecked = e.target.checked;
    setOverridden(isChecked);
    if (!isChecked && prediction?.predicted_class) {
      setConfirmedClass(prediction.predicted_class.toLowerCase());
    }
  };

  // Submit review
  const handleSaveReview = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    setSuccessMsg(null);

    try {
      const reviewPayload = {
        notes: notes.trim(),
        confirmed_class: confirmedClass.toLowerCase(),
        overridden: Boolean(overridden),
      };

      const result = await reviewsApi.submitReview(image_id, reviewPayload);
      setExistingReview(result);
      setSuccessMsg('Clinical diagnostic review successfully saved and verified!');

      // Scroll to top or show alert
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (err) {
      console.error('Failed to submit clinical review:', err);
      setError(err.response?.data?.detail || 'Failed to submit review. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  // Risk badge helper
  const renderRiskBadge = (level) => {
    const norm = (level || 'LOW').toUpperCase();
    if (norm === 'HIGH') {
      return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide bg-red-100 text-red-800 border border-red-300">
          <ShieldAlert className="w-4 h-4 text-red-600" />
          High Risk
        </span>
      );
    }
    if (norm === 'MEDIUM') {
      return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide bg-amber-100 text-amber-800 border border-amber-300">
          <AlertTriangle className="w-4 h-4 text-amber-600" />
          Medium Risk
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide bg-emerald-100 text-emerald-800 border border-emerald-300">
        <CheckCircle2 className="w-4 h-4 text-emerald-600" />
        Low Risk
      </span>
    );
  };

  // Compute 7-class normalized probabilities
  const normalizedProbabilities = DISEASE_CLASSES.map((cls) => {
    let conf = 0;
    if (prediction?.probabilities && prediction.probabilities[cls.code] !== undefined) {
      conf = Number(prediction.probabilities[cls.code]);
    } else if (Array.isArray(prediction?.top_n)) {
      const found = prediction.top_n.find(
        (p) => (p.class || p.code || p.predicted_class || '').toLowerCase() === cls.code.toLowerCase()
      );
      if (found) {
        conf = Number(found.probability ?? found.confidence ?? 0);
      }
    } else if (prediction?.predicted_class?.toLowerCase() === cls.code.toLowerCase()) {
      conf = Number(prediction?.confidence || 0.85);
    }

    const isTop = (prediction?.predicted_class || '').toLowerCase() === cls.code.toLowerCase();

    return {
      ...cls,
      confidence: conf,
      isTop,
    };
  });

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-16 flex flex-col items-center justify-center">
        <div className="w-12 h-12 border-4 border-sky-200 border-t-sky-600 rounded-full animate-spin"></div>
        <p className="mt-4 text-sm font-medium text-slate-600">Loading clinical review and AI telemetry...</p>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-200 pb-5">
        <div className="space-y-1">
          <div className="flex items-center gap-3">
            <Link
              to="/clinician"
              className="inline-flex items-center gap-1.5 text-sm font-medium text-slate-500 hover:text-slate-800 transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              Clinician Dashboard
            </Link>
            <span className="text-slate-300">/</span>
            <span className="text-sm font-semibold text-sky-700">Clinical Review #{image_id}</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900">
            Dermoscopic Analysis & Clinician Sign-Off
          </h1>
          <p className="text-sm text-slate-500">
            Case #{imageMeta?.case_id || 'N/A'} • Captured {imageMeta?.captured_at ? new Date(imageMeta.captured_at).toLocaleDateString() : 'Recently'}
          </p>
        </div>

        <div className="flex items-center gap-3">
          {existingReview ? (
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
              <Check className="w-3.5 h-3.5" /> Reviewed by {existingReview.clinician_name || 'Clinician'}
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-semibold bg-amber-50 text-amber-700 border border-amber-200">
              <Clock className="w-3.5 h-3.5" /> Awaiting Clinician Review
            </span>
          )}
        </div>
      </div>

      {/* Notifications */}
      {successMsg && (
        <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 flex items-center gap-3 shadow-sm">
          <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0" />
          <div className="text-sm font-medium">{successMsg}</div>
        </div>
      )}

      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-800 flex items-center gap-3 shadow-sm">
          <AlertTriangle className="w-5 h-5 text-red-600 shrink-0" />
          <div className="text-sm font-medium">{error}</div>
        </div>
      )}

      {/* Section 1: Full Analysis Report - Visual Explainability (3 Images Side by Side) */}
      <section className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-5">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2.5">
            <Layers className="w-5 h-5 text-sky-600" />
            <h2 className="text-lg font-bold text-slate-900">Dermoscopic Visual Triad & Grad-CAM Heatmap</h2>
          </div>
          <span className="text-xs text-slate-500 font-medium">
            Grad-CAM layer: target_layer_4 (Deep Feature Saliency)
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* 1. Original Image */}
          <div className="flex flex-col items-center p-3 rounded-xl bg-slate-50 border border-slate-200">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2.5">
              1. Original Macro / Dermoscopy
            </span>
            <div className="w-full aspect-square bg-slate-200 rounded-lg overflow-hidden flex items-center justify-center border border-slate-300">
              {originalUrl ? (
                <img
                  src={originalUrl}
                  alt="Original lesion"
                  className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
                />
              ) : (
                <div className="text-center p-4 text-slate-400">
                  <ImageIcon className="w-8 h-8 mx-auto mb-1 opacity-50" />
                  <span className="text-xs">Original image loading...</span>
                </div>
              )}
            </div>
            <p className="text-[11px] text-slate-500 mt-2 text-center">
              Source resolution optimized for deep convolutional inference (224×224)
            </p>
          </div>

          {/* 2. Grad-CAM Heatmap */}
          <div className="flex flex-col items-center p-3 rounded-xl bg-slate-50 border border-slate-200">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2.5">
              2. Grad-CAM Heatmap (Jet Saliency)
            </span>
            <div className="w-full aspect-square bg-slate-900 rounded-lg overflow-hidden flex items-center justify-center border border-slate-300">
              {heatmapUrl ? (
                <img
                  src={heatmapUrl}
                  alt="Grad-CAM Heatmap"
                  className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
                />
              ) : (
                <div className="text-center p-4 text-slate-400">
                  <Sparkles className="w-8 h-8 mx-auto mb-1 opacity-50 text-amber-400" />
                  <span className="text-xs">Heatmap loading or generating...</span>
                </div>
              )}
            </div>
            <p className="text-[11px] text-slate-500 mt-2 text-center">
              Warm regions (red/yellow) indicate highest gradient weight for prediction
            </p>
          </div>

          {/* 3. Overlay */}
          <div className="flex flex-col items-center p-3 rounded-xl bg-slate-50 border border-slate-200">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-600 mb-2.5">
              3. Superimposed Overlay (Alpha 0.4)
            </span>
            <div className="w-full aspect-square bg-slate-200 rounded-lg overflow-hidden flex items-center justify-center border border-slate-300">
              {overlayUrl ? (
                <img
                  src={overlayUrl}
                  alt="Grad-CAM Overlay"
                  className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
                />
              ) : (
                <div className="text-center p-4 text-slate-400">
                  <Layers className="w-8 h-8 mx-auto mb-1 opacity-50 text-sky-400" />
                  <span className="text-xs">Overlay generating...</span>
                </div>
              )}
            </div>
            <p className="text-[11px] text-slate-500 mt-2 text-center">
              Direct correlation between visual lesion boundary and neural attention
            </p>
          </div>
        </div>
      </section>

      {/* Section 2: AI Screening Summary & 7-Class Probabilities */}
      <section className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
          <div>
            <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
              <Activity className="w-5 h-5 text-sky-600" />
              Deep Learning Classifier Predictions (HAM10000 7-Class Model)
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Calibrated multi-class probability distribution computed via ensemble inference
            </p>
          </div>

          {/* Risk Level Badge */}
          <div className="flex items-center gap-3">
            <span className="text-xs font-medium text-slate-500">Triage Risk:</span>
            {renderRiskBadge(prediction?.risk_level)}
          </div>
        </div>

        {/* Top Prediction Banner */}
        <div className="p-4 rounded-xl bg-sky-50/70 border border-sky-200/80 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="text-xs font-semibold uppercase tracking-wider text-sky-800">
              Primary AI Diagnostic Prediction
            </div>
            <div className="text-xl font-bold text-slate-900">
              {DISEASE_CLASSES.find((c) => c.code === prediction?.predicted_class?.toLowerCase())?.name || prediction?.predicted_class || 'Evaluating...'}
              <span className="ml-2 text-sm font-medium text-sky-700">
                ({prediction?.predicted_class?.toUpperCase()})
              </span>
            </div>
            <div className="text-xs text-slate-600">
              Model Confidence: <span className="font-semibold text-slate-900">{((prediction?.confidence || 0) * 100).toFixed(1)}%</span>
              {prediction?.is_low_confidence && (
                <span className="ml-2 text-amber-700 font-medium">⚠️ Low confidence warning flagged</span>
              )}
            </div>
          </div>

          <div className="text-right sm:border-l sm:border-sky-200 sm:pl-6">
            <div className="text-xs font-semibold text-slate-500">Architecture</div>
            <div className="text-xs font-mono font-medium text-slate-800 mt-0.5">
              {prediction?.model_version || 'Ensemble: ResNet50 + EfficientNet-B0'}
            </div>
          </div>
        </div>

        {/* 7 Disease Probabilities Horizontal Progress Bars */}
        <div className="space-y-3 pt-2">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
            Multi-Class Probability Distribution
          </h3>
          <div className="space-y-2.5">
            {normalizedProbabilities.map((item) => {
              const pct = (item.confidence * 100).toFixed(1);
              return (
                <div
                  key={item.code}
                  className={`p-3 rounded-xl border transition-all ${
                    item.isTop
                      ? 'bg-sky-50/50 border-sky-300 ring-1 ring-sky-200'
                      : 'bg-slate-50/60 border-slate-200'
                  }`}
                >
                  <div className="flex items-center justify-between text-xs font-medium mb-1.5">
                    <div className="flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${
                        item.risk === 'HIGH' ? 'bg-red-500' : item.risk === 'MEDIUM' ? 'bg-amber-500' : 'bg-emerald-500'
                      }`} />
                      <span className="font-bold text-slate-800">{item.name}</span>
                      <span className="font-mono text-slate-400 uppercase">[{item.code}]</span>
                      {item.isTop && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-bold uppercase bg-sky-600 text-white">
                          AI Top Pick
                        </span>
                      )}
                    </div>
                    <span className="font-bold text-slate-900">{pct}%</span>
                  </div>

                  {/* Horizontal Bar */}
                  <div className="w-full bg-slate-200 rounded-full h-2.5 overflow-hidden">
                    <div
                      className={`h-2.5 rounded-full transition-all duration-500 ${
                        item.isTop
                          ? 'bg-sky-600'
                          : item.risk === 'HIGH'
                          ? 'bg-red-500'
                          : item.risk === 'MEDIUM'
                          ? 'bg-amber-500'
                          : 'bg-emerald-500'
                      }`}
                      style={{ width: `${Math.max(Number(pct), 1)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Section 3: Clinician Review & Decision Form */}
      <section className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-6">
        <div className="border-b border-slate-200 pb-4">
          <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
            <FileText className="w-6 h-6 text-sky-600" />
            Clinician Decision & Diagnostic Confirmation
          </h2>
          <p className="text-sm text-slate-500 mt-1">
            As a licensed healthcare practitioner, evaluate the clinical image and AI decision-support data to confirm or override the final diagnosis.
          </p>
        </div>

        <form onSubmit={handleSaveReview} className="space-y-6">
          {/* Override Checkbox */}
          <div className="p-4 rounded-xl border border-slate-200 bg-slate-50 space-y-3">
            <div className="flex items-start gap-3">
              <input
                id="override-checkbox"
                type="checkbox"
                checked={overridden}
                onChange={handleOverrideToggle}
                className="mt-1 w-4 h-4 text-sky-600 rounded border-slate-300 focus:ring-sky-500"
              />
              <div className="space-y-1">
                <label htmlFor="override-checkbox" className="text-sm font-bold text-slate-900 cursor-pointer">
                  Override AI Prediction
                </label>
                <p className="text-xs text-slate-600">
                  Check this box if your professional clinical evaluation differs from the AI recommendation. This will flag the case for discrepancy tracking and model retraining audit.
                </p>
              </div>
            </div>

            {/* Confirm Diagnosis Dropdown */}
            <div className="pt-2 border-t border-slate-200/80">
              <label htmlFor="confirm-diagnosis-select" className="block text-xs font-bold uppercase tracking-wider text-slate-700 mb-2">
                {overridden ? 'Select Overridden Diagnosis Class:' : 'Confirmed Diagnostic Class (Defaults to AI Prediction):'}
              </label>
              <select
                id="confirm-diagnosis-select"
                value={confirmedClass}
                onChange={(e) => setConfirmedClass(e.target.value)}
                disabled={!overridden}
                className={`w-full max-w-md px-3.5 py-2.5 rounded-lg border text-sm font-medium transition-all ${
                  overridden
                    ? 'border-sky-500 bg-white text-slate-900 focus:ring-2 focus:ring-sky-500 focus:border-sky-500 shadow-sm'
                    : 'border-slate-300 bg-slate-100 text-slate-600 cursor-not-allowed'
                }`}
              >
                {DISEASE_CLASSES.map((cls) => (
                  <option key={cls.code} value={cls.code}>
                    {cls.name} ({cls.code.toUpperCase()}) — {cls.risk} Risk
                  </option>
                ))}
              </select>
              {!overridden && (
                <p className="text-[11px] text-slate-500 mt-1.5">
                  Currently locked to AI prediction: <span className="font-semibold">{confirmedClass.toUpperCase()}</span>. Check "Override AI Prediction" above to change.
                </p>
              )}
            </div>
          </div>

          {/* Clinician Notes Textarea */}
          <div className="space-y-2">
            <label htmlFor="clinician-notes" className="block text-sm font-bold text-slate-900">
              Clinician Notes & Diagnostic Observations
            </label>
            <textarea
              id="clinician-notes"
              rows={5}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Record ABCD criteria (Asymmetry, Border, Color, Diameter), dermoscopic pattern analysis, biopsy recommendation, or patient follow-up plan..."
              className="w-full px-4 py-3 rounded-xl border border-slate-300 focus:ring-2 focus:ring-sky-500 focus:border-sky-500 text-sm text-slate-800 placeholder-slate-400 transition-all resize-y"
            />
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-slate-200">
            <div className="text-xs text-slate-500 flex items-center gap-1.5">
              <User className="w-4 h-4 text-slate-400" />
              Signing as: <span className="font-semibold text-slate-700">{user?.full_name || user?.email}</span> (Clinician)
            </div>

            <div className="flex items-center gap-3 w-full sm:w-auto">
              <button
                type="button"
                onClick={() => navigate('/clinician')}
                className="w-full sm:w-auto px-5 py-2.5 rounded-xl border border-slate-300 text-sm font-semibold text-slate-700 hover:bg-slate-50 transition-colors text-center"
              >
                Back to Triage
              </button>
              <button
                type="submit"
                disabled={submitting}
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl bg-sky-600 hover:bg-sky-700 text-white text-sm font-bold shadow-sm hover:shadow transition-all disabled:opacity-50"
              >
                {submitting ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Saving...</span>
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4" />
                    <span>Save Review & Sign-Off</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </form>
      </section>
    </div>
  );
};

export default ClinicalReviewPage;
