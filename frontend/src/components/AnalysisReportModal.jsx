import React, { useState, useEffect } from 'react';
import { jsPDF } from 'jspdf';
import {
  X,
  CheckCircle,
  FileText,
  Download,
  AlertTriangle,
  Sparkles,
  Calendar,
  User,
  ShieldAlert,
  Info,
} from 'lucide-react';
import { casesApi } from '../api/client';

export const DISEASE_MAP = {
  nv: 'Melanocytic Nevus',
  mel: 'Melanoma',
  bkl: 'Benign Keratosis-like Lesions',
  bcc: 'Basal Cell Carcinoma',
  akiec: 'Actinic Keratosis / Intraepithelial Carcinoma',
  df: 'Dermatofibroma',
  vasc: 'Vascular Lesions',
};

export const DIAGNOSTIC_NAMES = DISEASE_MAP;

const ALL_CLASSES = [
  { code: 'nv', name: 'Melanocytic Nevus' },
  { code: 'mel', name: 'Melanoma' },
  { code: 'bkl', name: 'Benign Keratosis-like Lesions' },
  { code: 'bcc', name: 'Basal Cell Carcinoma' },
  { code: 'akiec', name: 'Actinic Keratosis / Intraepithelial Carcinoma' },
  { code: 'df', name: 'Dermatofibroma' },
  { code: 'vasc', name: 'Vascular Lesions' },
];

export const AnalysisReportModal = ({
  isOpen,
  onClose,
  patientName = 'Patient',
  patientAge = 'N/A',
  caseId,
  imageId,
  imageUrl,
  predictions = [],
  probabilities = {},
  topPrediction,
  confidence,
  riskLevel = 'LOW',
  riskMessage,
  gradcamAvailable = true,
  analysisDate,
  disclaimer = 'DermAI is an educational & screening decision-support tool, NOT a medical diagnosis.',
}) => {
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const [overlayUrl, setOverlayUrl] = useState(null);
  const [gradcamLoading, setGradcamLoading] = useState(false);
  const [gradcamError, setGradcamError] = useState(false);

  // Normalize predictions across all 7 skin disease classes
  const normalized = ALL_CLASSES.map((cls) => {
    let conf = 0;
    if (probabilities && probabilities[cls.code] !== undefined) {
      conf = Number(probabilities[cls.code]);
    } else if (Array.isArray(predictions)) {
      const found = predictions.find(
        (p) =>
          (p.class || p.predicted_class || p.code || '').toLowerCase() === cls.code.toLowerCase() ||
          (p.disease || p.label || p.name || '').toLowerCase() === cls.name.toLowerCase()
      );
      if (found) {
        conf = Number(found.probability ?? found.confidence ?? 0);
      }
    }

    const topCode = (
      typeof topPrediction === 'object'
        ? topPrediction?.predicted_class || topPrediction?.class || topPrediction?.code
        : topPrediction
    )?.toLowerCase();

    const isTop = topCode
      ? topCode === cls.code || topCode === cls.name.toLowerCase()
      : false;

    return {
      code: cls.code,
      name: cls.name,
      confidence: conf,
      isTop,
    };
  });

  // Ensure top prediction is highlighted
  if (!normalized.some((item) => item.isTop)) {
    let highest = normalized[0];
    normalized.forEach((item) => {
      if (item.confidence > highest.confidence) highest = item;
    });
    if (highest) highest.isTop = true;
  }

  // Sort descending by confidence score
  normalized.sort((a, b) => b.confidence - a.confidence);

  const topItem = normalized.find((item) => item.isTop) || normalized[0];
  const topCodeFormatted = (topItem?.code || 'NEV').toUpperCase();
  const topConfidenceVal = topItem?.confidence ?? confidence ?? 0;
  const topLabel = `${topCodeFormatted} ${(topConfidenceVal * 100).toFixed(2)}%`;

  const finalRiskLevel = (riskLevel || (topItem.code === 'mel' ? 'HIGH' : 'LOW')).toUpperCase();

  const formattedDate = analysisDate
    ? new Date(analysisDate).toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    : new Date().toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });

  const exportDateStr = new Date().toISOString().split('T')[0];

  // Load Grad-CAM images when modal opens
  useEffect(() => {
    if (!isOpen || !imageId || !gradcamAvailable) {
      setHeatmapUrl(null);
      setOverlayUrl(null);
      setGradcamError(false);
      return;
    }

    let isMounted = true;
    setGradcamLoading(true);
    setGradcamError(false);

    Promise.all([
      casesApi.getGradcamBlobUrl(imageId, 'heatmap').catch(() => null),
      casesApi.getGradcamBlobUrl(imageId, 'overlay').catch(() => null),
    ])
      .then(([heatUrl, overUrl]) => {
        if (!isMounted) return;
        if (!heatUrl && !overUrl) {
          setGradcamError(true);
        } else {
          setHeatmapUrl(heatUrl);
          setOverlayUrl(overUrl || heatUrl);
        }
      })
      .catch(() => {
        if (isMounted) setGradcamError(true);
      })
      .finally(() => {
        if (isMounted) setGradcamLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [isOpen, imageId, gradcamAvailable]);

  // Handle ESC key to close modal
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  // JSON Export handler
  const handleExportJSON = () => {
    const payload = {
      caseId: caseId ?? null,
      patientName: patientName || 'Patient',
      date: exportDateStr,
      topPrediction: topItem.name,
      confidence: Number(topItem.confidence.toFixed(4)),
      riskLevel: finalRiskLevel,
      allPredictions: normalized.map((p) => ({
        disease: p.name,
        confidence: Number(p.confidence.toFixed(4)),
      })),
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `DermAI_Analysis_${caseId || 'case'}_${exportDateStr}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // CSV Export handler
  const handleExportCSV = () => {
    const headers = ['Disease', 'Confidence', 'IsTopPrediction', 'RiskLevel'];
    const rows = normalized.map((p) => [
      `"${p.name.replace(/"/g, '""')}"`,
      p.confidence.toFixed(4),
      p.isTop ? 'true' : 'false',
      finalRiskLevel,
    ]);

    const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `DermAI_Analysis_${caseId || 'case'}_${exportDateStr}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // PDF Export handler using jsPDF
  const handleExportPDF = () => {
    const doc = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4',
    });

    let y = 18;

    // Header Title & Subtitle
    doc.setFillColor(15, 23, 42); // slate-900
    doc.rect(0, 0, 210, 24, 'F');

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(14);
    doc.setTextColor(255, 255, 255);
    doc.text('Analysis Report', 15, 12);

    doc.setFontSize(8.5);
    doc.setTextColor(148, 163, 184); // slate-400
    doc.text('DERM AI REPORT  |  CONFIDENTIAL CLINICAL SCREENING DECISION SUPPORT', 15, 18);

    y = 32;

    // Status Pill
    doc.setFillColor(236, 253, 245);
    doc.setDrawColor(167, 243, 208);
    doc.roundedRect(15, y, 62, 7, 2, 2, 'FD');
    doc.setFontSize(8);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(5, 150, 105);
    doc.text('[✓] Analysis Completed', 19, y + 4.8);

    y += 13;

    // Patient Info Box
    doc.setFillColor(248, 250, 252);
    doc.setDrawColor(226, 232, 240);
    doc.roundedRect(15, y, 180, 22, 2, 2, 'FD');

    doc.setFontSize(8);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(100, 116, 139);
    doc.text('PATIENT NAME', 20, y + 7);
    doc.text('AGE', 80, y + 7);
    doc.text('CASE ID', 115, y + 7);
    doc.text('DATE OF ANALYSIS', 145, y + 7);

    doc.setFontSize(9.5);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(15, 23, 42);
    doc.text(String(patientName || 'Patient'), 20, y + 14);
    doc.text(String(patientAge || 'N/A'), 80, y + 14);
    doc.text(`#${caseId || 'N/A'}`, 115, y + 14);

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8.5);
    doc.text(String(formattedDate), 145, y + 14);

    y += 28;

    // Risk Stratification Banner
    const isHigh = finalRiskLevel === 'HIGH';
    const isMed = finalRiskLevel === 'MEDIUM';
    const riskBorder = isHigh ? [254, 205, 211] : isMed ? [253, 230, 138] : [167, 243, 208];
    const riskBg = isHigh ? [255, 241, 242] : isMed ? [254, 252, 232] : [240, 253, 244];
    const riskText = isHigh ? [159, 18, 57] : isMed ? [146, 64, 14] : [6, 95, 70];

    doc.setDrawColor(...riskBorder);
    doc.setFillColor(...riskBg);
    doc.roundedRect(15, y, 180, 14, 2, 2, 'FD');

    doc.setFontSize(8.5);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...riskText);
    doc.text(`CLINICAL RISK LEVEL: ${finalRiskLevel}`, 20, y + 5.5);

    doc.setFontSize(8);
    doc.setFont('helvetica', 'normal');
    doc.text(
      riskMessage ||
        (isHigh
          ? 'Urgent dermatologist review recommended. High clinical priority.'
          : isMed
          ? 'Clinical consultation recommended. Low model certainty or moderate-risk lesion.'
          : 'Routine monitoring recommended. Likely benign lesion.'),
      20,
      y + 10.5
    );

    y += 20;

    // Section 3: Disease Classification Results Table
    doc.setFontSize(10);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(15, 23, 42);
    doc.text('Disease Classification Results', 15, y);

    y += 4;
    doc.setFillColor(241, 245, 249);
    doc.setDrawColor(203, 213, 225);
    doc.rect(15, y, 180, 7, 'FD');

    doc.setFontSize(8);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(71, 85, 105);
    doc.text('SKIN DISEASE CLASS', 20, y + 5);
    doc.text('CODE', 125, y + 5);
    doc.text('CONFIDENCE', 160, y + 5);

    normalized.forEach((row, idx) => {
      y += 6.5;
      const rowBg = row.isTop ? [238, 242, 255] : idx % 2 === 0 ? [255, 255, 255] : [248, 250, 252];
      doc.setFillColor(...rowBg);
      doc.setDrawColor(226, 232, 240);
      doc.rect(15, y, 180, 6.5, 'FD');

      doc.setFont('helvetica', row.isTop ? 'bold' : 'normal');
      doc.setTextColor(row.isTop ? 2 : 51, row.isTop ? 132 : 65, row.isTop ? 199 : 85);

      const titleText = row.isTop ? `${row.name} (Top Prediction)` : row.name;
      doc.text(titleText, 20, y + 4.5);
      doc.text(row.code.toUpperCase(), 125, y + 4.5);
      doc.text(`${(row.confidence * 100).toFixed(2)}%`, 160, y + 4.5);
    });

    y += 14;

    // Visual Explainability Note in PDF
    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(15, 23, 42);
    doc.text('Visual Explanations (Grad-CAM):', 15, y);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8);
    doc.setTextColor(100, 116, 139);
    doc.text(
      gradcamAvailable && !gradcamError
        ? `Grad-CAM explainability maps generated for image #${imageId || 'N/A'}. Refer to interactive report for visual overlays.`
        : 'Visual explanation not available for demo model.',
      15,
      y + 5
    );

    y += 15;

    // Disclaimer Box
    doc.setDrawColor(254, 215, 170);
    doc.setFillColor(255, 247, 237);
    doc.roundedRect(15, y, 180, 12, 2, 2, 'FD');
    doc.setFontSize(7.5);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(154, 52, 18);
    doc.text(disclaimer, 105, y + 7, { align: 'center' });

    // Footer
    doc.setDrawColor(226, 232, 240);
    doc.line(15, 282, 195, 282);
    doc.setFontSize(7.5);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(148, 163, 184);
    doc.text('DermAI Milestone 3 Analysis Report  |  Automated Decision Support System', 105, 288, {
      align: 'center',
    });

    doc.save(`DermAI_Analysis_${caseId || 'case'}_${exportDateStr}.pdf`);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 md:p-6 bg-slate-900/60 backdrop-blur-xs overflow-y-auto animate-in fade-in duration-200"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="relative bg-white rounded-3xl shadow-2xl max-w-4xl w-full max-h-[92vh] overflow-y-auto border border-slate-200/80 my-auto flex flex-col">
        {/* Section 1 — Header */}
        <div className="sticky top-0 z-20 bg-white/95 backdrop-blur-md px-6 sm:px-8 py-5 border-b border-slate-100 flex items-center justify-between">
          <div>
            <div className="text-[10px] sm:text-xs font-bold uppercase tracking-widest text-sky-600">
              DERM AI REPORT
            </div>
            <div className="flex items-center space-x-3 mt-0.5">
              <h1 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight">
                Analysis Report
              </h1>
              <span className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 text-xs font-bold tracking-wide shadow-2xs">
                <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                <span>Analysis Completed</span>
              </span>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            aria-label="Close modal"
            className="w-9 h-9 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-500 hover:text-slate-800 flex items-center justify-center transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 sm:p-8 space-y-8 flex-1">
          {/* Section 2 — Patient Info */}
          <div className="bg-slate-50/80 rounded-2xl border border-slate-200/80 p-5 sm:p-6 shadow-2xs">
            <div className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-3 flex items-center space-x-1.5">
              <User className="w-4 h-4 text-slate-400" />
              <span>Patient &amp; Case Information</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div>
                <span className="text-[11px] font-semibold text-slate-400 block uppercase tracking-wide">
                  Patient Name
                </span>
                <span className="text-sm sm:text-base font-bold text-slate-900 truncate block mt-0.5">
                  {patientName || 'Patient'}
                </span>
              </div>
              <div>
                <span className="text-[11px] font-semibold text-slate-400 block uppercase tracking-wide">
                  Age
                </span>
                <span className="text-sm sm:text-base font-bold text-slate-900 block mt-0.5">
                  {patientAge ? `${patientAge} yrs` : 'N/A'}
                </span>
              </div>
              <div>
                <span className="text-[11px] font-semibold text-slate-400 block uppercase tracking-wide">
                  Case ID
                </span>
                <span className="text-sm sm:text-base font-bold text-sky-700 block mt-0.5">
                  #{caseId ?? 'N/A'}
                </span>
              </div>
              <div>
                <span className="text-[11px] font-semibold text-slate-400 block uppercase tracking-wide">
                  Date of Analysis
                </span>
                <span className="text-xs sm:text-sm font-semibold text-slate-700 block mt-0.5">
                  {formattedDate}
                </span>
              </div>
            </div>
          </div>

          {/* Section 3 — Disease Classification Results */}
          <div className="space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-slate-100">
              <div>
                <h2 className="text-base font-bold text-slate-900">
                  Disease Classification Results
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Full 7-class dermatological breakdown with model confidence scores
                </p>
              </div>

              {/* Top prediction risk badge indicator */}
              <div className="flex items-center space-x-2">
                <span className="text-xs font-semibold text-slate-500">Risk Assessment:</span>
                <span
                  className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-bold tracking-wide border shadow-2xs ${
                    finalRiskLevel === 'HIGH'
                      ? 'bg-rose-100 text-rose-800 border-rose-300'
                      : finalRiskLevel === 'MEDIUM'
                      ? 'bg-amber-100 text-amber-800 border-amber-300'
                      : 'bg-emerald-100 text-emerald-800 border-emerald-300'
                  }`}
                >
                  {finalRiskLevel} RISK
                </span>
              </div>
            </div>

            {/* List / Table of ALL 7 classes */}
            <div className="space-y-2.5">
              {normalized.map((item) => {
                const pct = Math.round(item.confidence * 100);
                const precisePct = (item.confidence * 100).toFixed(2);

                return (
                  <div
                    key={item.code}
                    className={`p-3.5 rounded-xl border transition-colors ${
                      item.isTop
                        ? 'bg-sky-50/70 border-sky-300 shadow-xs'
                        : 'bg-white border-slate-200/80 hover:bg-slate-50/60'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-3 mb-1.5">
                      <div className="flex items-center space-x-2.5 min-w-0">
                        <span
                          className={`text-xs uppercase font-mono px-2 py-0.5 rounded-md ${
                            item.isTop
                              ? 'bg-sky-200/80 text-sky-900 font-bold'
                              : 'bg-slate-100 text-slate-600 font-medium'
                          }`}
                        >
                          {item.code}
                        </span>
                        <span
                          className={`text-sm truncate ${
                            item.isTop
                              ? 'font-extrabold text-slate-900'
                              : 'font-medium text-slate-700'
                          }`}
                        >
                          {item.name}
                        </span>

                        {item.isTop && (
                          <div className="flex items-center space-x-1.5 shrink-0">
                            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold bg-sky-600 text-white shadow-2xs">
                              Top Prediction
                            </span>
                            <span
                              className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold tracking-wide border ${
                                finalRiskLevel === 'HIGH'
                                  ? 'bg-rose-100 text-rose-800 border-rose-300'
                                  : finalRiskLevel === 'MEDIUM'
                                  ? 'bg-amber-100 text-amber-800 border-amber-300'
                                  : 'bg-emerald-100 text-emerald-800 border-emerald-300'
                              }`}
                            >
                              {finalRiskLevel}
                            </span>
                          </div>
                        )}
                      </div>

                      <div className="text-right shrink-0">
                        <span
                          className={`text-sm ${
                            item.isTop
                              ? 'font-extrabold text-sky-800'
                              : 'font-semibold text-slate-700'
                          }`}
                        >
                          {precisePct}%
                        </span>
                      </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          item.isTop
                            ? finalRiskLevel === 'HIGH'
                              ? 'bg-rose-500'
                              : 'bg-sky-600'
                            : 'bg-slate-300'
                        }`}
                        style={{ width: `${Math.min(100, Math.max(1, pct))}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Section 4 — Visual Explanations (Grad-CAM) */}
          <div className="space-y-4">
            <div>
              <h2 className="text-base font-bold text-slate-900">
                Visual Explanations
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Explainability visualization where available
              </p>
            </div>

            {!gradcamAvailable || gradcamError ? (
              /* Fallback message for demo/smoke model */
              <div className="p-8 rounded-2xl border-2 border-dashed border-slate-200 bg-slate-50 text-center max-w-lg mx-auto">
                <AlertTriangle className="w-9 h-9 text-amber-500 mx-auto mb-2" />
                <h4 className="text-sm font-bold text-slate-800">
                  Visual explanation not available for demo model
                </h4>
                <p className="text-xs text-slate-500 mt-1 max-w-md mx-auto">
                  Grad-CAM activation overlays require convolutional feature maps from fully trained
                  production weights.
                </p>
              </div>
            ) : gradcamLoading ? (
              <div className="py-12 text-center text-slate-500 text-xs flex flex-col items-center justify-center space-y-2">
                <div className="w-7 h-7 border-3 border-sky-200 border-t-sky-600 rounded-full animate-spin" />
                <span>Loading Grad-CAM attention visualizations...</span>
              </div>
            ) : (
              /* 3 side-by-side images */
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* 1. Original Image */}
                <div className="bg-slate-50 rounded-2xl border border-slate-200 p-3.5 flex flex-col justify-between">
                  <div>
                    <div className="text-xs font-bold text-slate-800 mb-1">
                      1. Original Image
                    </div>
                    <p className="text-[11px] text-slate-400 mb-2">
                      Uploaded skin lesion photograph
                    </p>
                  </div>
                  <div className="relative rounded-xl overflow-hidden bg-slate-900 aspect-square flex items-center justify-center shadow-inner">
                    {imageUrl ? (
                      <img
                        src={imageUrl}
                        alt="Original Lesion"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <span className="text-xs text-slate-400">Photo unavailable</span>
                    )}
                  </div>
                </div>

                {/* 2. Grad-CAM Heatmap */}
                <div className="bg-slate-50 rounded-2xl border border-slate-200 p-3.5 flex flex-col justify-between">
                  <div>
                    <div className="text-xs font-bold text-slate-800 mb-1">
                      2. Grad-CAM Heatmap
                    </div>
                    <p className="text-[11px] text-slate-400 mb-2">
                      Pure jet colormap on black background
                    </p>
                  </div>
                  <div className="relative rounded-xl overflow-hidden bg-black aspect-square flex items-center justify-center shadow-inner">
                    {heatmapUrl ? (
                      <img
                        src={heatmapUrl}
                        alt="Grad-CAM Heatmap"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <span className="text-xs text-slate-400">Heatmap unavailable</span>
                    )}
                  </div>
                </div>

                {/* 3. Grad-CAM Overlay */}
                <div className="bg-slate-50 rounded-2xl border border-slate-200 p-3.5 flex flex-col justify-between">
                  <div>
                    <div className="text-xs font-bold text-slate-800 mb-1">
                      3. Grad-CAM Overlay
                    </div>
                    <p className="text-[11px] text-slate-400 mb-2">
                      Heatmap blended on original photo
                    </p>
                  </div>
                  <div className="relative rounded-xl overflow-hidden bg-slate-900 aspect-square flex items-center justify-center shadow-inner">
                    {overlayUrl ? (
                      <>
                        <img
                          src={overlayUrl}
                          alt="Grad-CAM Overlay"
                          className="w-full h-full object-cover"
                        />
                        <div className="absolute top-2.5 right-2.5 px-2.5 py-1 rounded-lg bg-slate-950/85 backdrop-blur-xs text-white font-mono font-bold text-[11px] border border-white/20 shadow-md">
                          {topLabel}
                        </div>
                      </>
                    ) : (
                      <span className="text-xs text-slate-400">Overlay unavailable</span>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Section 5 — Export */}
          <div className="pt-6 border-t border-slate-200/80 space-y-3">
            <div className="text-xs font-bold uppercase tracking-wider text-slate-500">
              Available Export Formats: PDF &bull; JSON &bull; CSV
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <button
                type="button"
                onClick={handleExportPDF}
                className="py-3 px-4 rounded-xl bg-sky-600 hover:bg-sky-700 text-white font-semibold text-xs flex items-center justify-center space-x-2 transition-all shadow-xs active:scale-[0.99] cursor-pointer"
              >
                <Download className="w-4 h-4 text-sky-200" />
                <span>Export PDF Report</span>
              </button>

              <button
                type="button"
                onClick={handleExportJSON}
                className="py-3 px-4 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-800 font-semibold text-xs flex items-center justify-center space-x-2 transition-all border border-slate-300 shadow-2xs active:scale-[0.99] cursor-pointer"
              >
                <FileText className="w-4 h-4 text-slate-600" />
                <span>Download JSON</span>
              </button>

              <button
                type="button"
                onClick={handleExportCSV}
                className="py-3 px-4 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-800 font-semibold text-xs flex items-center justify-center space-x-2 transition-all border border-slate-300 shadow-2xs active:scale-[0.99] cursor-pointer"
              >
                <Download className="w-4 h-4 text-slate-600" />
                <span>Download CSV</span>
              </button>
            </div>

            {/* Medical Disclaimer */}
            <div className="p-3 rounded-xl bg-amber-50/70 border border-amber-200/60 text-xs text-amber-800 flex items-start space-x-2 mt-4 leading-relaxed">
              <Info className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
              <span>
                <strong>Medical Notice:</strong> {disclaimer}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisReportModal;
