import React, { useState } from 'react';
import { jsPDF } from 'jspdf';
import {
  Sparkles,
  AlertTriangle,
  Layers,
  Eye,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Info,
  CheckCircle,
  Download,
} from 'lucide-react';
import { casesApi } from '../api/client';

const DIAGNOSTIC_NAMES = {
  mel: 'Melanoma',
  bcc: 'Basal Cell Carcinoma',
  akiec: "Actinic Keratosis / Bowen's Disease",
  bkl: 'Benign Keratosis-like Lesions',
  df: 'Dermatofibroma',
  nv: 'Melanocytic Nevus',
  vasc: 'Vascular Lesion',
};

export const getRiskInfo = (pred) => {
  if (!pred) return null;
  if (pred.risk_level) {
    const lvl = pred.risk_level.toUpperCase();
    const color = pred.risk_color || (lvl === 'HIGH' ? 'red' : lvl === 'MEDIUM' ? 'amber' : 'green');
    return {
      level: lvl,
      color,
      message:
        pred.risk_message ||
        (lvl === 'HIGH'
          ? 'Urgent dermatologist review recommended. High clinical priority.'
          : lvl === 'MEDIUM'
          ? 'Clinical consultation recommended.'
          : 'Routine monitoring recommended.'),
      urgencyDays: pred.urgency_days ?? (lvl === 'HIGH' ? 1 : lvl === 'MEDIUM' ? 14 : 90),
    };
  }
  const cls = (pred.predicted_class || '').toLowerCase();
  const conf = Number(pred.confidence || 0);

  if (cls === 'mel' || ((cls === 'bcc' || cls === 'akiec') && conf >= 0.5)) {
    return {
      level: 'HIGH',
      color: 'red',
      message: 'Urgent dermatologist review recommended. High clinical priority.',
      urgencyDays: 1,
    };
  }
  if (((cls === 'bcc' || cls === 'akiec') && conf < 0.5) || (cls === 'bkl' && conf < 0.4) || conf < 0.45) {
    return {
      level: 'MEDIUM',
      color: 'amber',
      message: 'Clinical consultation recommended. Low model certainty or moderate-risk lesion.',
      urgencyDays: 14,
    };
  }
  return {
    level: 'LOW',
    color: 'green',
    message: 'Routine monitoring recommended. Likely benign lesion with no urgent action needed.',
    urgencyDays: 90,
  };
};

export const ImageAnalysisCard = ({
  image,
  caseId,
  initialPredictions = [],
  onAnalysisUpdated,
  patientName,
  patientAge,
  imageUrl,
  allPredictions,
}) => {
  const [prediction, setPrediction] = useState(
    initialPredictions.length > 0 ? initialPredictions[0] : image.predictions?.[0] || null
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showOverlay, setShowOverlay] = useState(false);
  const [showTopN, setShowTopN] = useState(true);
  const [showGradcam, setShowGradcam] = useState(false);
  const [gradcamUrl, setGradcamUrl] = useState(null);
  const [gradcamLoading, setGradcamLoading] = useState(false);
  const [gradcamError, setGradcamError] = useState('');

  const isPoorQuality = (image.quality_label || '').toLowerCase() === 'poor';

  const handleAnalyze = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await casesApi.analyzeImage(image.id);
      setPrediction(res);
      setGradcamUrl(null);
      setShowGradcam(false);
      if (onAnalysisUpdated) {
        onAnalysisUpdated(res);
      }
    } catch (err) {
      const msg = err.response?.data?.detail || 'Analysis request failed. Please try again.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleGradcam = async () => {
    if (!showGradcam && !gradcamUrl) {
      setGradcamLoading(true);
      setGradcamError('');
      try {
        const url = await casesApi.getGradcamBlobUrl(image.id);
        setGradcamUrl(url);
      } catch (err) {
        console.error('Failed to load Grad-CAM overlay:', err);
        setGradcamError('Could not load AI attention map overlay.');
      } finally {
        setGradcamLoading(false);
      }
    }
    setShowGradcam((prev) => !prev);
  };

  const handleDownloadReport = async () => {
    if (!prediction) return;

    const doc = new jsPDF({ unit: 'mm', format: 'a4' });

    // Optional: resolve image for thumbnail in PDF
    let resolvedImageUrl = imageUrl;
    if (!resolvedImageUrl && caseId && image?.id) {
      try {
        resolvedImageUrl = await casesApi.getImageBlobUrl(caseId, image.id);
      } catch (err) {
        console.warn('Could not retrieve image blob for PDF:', err);
      }
    }

    let imageBase64 = null;
    if (resolvedImageUrl) {
      try {
        const imgEl = new Image();
        imgEl.crossOrigin = 'Anonymous';
        await new Promise((resolve) => {
          const timeout = setTimeout(() => resolve(null), 1500);
          imgEl.onload = () => {
            clearTimeout(timeout);
            resolve(imgEl);
          };
          imgEl.onerror = () => {
            clearTimeout(timeout);
            resolve(null);
          };
          imgEl.src = resolvedImageUrl;
        });

        if (imgEl.complete && imgEl.naturalWidth > 0) {
          const canvas = document.createElement('canvas');
          canvas.width = imgEl.naturalWidth;
          canvas.height = imgEl.naturalHeight;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(imgEl, 0, 0);
          imageBase64 = canvas.toDataURL('image/jpeg', 0.85);
        }
      } catch (err) {
        console.warn('Image embedding in PDF failed:', err);
      }
    }

    // 1. Header with blue background
    doc.setFillColor(2, 132, 199); // Sky-600 Blue
    doc.rect(0, 0, 210, 32, 'F');
    doc.setTextColor(255, 255, 255);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(22);
    doc.text('DermAI', 15, 16);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(11);
    doc.text('Clinical Skin Lesion Analysis Report', 15, 24);

    const reportDateStr = new Date().toISOString().split('T')[0];
    doc.setFontSize(9);
    doc.text(`Report Date: ${reportDateStr}`, 195, 24, { align: 'right' });

    // 2. Patient Information section
    let y = 42;
    doc.setTextColor(15, 23, 42);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(12);
    doc.text('Patient Information', 15, y);

    y += 4;
    const hasImage = Boolean(imageBase64);
    const boxWidth = hasImage ? 130 : 180;
    doc.setDrawColor(226, 232, 240);
    doc.setFillColor(248, 250, 252);
    doc.roundedRect(15, y, boxWidth, 30, 2, 2, 'FD');

    if (hasImage) {
      doc.roundedRect(150, y, 45, 30, 2, 2, 'FD');
      try {
        doc.addImage(imageBase64, 'JPEG', 151, y + 1, 43, 28);
      } catch (err) {
        console.warn('doc.addImage failed:', err);
      }
    }

    // Extract filename
    const imageFilename =
      image.filename ||
      image.original_filename ||
      (image.storage_key ? image.storage_key.split('/').pop() : '') ||
      (resolvedImageUrl && !resolvedImageUrl.startsWith('blob:') ? resolvedImageUrl.split('/').pop().split('?')[0] : '') ||
      `scan_image_${image.id || '1'}.jpg`;

    const pY = y + 7;
    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(71, 85, 105);
    doc.text('Patient Name:', 20, pY);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(15, 23, 42);
    doc.text(String(patientName || 'Patient'), 46, pY);

    doc.setFont('helvetica', 'bold');
    doc.setTextColor(71, 85, 105);
    doc.text('Case ID:', hasImage ? 80 : 110, pY);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(15, 23, 42);
    doc.text(`#${caseId || image.case_id || 'N/A'}`, hasImage ? 98 : 130, pY);

    doc.setFont('helvetica', 'bold');
    doc.setTextColor(71, 85, 105);
    doc.text('Patient Age:', 20, pY + 8);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(15, 23, 42);
    const ageDisplay = patientAge !== undefined && patientAge !== null && patientAge !== '' ? `${patientAge} years` : 'N/A';
    doc.text(ageDisplay, 46, pY + 8);

    doc.setFont('helvetica', 'bold');
    doc.setTextColor(71, 85, 105);
    doc.text('Image File:', hasImage ? 80 : 110, pY + 8);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(15, 23, 42);
    const maxChars = hasImage ? 16 : 24;
    const shortFilename = imageFilename.length > maxChars ? imageFilename.slice(0, maxChars - 3) + '...' : imageFilename;
    doc.text(shortFilename, hasImage ? 98 : 130, pY + 8);

    doc.setFont('helvetica', 'bold');
    doc.setTextColor(71, 85, 105);
    doc.text('Quality Assessment:', 20, pY + 16);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(15, 23, 42);
    doc.text(image.quality_label ? image.quality_label.toUpperCase() : 'VERIFIED', 54, pY + 16);

    // 3. AI Analysis Results section
    y += 38;
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(12);
    doc.setTextColor(15, 23, 42);
    doc.text('AI Analysis Results', 15, y);

    // Highlighted box for top prediction
    y += 4;
    doc.setDrawColor(186, 230, 253);
    doc.setFillColor(240, 249, 255);
    doc.roundedRect(15, y, 180, 22, 2, 2, 'FD');

    y += 6;
    doc.setFontSize(8);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(3, 105, 161);
    doc.text('TOP PREDICTED CONDITION', 20, y);
    doc.setTextColor(71, 85, 105);
    doc.text('CONFIDENCE SCORE', 140, y);

    y += 8;
    doc.setFontSize(13);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(15, 23, 42);
    doc.text(`${conditionName} (${prediction.predicted_class || ''})`, 20, y);

    doc.setFontSize(15);
    doc.setTextColor(2, 132, 199);
    doc.text(`${confidencePct}%`, 140, y);

    // Build probability map for all conditions
    const ALL_CONDITIONS_META = [
      { code: 'mel', name: 'Melanoma' },
      { code: 'nv', name: 'Melanocytic Nevus' },
      { code: 'bkl', name: 'Benign Keratosis-like Lesions' },
      { code: 'bcc', name: 'Basal Cell Carcinoma' },
      { code: 'akiec', name: "Actinic Keratosis / Bowen's Disease" },
      { code: 'vasc', name: 'Vascular Lesion' },
      { code: 'df', name: 'Dermatofibroma' },
    ];

    const probMap = {};
    if (prediction.predicted_class && prediction.confidence !== undefined) {
      probMap[prediction.predicted_class.toLowerCase()] = prediction.confidence;
    }
    if (Array.isArray(prediction.top_n)) {
      prediction.top_n.forEach((item) => {
        const c = (item.class || item.code || '').toLowerCase();
        if (c) {
          probMap[c] = item.probability !== undefined ? item.probability : (item.confidence || 0);
        }
      });
    }
    if (Array.isArray(allPredictions)) {
      allPredictions.forEach((item) => {
        const c = (item.class || item.code || '').toLowerCase();
        if (c && (item.probability !== undefined || item.confidence !== undefined)) {
          probMap[c] = item.probability !== undefined ? item.probability : item.confidence;
        } else if (item.top_n && Array.isArray(item.top_n)) {
          item.top_n.forEach((sub) => {
            const sc = (sub.class || sub.code || '').toLowerCase();
            if (sc && probMap[sc] === undefined) {
              probMap[sc] = sub.probability !== undefined ? sub.probability : sub.confidence;
            }
          });
        }
      });
    }

    const tableRows = ALL_CONDITIONS_META.map((meta) => {
      const prob = probMap[meta.code.toLowerCase()];
      return {
        code: meta.code,
        name: meta.name,
        probValue: prob !== undefined ? prob : 0,
        pct: prob !== undefined ? `${(prob * 100).toFixed(1)}%` : '< 1.0%',
      };
    });
    tableRows.sort((a, b) => b.probValue - a.probValue);

    // Clinical Risk Assessment box in PDF
    const pdfRisk = getRiskInfo(prediction);
    if (pdfRisk) {
      y += 12;
      const isHigh = pdfRisk.level === 'HIGH';
      const isMed = pdfRisk.level === 'MEDIUM';

      const boxBorder = isHigh ? [254, 205, 211] : isMed ? [253, 230, 138] : [167, 243, 208];
      const boxBg = isHigh ? [255, 241, 242] : isMed ? [254, 252, 232] : [240, 253, 244];
      const titleColor = isHigh ? [159, 18, 57] : isMed ? [146, 64, 14] : [6, 95, 70];
      const msgColor = isHigh ? [190, 18, 60] : isMed ? [180, 83, 9] : [4, 120, 87];

      doc.setDrawColor(...boxBorder);
      doc.setFillColor(...boxBg);
      doc.roundedRect(15, y, 180, 15, 2, 2, 'FD');

      doc.setFontSize(8);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(...titleColor);
      doc.text(
        `CLINICAL RISK LEVEL: ${pdfRisk.level} (Recommended follow-up: within ${pdfRisk.urgencyDays} day${pdfRisk.urgencyDays === 1 ? '' : 's'})`,
        20,
        y + 5
      );

      doc.setFontSize(8.5);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(...msgColor);
      doc.text(pdfRisk.message, 20, y + 10.5);

      y += 18;
    } else {
      y += 15;
    }

    // Table showing ALL predicted conditions with their confidence percentages
    doc.setFontSize(10);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(15, 23, 42);
    doc.text('All Predicted Conditions & Probabilities', 15, y);

    y += 4;
    doc.setFillColor(224, 242, 254);
    doc.setDrawColor(186, 230, 253);
    doc.rect(15, y, 180, 7, 'FD');
    doc.setFontSize(8);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(3, 105, 161);
    doc.text('#', 18, y + 5);
    doc.text('CONDITION / PATHOLOGY', 28, y + 5);
    doc.text('CODE', 125, y + 5);
    doc.text('CONFIDENCE', 160, y + 5);

    tableRows.forEach((row, idx) => {
      y += 6.5;
      doc.setFillColor(idx % 2 === 0 ? 255 : 248, idx % 2 === 0 ? 255 : 250, idx % 2 === 0 ? 255 : 252);
      doc.setDrawColor(226, 232, 240);
      doc.rect(15, y, 180, 6.5, 'FD');
      doc.setFont('helvetica', idx === 0 ? 'bold' : 'normal');
      doc.setTextColor(idx === 0 ? 2 : 15, idx === 0 ? 132 : 23, idx === 0 ? 199 : 42);
      doc.text(String(idx + 1), 18, y + 4.5);
      doc.text(row.name, 28, y + 4.5);
      doc.text(row.code, 125, y + 4.5);
      doc.text(row.pct, 160, y + 4.5);
    });

    // 4. Disclaimer box at the bottom
    y += 14;
    doc.setDrawColor(254, 215, 170);
    doc.setFillColor(255, 247, 237);
    doc.roundedRect(15, y, 180, 13, 2, 2, 'FD');
    doc.setFontSize(8.5);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(154, 52, 18);
    doc.text(
      'AI-assisted analysis only. Not a medical diagnosis. Consult a qualified dermatologist.',
      105,
      y + 7.5,
      { align: 'center' }
    );

    // 5. Footer
    doc.setDrawColor(226, 232, 240);
    doc.line(15, 280, 195, 280);
    doc.setFontSize(8);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(100, 116, 139);
    doc.text(
      'DermAI — Clinical Support Tool | Dr. MGR Educational and Research Institute, Chennai',
      105,
      286,
      { align: 'center' }
    );

    // 6. Save PDF
    const cleanPatient = (patientName || 'Patient')
      .toString()
      .trim()
      .replace(/\s+/g, '_')
      .replace(/[^a-zA-Z0-9_-]/g, '');
    const filename = `DermAI_Report_${cleanPatient || 'Patient'}_${reportDateStr}.pdf`;
    doc.save(filename);
  };
  const handleDownloadPdf = handleDownloadReport;

  const predClass = prediction?.predicted_class?.toLowerCase();
  const conditionName =
    DIAGNOSTIC_NAMES[predClass] || prediction?.predicted_class_name || prediction?.predicted_class || 'Lesion Analysis';

  const confidencePct = prediction ? Math.round(prediction.confidence * 100) : 0;
  const isLowConf = prediction?.is_low_confidence || (prediction && prediction.confidence < 0.60);
  const riskInfo = getRiskInfo(prediction);

  return (
    <div className="mt-4 pt-4 border-t border-slate-200/80 space-y-3">
      {/* Action / Trigger Header */}
      {!prediction && (
        <div>
          <button
            onClick={handleAnalyze}
            disabled={loading}
            className={`w-full py-2.5 px-4 rounded-xl text-xs font-semibold flex items-center justify-center space-x-2 transition-all shadow-xs ${
              isPoorQuality
                ? 'bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-300'
                : 'bg-sky-600 hover:bg-sky-700 text-white active:scale-[0.99]'
            } disabled:opacity-60 disabled:cursor-not-allowed`}
          >
            {loading ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white/40 border-t-white rounded-full animate-spin" />
                <span>Running AI Analysis...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 text-sky-200" />
                <span>Analyze with AI</span>
              </>
            )}
          </button>

          {isPoorQuality && (
            <p className="text-[11px] text-amber-700 bg-amber-50/80 p-2 rounded-lg border border-amber-200/60 mt-2 text-center">
              Notice: Image quality is marked as 'poor'. System may request a retake.
            </p>
          )}
        </div>
      )}

      {/* Error Banner */}
      {error && (
        <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start space-x-2">
          <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
          <div className="leading-relaxed">
            <span className="font-semibold block">Analysis Notice:</span>
            {error}
          </div>
        </div>
      )}

      {/* Prediction Results Box */}
      {prediction && (
        <div className="rounded-xl bg-white border border-slate-200/90 p-3.5 space-y-3 shadow-xs">
          {/* Header: Condition & Re-analyze */}
          <div className="flex items-start justify-between gap-2">
            <div>
              <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400">
                Top prediction
              </div>
              <div className="text-sm font-bold text-slate-900 mt-0.5">
                {conditionName}
                <span className="text-xs font-normal text-slate-500 ml-1.5 uppercase font-mono">
                  ({prediction.predicted_class})
                </span>
              </div>
            </div>

            <button
              onClick={handleAnalyze}
              disabled={loading}
              title="Re-run analysis"
              className="text-slate-400 hover:text-sky-600 p-1 rounded-lg hover:bg-slate-100 transition-colors"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>

          {/* 3-Tier Clinical Risk Stratification */}
          {riskInfo && (
            <div className="space-y-1.5 pt-0.5">
              <div className="flex items-center space-x-2">
                <span
                  className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-bold tracking-wide border shadow-2xs ${
                    riskInfo.color === 'red' || riskInfo.level === 'HIGH'
                      ? 'bg-rose-100 text-rose-800 border-rose-300'
                      : riskInfo.color === 'amber' || riskInfo.level === 'MEDIUM'
                      ? 'bg-amber-100 text-amber-800 border-amber-300'
                      : 'bg-emerald-100 text-emerald-800 border-emerald-300'
                  }`}
                >
                  {riskInfo.level}
                </span>
                <span className="text-[11px] text-slate-600 font-medium">
                  Recommended follow-up: within {riskInfo.urgencyDays} day{riskInfo.urgencyDays === 1 ? '' : 's'}
                </span>
              </div>

              {/* Risk Message Box */}
              <div
                className={`p-2.5 rounded-xl border text-xs leading-relaxed ${
                  riskInfo.color === 'red' || riskInfo.level === 'HIGH'
                    ? 'bg-rose-50/90 border-rose-200 text-rose-900'
                    : riskInfo.color === 'amber' || riskInfo.level === 'MEDIUM'
                    ? 'bg-amber-50/90 border-amber-200 text-amber-900'
                    : 'bg-emerald-50/90 border-emerald-200 text-emerald-900'
                }`}
              >
                <p className="font-medium">{riskInfo.message}</p>
              </div>
            </div>
          )}

          {/* Confidence Progress Bar */}
          <div>
            <div className="flex justify-between items-center text-xs mb-1">
              <span className="text-slate-600 font-medium">Confidence Score</span>
              <span className={`font-bold ${isLowConf ? 'text-amber-600' : 'text-sky-700'}`}>
                {confidencePct}%
              </span>
            </div>
            <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  isLowConf ? 'bg-amber-500' : 'bg-sky-600'
                }`}
                style={{ width: `${Math.min(100, Math.max(5, confidencePct))}%` }}
              />
            </div>
          </div>

          {/* AI Attention Map (Grad-CAM) - Only show when gradcam_key exists and not a demo/smoke model */}
          {prediction.gradcam_key && !prediction.demo_model && (
            <div className="pt-1.5 border-t border-slate-100">
              <button
                type="button"
                onClick={handleToggleGradcam}
                className="inline-flex items-center space-x-1.5 text-[11px] font-semibold text-indigo-600 hover:text-indigo-700 py-1 transition-colors"
              >
                <Eye className="w-3.5 h-3.5" />
                <span>{showGradcam ? 'Hide AI Attention Map' : 'View AI Attention Map'}</span>
              </button>

              {gradcamLoading && (
                <div className="mt-2 text-xs text-slate-500 flex items-center space-x-2">
                  <div className="w-3 h-3 border-2 border-indigo-600/40 border-t-indigo-600 rounded-full animate-spin" />
                  <span>Loading attention map...</span>
                </div>
              )}

              {gradcamError && (
                <p className="mt-1 text-xs text-rose-600">{gradcamError}</p>
              )}

              {showGradcam && gradcamUrl && (
                <div className="mt-2 rounded-xl overflow-hidden border border-slate-200 bg-black/5 shadow-inner">
                  <img
                    src={gradcamUrl}
                    alt="AI Attention Map (Grad-CAM)"
                    className="w-full h-44 object-cover"
                  />
                  <div className="p-2 bg-slate-900/90 text-[10px] text-slate-200 text-center font-medium">
                    Highlighted regions show which areas influenced the AI prediction
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Other possible conditions (top-3 predictions) */}
          {prediction.top_n && prediction.top_n.length > 0 && (
            <div className="pt-1 border-t border-slate-100">
              <button
                onClick={() => setShowTopN(!showTopN)}
                className="w-full flex items-center justify-between text-[11px] font-semibold text-slate-500 hover:text-slate-800 py-1"
              >
                <span>Other possible conditions</span>
                {showTopN ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </button>

              {showTopN && (
                <div className="space-y-1.5 mt-1.5">
                  {prediction.top_n.slice(0, 3).map((item, idx) => {
                    const pct = Math.round((item.probability || 0) * 100);
                    const label = item.label || DIAGNOSTIC_NAMES[item.class] || item.class;
                    return (
                      <div key={idx} className="text-xs">
                        <div className="flex justify-between text-[11px] text-slate-600 mb-0.5">
                          <span className="truncate">{label}</span>
                          <span className="font-semibold text-slate-800">{pct}%</span>
                        </div>
                        <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                          <div
                            className={`h-full rounded-full ${idx === 0 ? 'bg-sky-500' : 'bg-slate-400'}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* Segmentation Overlay (only if not smoke weights) */}
          {prediction.segmentation_overlay && (
            <div className="pt-2 border-t border-slate-100">
              <button
                onClick={() => setShowOverlay(!showOverlay)}
                className="inline-flex items-center space-x-1.5 text-[11px] font-semibold text-sky-600 hover:text-sky-700 py-1"
              >
                <Layers className="w-3.5 h-3.5" />
                <span>{showOverlay ? 'Hide Lesion Segmentation' : 'View Lesion Segmentation'}</span>
              </button>

              {showOverlay && (
                <div className="mt-2 rounded-xl overflow-hidden border border-slate-200 bg-black/5">
                  <img
                    src={prediction.segmentation_overlay}
                    alt="Lesion Segmentation Overlay"
                    className="w-full h-40 object-cover"
                  />
                  <div className="p-1.5 bg-slate-900/80 text-[10px] text-slate-200 text-center font-medium">
                    Green contour: Lesion boundary mask &bull; Gold: Bounding ROI
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Metadata & Version */}
          <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-[10px] text-slate-400">
            <span>Model: <span className="font-mono text-slate-600">{prediction.model_version}</span></span>
            <span className="flex items-center space-x-1">
              <CheckCircle className="w-3 h-3 text-emerald-500" />
              <span>Decision Support Ready</span>
            </span>
          </div>

          {/* Download Report as PDF */}
          <button
            type="button"
            onClick={handleDownloadPdf}
            className="w-full py-2 px-3 rounded-xl bg-sky-50 hover:bg-sky-100 text-sky-700 border border-sky-200/80 text-xs font-semibold flex items-center justify-center space-x-2 transition-colors shadow-xs active:scale-[0.99]"
          >
            <Download className="w-4 h-4 text-sky-600" />
            <span>Download Report as PDF</span>
          </button>

          {/* Mandatory Medical Disclaimer */}
          <div className="p-2 rounded-lg bg-sky-50/70 border border-sky-100 text-[11px] text-sky-800 flex items-start space-x-1.5 leading-snug">
            <Info className="w-3.5 h-3.5 text-sky-600 shrink-0 mt-0.5" />
            <span className="italic">
              {prediction.disclaimer || 'AI-assisted analysis. Not a medical diagnosis.'}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
