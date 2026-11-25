import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { casesApi } from '../api/client';
import { useAuth } from '../context/AuthContext';
import { QualityBadge } from '../components/QualityBadge';
import { ImageAnalysisCard } from '../components/ImageAnalysisCard';
import { AnalysisReportModal, DISEASE_MAP, DIAGNOSTIC_NAMES } from '../components/AnalysisReportModal';
import {
  ArrowLeft,
  PlusCircle,
  Calendar,
  User,
  MapPin,
  FileText,
  Image as ImageIcon,
  AlertTriangle,
  Layers,
  Sparkles,
  RefreshCw,
  CheckCircle,
} from 'lucide-react';

const ImagePreviewThumbnail = ({ caseId, imageId, alt, src }) => {
  const [blobUrl, setBlobUrl] = useState(src || null);
  const [loading, setLoading] = useState(!src);

  useEffect(() => {
    if (src) {
      setBlobUrl(src);
      setLoading(false);
      return;
    }
    let active = true;
    casesApi.getImageBlobUrl(caseId, imageId)
      .then((url) => {
        if (active) {
          setBlobUrl(url);
          setLoading(false);
        }
      })
      .catch(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [caseId, imageId, src]);

  if (loading) {
    return (
      <div className="h-44 bg-slate-100 flex items-center justify-center text-slate-400 text-xs">
        Loading photo...
      </div>
    );
  }

  if (!blobUrl) {
    return (
      <div className="h-44 bg-slate-100 flex items-center justify-center text-slate-400 text-xs">
        Photo preview unavailable
      </div>
    );
  }

  return (
    <div className="h-48 bg-slate-900/5 relative overflow-hidden flex items-center justify-center border-b border-slate-100">
      <img src={blobUrl} alt={alt} className="w-full h-full object-cover" />
    </div>
  );
};

export const CaseDetailPage = () => {
  const { caseId } = useParams();
  const { user } = useAuth();
  const navigate = useNavigate();
  const [caseData, setCaseData] = useState(null);
  const [images, setImages] = useState([]);
  const [imageUrls, setImageUrls] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [isNotFound, setIsNotFound] = useState(false);
  const [selectedModalImage, setSelectedModalImage] = useState(null);
  const [selectedModalPred, setSelectedModalPred] = useState(null);
  const [isReportModalOpen, setIsReportModalOpen] = useState(false);
  const [analyzingImageId, setAnalyzingImageId] = useState(null);
  const [analysisErrors, setAnalysisErrors] = useState({});

  const handleOpenReport = (image, pred = null) => {
    const targetPred =
      pred ||
      image.latestPrediction ||
      (image.predictions && image.predictions.length > 0 ? image.predictions[0] : null);
    setSelectedModalImage(image);
    setSelectedModalPred(targetPred);
    setIsReportModalOpen(true);
  };

  const handleAnalyzeImage = async (image) => {
    setAnalyzingImageId(image.id);
    setAnalysisErrors((prev) => ({ ...prev, [image.id]: '' }));
    try {
      const predResult = await casesApi.analyzeImage(image.id);
      toast.success('Analysis complete!');
      const updatedImage = {
        ...image,
        latestPrediction: predResult,
        predictions: [predResult, ...(image.predictions || [])],
      };
      setImages((prev) =>
        prev.map((img) => (img.id === image.id ? updatedImage : img))
      );
      setSelectedModalImage(updatedImage);
      setSelectedModalPred(predResult);
      setIsReportModalOpen(true);
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.message ||
        'AI Analysis failed. Please check image quality or server connection.';
      setAnalysisErrors((prev) => ({ ...prev, [image.id]: msg }));
      toast.error(msg);
    } finally {
      setAnalyzingImageId(null);
    }
  };

  const loadData = async () => {
    setLoading(true);
    setError('');
    setIsNotFound(false);
    try {
      const [caseRes, imagesRes] = await Promise.all([
        casesApi.getCase(caseId),
        casesApi.getCaseImages(caseId),
      ]);
      setCaseData(caseRes);
      setImages(imagesRes);
    } catch (err) {
      if (err.response?.status === 404) {
        setIsNotFound(true);
        setError('Case not found');
      } else if (!err.response) {
        setIsNotFound(false);
        setError('Failed to load case. Please check your connection.');
      } else {
        setIsNotFound(false);
        setError(err.response?.data?.detail || 'Failed to load case. Please check your connection.');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    document.title = 'Case Details | DermAI';
    loadData();
  }, [caseId]);

  useEffect(() => {
    if (!images || images.length === 0) return;
    let active = true;
    images.forEach((img) => {
      casesApi.getImageBlobUrl(caseId, img.id)
        .then((url) => {
          if (active) {
            setImageUrls((prev) => ({ ...prev, [img.id]: url }));
          }
        })
        .catch((err) => {
          console.warn(`Failed to preload image blob for #${img.id}:`, err);
        });
    });
    return () => {
      active = false;
    };
  }, [caseId, images]);

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-20 text-center">
        <div className="w-8 h-8 border-3 border-sky-200 border-t-sky-600 rounded-full animate-spin mx-auto mb-3"></div>
        <p className="text-sm text-slate-500">Loading case details...</p>
      </div>
    );
  }

  if (isNotFound) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-16 text-center">
        <div className="p-8 rounded-2xl bg-slate-50 border border-slate-200 text-slate-800 shadow-xs">
          <AlertTriangle className="w-10 h-10 text-amber-500 mx-auto mb-3" />
          <h2 className="text-lg font-bold text-slate-900">Case not found</h2>
          <p className="text-sm text-slate-500 mt-1 mb-6">
            The requested clinical screening case #{caseId} was not found or has been removed.
          </p>
          <button
            onClick={() => navigate(-1)}
            className="inline-flex items-center space-x-1.5 px-4 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold shadow-xs transition-all"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Go Back</span>
          </button>
        </div>
      </div>
    );
  }

  if (error || !caseData) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-16 text-center">
        <div className="p-6 rounded-2xl bg-rose-50 border border-rose-200 text-rose-800">
          <AlertTriangle className="w-8 h-8 text-rose-600 mx-auto mb-2" />
          <h2 className="text-base font-bold">Error loading case</h2>
          <p className="text-sm mt-1 mb-4">{error || 'Failed to load case. Please check your connection.'}</p>
          <button
            onClick={() => navigate(-1)}
            className="inline-flex items-center space-x-1.5 text-xs font-semibold text-rose-700 underline"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Go Back</span>
          </button>
        </div>
      </div>
    );
  }

  const formattedDate = new Date(caseData.created_at).toLocaleDateString(undefined, {
    month: 'long',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

  const patientName =
    caseData.patient?.name ||
    caseData.patient_name ||
    (user?.role === 'patient' && user?.full_name ? user.full_name : null) ||
    (user?.full_name ? user.full_name : null) ||
    `Patient #${caseData.patient_id || caseId}`;

  const patientAge =
    caseData.patient?.age !== undefined && caseData.patient?.age !== null
      ? `${caseData.patient.age}`
      : 'N/A';

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10">
      {/* Navigation */}
      <div className="mb-6 flex items-center justify-between">
        <Link
          to="/dashboard"
          className="inline-flex items-center space-x-1.5 text-xs font-semibold text-slate-500 hover:text-slate-900 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Dashboard</span>
        </Link>

        <Link
          to={`/cases/${caseId}/upload`}
          className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-sky-600 hover:bg-sky-700 text-white font-semibold text-xs shadow-sm transition-all"
        >
          <PlusCircle className="w-4 h-4" />
          <span>Upload Image</span>
        </Link>
      </div>

      {/* Case Overview Card */}
      <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-6 sm:p-8 mb-8">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-100">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 rounded-2xl bg-sky-100 text-sky-700 flex items-center justify-center font-bold text-base border border-sky-200/60 shrink-0">
              #{caseData.id}
            </div>
            <div>
              <div className="flex items-center space-x-2.5">
                <h1 className="text-xl sm:text-2xl font-bold text-slate-900">
                  {caseData.body_location}
                </h1>
                <span className="text-xs uppercase font-semibold tracking-wider px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-700">
                  {caseData.status}
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-1 flex items-center space-x-1">
                <Calendar className="w-3.5 h-3.5" />
                <span>Submitted on {formattedDate}</span>
              </p>
            </div>
          </div>

          {(() => {
            const firstWithPred = images.find(
              (img) => img.latestPrediction || (img.predictions && img.predictions.length > 0)
            );
            if (!firstWithPred) return null;
            return (
              <button
                type="button"
                onClick={() =>
                  handleOpenReport(
                    firstWithPred,
                    firstWithPred.latestPrediction || firstWithPred.predictions[0]
                  )
                }
                className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-sky-50 hover:bg-sky-100 text-sky-700 font-semibold text-xs border border-sky-200 shadow-2xs transition-all cursor-pointer active:scale-[0.99]"
              >
                <FileText className="w-4 h-4 text-sky-600" />
                <span>View Analysis Report</span>
              </button>
            );
          })()}
        </div>

        {/* Demographics Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 pt-6">
          <div className="flex items-start space-x-3">
            <div className="w-9 h-9 rounded-xl bg-slate-50 border border-slate-200 flex items-center justify-center text-slate-600 shrink-0">
              <User className="w-4 h-4" />
            </div>
            <div>
              <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Patient Demographics</div>
              <div className="text-sm font-bold text-slate-900 mt-0.5">
                {caseData.patient?.age ? `${caseData.patient.age} years old` : 'Age unrecorded'},{' '}
                <span className="capitalize">{caseData.patient?.gender || 'Unspecified'}</span>
              </div>
            </div>
          </div>

          <div className="flex items-start space-x-3">
            <div className="w-9 h-9 rounded-xl bg-slate-50 border border-slate-200 flex items-center justify-center text-slate-600 shrink-0">
              <MapPin className="w-4 h-4" />
            </div>
            <div>
              <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Body Site</div>
              <div className="text-sm font-bold text-slate-900 mt-0.5">{caseData.body_location}</div>
            </div>
          </div>

          <div className="flex items-start space-x-3">
            <div className="w-9 h-9 rounded-xl bg-slate-50 border border-slate-200 flex items-center justify-center text-slate-600 shrink-0">
              <ImageIcon className="w-4 h-4" />
            </div>
            <div>
              <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Scans Captured</div>
              <div className="text-sm font-bold text-slate-900 mt-0.5">{images.length} Photographs</div>
            </div>
          </div>
        </div>

        {/* Clinical Notes Box */}
        {caseData.patient?.notes && (
          <div className="mt-6 pt-6 border-t border-slate-100 bg-slate-50/70 p-4 rounded-xl border border-slate-200/60">
            <div className="flex items-center space-x-1.5 text-xs font-semibold text-slate-700 mb-1">
              <FileText className="w-3.5 h-3.5 text-slate-500" />
              <span>Clinical Observations:</span>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">{caseData.patient.notes}</p>
          </div>
        )}
      </div>

      {/* Images Section */}
      <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-6 sm:p-8">
        <div className="flex items-center justify-between pb-6 border-b border-slate-100 mb-6">
          <div>
            <h2 className="text-lg font-bold text-slate-900">Lesion Photographs &amp; Quality Assessments</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Review captured images, automated blur &amp; contrast quality labels, and MinIO storage references
            </p>
          </div>

          <Link
            to={`/cases/${caseId}/upload`}
            className="text-xs font-semibold text-sky-600 hover:text-sky-700 flex items-center space-x-1"
          >
            <PlusCircle className="w-4 h-4" />
            <span>Add Photograph</span>
          </Link>
        </div>

        {images.length === 0 ? (
          <div className="py-12 text-center max-w-sm mx-auto">
            <div className="w-14 h-14 rounded-2xl bg-sky-50 text-sky-500 flex items-center justify-center mx-auto mb-3">
              <ImageIcon className="w-7 h-7" />
            </div>
            <h3 className="text-sm font-semibold text-slate-800">No images uploaded yet</h3>
            <p className="text-xs text-slate-500 mt-1 mb-4">
              Capture or upload a lesion photograph with automated quality checks.
            </p>
            <Link
              to={`/cases/${caseId}/upload`}
              className="inline-flex items-center space-x-2 px-3.5 py-2 rounded-xl bg-sky-600 text-white font-medium text-xs hover:bg-sky-700 transition-colors shadow-xs"
            >
              <PlusCircle className="w-3.5 h-3.5" />
              <span>Upload Photo</span>
            </Link>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {images.map((img) => {
              const details = img.quality_details || {};
              const captureTime = new Date(img.captured_at).toLocaleDateString(undefined, {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
              });

              return (
                <div
                  key={img.id}
                  className="rounded-2xl border border-slate-200 overflow-hidden bg-slate-50/50 hover:shadow-md transition-shadow flex flex-col"
                >
                  {/* Photo Header Card */}
                  <div className="p-4 bg-white border-b border-slate-100 flex items-center justify-between">
                    <div>
                      <div className="font-semibold text-sm text-slate-800">Image #{img.id}</div>
                      <div className="text-[11px] text-slate-400 capitalize">{img.source} &bull; {captureTime}</div>
                    </div>
                    <QualityBadge label={img.quality_label} size="sm" />
                  </div>

                  {/* Lesion Photo Preview */}
                  <ImagePreviewThumbnail
                    caseId={caseId}
                    imageId={img.id}
                    alt={`Lesion scan ${img.id}`}
                    src={imageUrls[img.id]}
                  />

                  {/* Quality Details Body */}
                  <div className="p-4 flex-1 flex flex-col justify-between space-y-4">
                    {/* Storage Key reference */}
                    <div className="bg-white p-2.5 rounded-xl border border-slate-200 text-[11px] text-slate-600 font-mono truncate">
                      <span className="text-slate-400 block text-[9px] uppercase tracking-wider font-sans font-semibold">
                        MinIO Storage Key
                      </span>
                      {img.storage_key}
                    </div>

                    {/* Metrics grid */}
                    <div className="grid grid-cols-3 gap-2 text-center text-xs">
                      <div className="bg-white p-2 rounded-lg border border-slate-200/80">
                        <div className="text-[10px] text-slate-400">Blur</div>
                        <div className="font-bold text-slate-800 mt-0.5">{details.blur_score ?? 'N/A'}</div>
                      </div>
                      <div className="bg-white p-2 rounded-lg border border-slate-200/80">
                        <div className="text-[10px] text-slate-400">Brightness</div>
                        <div className="font-bold text-slate-800 mt-0.5">{details.brightness ?? 'N/A'}</div>
                      </div>
                      <div className="bg-white p-2 rounded-lg border border-slate-200/80">
                        <div className="text-[10px] text-slate-400">Contrast</div>
                        <div className="font-bold text-slate-800 mt-0.5">{details.contrast ?? 'N/A'}</div>
                      </div>
                    </div>

                    {/* Dimensions & Issues */}
                    <div className="text-xs text-slate-500 space-y-1">
                      <div>
                        Resolution:{' '}
                        <span className="font-semibold text-slate-700">
                          {details.resolution?.width || details.width || 'N/A'}x
                          {details.resolution?.height || details.height || 'N/A'}
                        </span>
                      </div>

                      {details.issues && details.issues.length > 0 && (
                        <div className="text-[11px] text-amber-700 bg-amber-50/70 p-2 rounded-lg border border-amber-200/60 leading-tight">
                          <span className="font-semibold">Notice:</span> {details.issues.join(', ')}
                        </div>
                      )}
                    </div>

                    {/* AI Decision Support & Lesion Analysis (Milestone 3 Analysis Report Modal Integration) */}
                    <div className="mt-4 pt-4 border-t border-slate-200/80 space-y-3">
                      {/* Analysis Error Notice */}
                      {analysisErrors[img.id] && (
                        <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start space-x-2">
                          <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                          <div className="leading-relaxed">
                            <span className="font-semibold block">Analysis Notice:</span>
                            {analysisErrors[img.id]}
                          </div>
                        </div>
                      )}

                      {(() => {
                        const currentPred =
                          img.latestPrediction ||
                          (img.predictions && img.predictions.length > 0 ? img.predictions[0] : null);
                        const isPoor = (img.quality_label || '').toLowerCase() === 'poor';

                        if (currentPred) {
                          const clsCode = (currentPred.predicted_class || '').toLowerCase();
                          const clsName =
                            DIAGNOSTIC_NAMES[clsCode] ||
                            DIAGNOSTIC_NAMES[clsCode.toLowerCase()] ||
                            currentPred.predicted_class_name ||
                            clsCode.toUpperCase();
                          const confPct = Math.round((currentPred.confidence || 0) * 100);
                          const riskLvl = (
                            currentPred.risk_level || (clsCode === 'mel' ? 'HIGH' : 'LOW')
                          ).toUpperCase();

                          return (
                            <div className="space-y-3">
                              {/* Prediction Summary Pill */}
                              <div className="rounded-xl bg-slate-50 border border-slate-200/90 p-3 space-y-2">
                                <div className="flex items-center justify-between">
                                  <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400">
                                    AI Diagnosis
                                  </span>
                                  <span
                                    className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold tracking-wide border ${
                                      riskLvl === 'HIGH'
                                        ? 'bg-rose-100 text-rose-800 border-rose-300'
                                        : riskLvl === 'MEDIUM'
                                        ? 'bg-amber-100 text-amber-800 border-amber-300'
                                        : 'bg-emerald-100 text-emerald-800 border-emerald-300'
                                    }`}
                                  >
                                    {riskLvl} RISK
                                  </span>
                                </div>
                                <div className="flex items-center justify-between text-xs">
                                  <span className="font-bold text-slate-900 truncate">
                                    {clsName}
                                  </span>
                                  <span className="font-bold text-sky-700 ml-2">
                                    {confPct}%
                                  </span>
                                </div>
                              </div>

                              {/* View Analysis Report Button */}
                              <button
                                type="button"
                                onClick={() => handleOpenReport(img, currentPred)}
                                className="w-full py-2.5 px-4 rounded-xl bg-sky-600 hover:bg-sky-700 text-white text-xs font-semibold flex items-center justify-center space-x-2 transition-all shadow-xs active:scale-[0.99] cursor-pointer"
                              >
                                <FileText className="w-4 h-4 text-sky-200" />
                                <span>View Analysis Report</span>
                              </button>

                              {/* Re-analyze link */}
                              <button
                                type="button"
                                onClick={() => handleAnalyzeImage(img)}
                                disabled={analyzingImageId === img.id}
                                className="w-full py-1 text-slate-400 hover:text-sky-600 text-[11px] font-medium flex items-center justify-center space-x-1.5 transition-colors cursor-pointer disabled:opacity-50"
                                title="Re-run AI screening"
                              >
                                <RefreshCw
                                  className={`w-3 h-3 ${
                                    analyzingImageId === img.id ? 'animate-spin text-sky-600' : ''
                                  }`}
                                />
                                <span>
                                  {analyzingImageId === img.id ? 'Re-analyzing...' : 'Re-run Analysis'}
                                </span>
                              </button>
                            </div>
                          );
                        }

                        return (
                          <div>
                            <button
                              type="button"
                              onClick={() => handleAnalyzeImage(img)}
                              disabled={analyzingImageId === img.id}
                              className={`w-full py-2.5 px-4 rounded-xl text-xs font-semibold flex items-center justify-center space-x-2 transition-all shadow-xs cursor-pointer ${
                                isPoor
                                  ? 'bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-300'
                                  : 'bg-sky-600 hover:bg-sky-700 text-white active:scale-[0.99]'
                              } disabled:opacity-60 disabled:cursor-not-allowed`}
                            >
                              {analyzingImageId === img.id ? (
                                <>
                                  <div className="w-3.5 h-3.5 border-2 border-white/40 border-t-white rounded-full animate-spin" />
                                  <span>Analyzing...</span>
                                </>
                              ) : (
                                <>
                                  <Sparkles className="w-4 h-4 text-sky-200" />
                                  <span>Analyze with AI</span>
                                </>
                              )}
                            </button>

                            {isPoor && (
                              <p className="text-[11px] text-amber-700 bg-amber-50/80 p-2 rounded-lg border border-amber-200/60 mt-2 text-center">
                                Notice: Image quality is marked as 'poor'.
                              </p>
                            )}
                          </div>
                        );
                      })()}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Milestone 3 Analysis Report Modal */}
      <AnalysisReportModal
        isOpen={isReportModalOpen}
        onClose={() => setIsReportModalOpen(false)}
        patientName={patientName}
        patientAge={patientAge}
        caseId={caseId}
        imageId={selectedModalImage?.id}
        imageUrl={selectedModalImage ? imageUrls[selectedModalImage.id] || '' : ''}
        predictions={
          selectedModalPred?.all_predictions ||
          selectedModalPred?.top_n ||
          selectedModalImage?.predictions ||
          []
        }
        probabilities={selectedModalPred?.probabilities || {}}
        topPrediction={selectedModalPred?.predicted_class}
        confidence={selectedModalPred?.confidence}
        riskLevel={selectedModalPred?.risk_level}
        riskMessage={selectedModalPred?.risk_message}
        gradcamAvailable={!(selectedModalPred?.demo_model || selectedModalImage?.demo_model)}
        analysisDate={selectedModalPred?.created_at}
      />
    </div>
  );
};
