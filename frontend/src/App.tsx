import { FormEvent, useEffect, useMemo, useState } from 'react';

type Page = 'dashboard' | 'cases' | 'upload' | 'dataset' | 'models' | 'case-detail';

type CaseRecord = {
  id: number;
  title: string;
  description: string | null;
  status: string;
  review_status?: string | null;
  doctor_review_notes?: string | null;
  reviewed_by_id?: number | null;
  created_at: string;
  updated_at: string;
  user_id: number;
  images?: CaseImageSummary[];
  predictions?: PredictionSummary[];
};

type PredictionSummary = {
  id: number;
  case_id: number;
  image_id: number | null;
  model_version: string;
  prediction_label: string | null;
  confidence: number | null;
  risk_level: string | null;
  status: string;
  created_at: string;
};

type CaseImageSummary = {
  id: number;
  case_id: number;
  filename: string;
  content_type: string;
  file_size: number;
  width: number | null;
  height: number | null;
  status: string;
  created_at: string;
  analysis?: ImageAnalysisData | null;
};

type DatasetStatistics = {
  dataset_name: string;
  image_type: string;
  image_count: number;
  unique_lesion_count: number;
  class_count: number;
  class_counts: Record<string, number>;
  source: string;
};

type TopPrediction = {
  class_name: string;
  label?: string;
  confidence: number;
};

type ClassificationData = {
  status: string;
  predicted_class?: string;
  confidence?: number;
  confidence_level?: string;
  architecture?: string;
  model_category?: string;
  constituent_models?: string[];
  uncertainty_entropy?: number;
  top_predictions?: TopPrediction[];
  reason?: string;
};

type ExplainabilityData = {
  status: string;
  method?: string;
  heatmap_data_url?: string;
  overlay_data_url?: string;
  peak_attention_coords?: [number, number];
  attention_coverage_percent?: number;
  lesion_overlap_score?: number | null;
  interpretation?: string;
};

type RiskAssessmentData = {
  risk_level: 'High Risk' | 'Moderate Risk' | 'Low Risk' | 'Requires Clinical Review' | string;
  base_risk?: string;
  severity_score: number;
  urgency: string;
  clinical_name: string;
  clinical_category?: string;
  description?: string;
  recommended_action: string;
  risk_factors?: string[];
  confidence_level?: string;
  disclaimer: string;
};

type LesionDetectionData = {
  status: string;
  reason?: string;
  architecture?: string;
  mask_data_url?: string;
  overlay_data_url?: string;
};

type ImageAnalysisData = {
  preprocessing?: {
    pixel_value_range?: [number, number];
    [key: string]: unknown;
  } | null;
  quality_assessment?: {
    quality_category?: string;
    quality_score?: number;
    metrics?: Record<string, number>;
    [key: string]: unknown;
  } | null;
  classification?: ClassificationData | null;
  lesion_detection?: LesionDetectionData | null;
  explainability?: ExplainabilityData | null;
  risk_assessment?: RiskAssessmentData | null;
  analysis_errors?: string[];
  processing_error?: string | null;
};

type UploadedImage = {
  id: number;
  case_id: number;
  filename: string;
  content_type: string;
  file_size: number;
  width: number;
  height: number;
  color_mode: string;
  status: string;
  classification: ClassificationData | null;
  lesion_detection: LesionDetectionData;
  explainability?: ExplainabilityData | null;
  risk_assessment?: RiskAssessmentData | null;
  analysis_errors: string[];
  created_at: string;
};

type ModelBenchmark = {
  id: string;
  name: string;
  family: string;
  parameters_m: number;
  input_resolution: string;
  validation_accuracy: number;
  validation_macro_f1: number;
  test_accuracy: number;
  test_macro_f1: number;
  auc_roc: number;
  inference_latency_ms: number;
  status: string;
  strengths: string;
};

type ModelsComparisonData = {
  selected_default: string;
  classes: string[];
  models: ModelBenchmark[];
  confusion_matrix_summary?: {
    classes: string[];
    notes: string;
  };
};

type ComparisonResponse = {
  baseline: {
    image_id: number;
    filename: string;
    date: string;
    predicted_class: string | null;
    confidence: number;
    risk_level: string;
    severity_score: number;
    architecture?: string;
  };
  followup: {
    image_id: number;
    filename: string;
    date: string;
    predicted_class: string | null;
    confidence: number;
    risk_level: string;
    severity_score: number;
    architecture?: string;
  };
  comparison: {
    same_condition_predicted: boolean | null;
    confidence_delta: number;
    severity_score_delta: number;
    trajectory: string;
    clinical_recommendation: string;
  };
  disclaimer: string;
};

type Notice = { kind: 'success' | 'error' | 'info'; text: string };

class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

const apiBase = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1').replace(/\/$/, '');
const maxUploadBytes = Number(import.meta.env.VITE_MAX_UPLOAD_SIZE_BYTES || 10 * 1024 * 1024);
const acceptedTypes = new Set(['image/jpeg', 'image/png']);

const diagnosisLabels: Record<string, string> = {
  akiec: 'Actinic keratoses / intraepithelial carcinoma',
  bcc: 'Basal cell carcinoma',
  bkl: 'Benign keratosis-like lesion',
  df: 'Dermatofibroma',
  mel: 'Melanoma (Malignant)',
  nv: 'Melanocytic nevus (Mole)',
  vasc: 'Vascular lesion',
};

async function apiRequest<T>(
  path: string,
  options: RequestInit = {},
  token?: string,
): Promise<T> {
  const headers = new Headers(options.headers);
  if (token) headers.set('Authorization', `Bearer ${token}`);
  if (options.body && !(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }

  let response: Response;
  try {
    response = await fetch(`${apiBase}${path}`, { ...options, headers });
  } catch {
    throw new Error(`Cannot reach the DermAI API at ${apiBase}. Check that the backend is running.`);
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status}).`;
    try {
      const body = (await response.json()) as {
        detail?: string | Array<{ msg?: string; loc?: Array<string | number> }>;
      };
      if (typeof body.detail === 'string') {
        detail = body.detail;
      } else if (Array.isArray(body.detail)) {
        detail = body.detail.map((issue) => issue.msg || 'Invalid input').join(' ');
      }
    } catch {
      // Use fallback error message
    }
    if (response.status === 401) {
      throw new ApiError(401, token ? 'Your session expired. Please sign in again.' : detail);
    }
    throw new ApiError(response.status, detail);
  }

  return response.json() as Promise<T>;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatDate(value: string): string {
  return new Date(value).toLocaleString();
}

function getRiskBadgeClass(level?: string | null): string {
  if (!level) return 'risk-badge risk-review';
  const l = level.toLowerCase();
  if (l.includes('high')) return 'risk-badge risk-high';
  if (l.includes('moderate')) return 'risk-badge risk-moderate';
  if (l.includes('low')) return 'risk-badge risk-low';
  return 'risk-badge risk-review';
}

function getRiskCardClass(level?: string | null): string {
  if (!level) return 'risk-card review';
  const l = level.toLowerCase();
  if (l.includes('high')) return 'risk-card high';
  if (l.includes('moderate')) return 'risk-card moderate';
  if (l.includes('low')) return 'risk-card low';
  return 'risk-card review';
}

export default function App() {
  const [page, setPage] = useState<Page>('dashboard');
  const [token, setToken] = useState<string | null>(null);
  const [userRole, setUserRole] = useState<'patient' | 'doctor'>('patient');
  const [userName, setUserName] = useState<string>('');
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const [authRole, setAuthRole] = useState<'patient' | 'doctor'>('patient');
  const [authBusy, setAuthBusy] = useState(false);
  const [authForm, setAuthForm] = useState({ email: '', password: '', fullName: '' });

  // Doctor review state
  const [doctorNotes, setDoctorNotes] = useState<string>('');
  const [reviewDecision, setReviewDecision] = useState<string>('reviewed');
  const [reviewBusy, setReviewBusy] = useState<boolean>(false);

  // Cases state
  const [cases, setCases] = useState<CaseRecord[]>([]);
  const [casesLoading, setCasesLoading] = useState(false);
  const [caseBusy, setCaseBusy] = useState(false);
  const [caseTitle, setCaseTitle] = useState('');
  const [caseDescription, setCaseDescription] = useState('');
  const [selectedCase, setSelectedCase] = useState<CaseRecord | null>(null);
  const [caseDetailLoading, setCaseDetailLoading] = useState(false);

  // Active analysis in case detail
  const [activeAnalysisImage, setActiveAnalysisImage] = useState<CaseImageSummary | null>(null);

  // Dataset & Models
  const [dataset, setDataset] = useState<DatasetStatistics | null>(null);
  const [datasetLoading, setDatasetLoading] = useState(true);
  const [datasetError, setDatasetError] = useState('');
  const [modelsData, setModelsData] = useState<ModelsComparisonData | null>(null);
  const [modelsLoading, setModelsLoading] = useState(false);

  // Upload state
  const [notice, setNotice] = useState<Notice | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [uploadCaseId, setUploadCaseId] = useState('');
  const [uploadBusy, setUploadBusy] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadedImage | null>(null);
  const [fileError, setFileError] = useState('');
  const [selectedArch, setSelectedArch] = useState('mobilenet_v2');

  // Medical image view mode in analysis screen
  const [viewMode, setViewMode] = useState<'all' | 'original' | 'mask' | 'heatmap' | 'overlay'>('all');

  // Longitudinal comparison state
  const [compBaseId, setCompBaseId] = useState<number | null>(null);
  const [compFollId, setCompFollId] = useState<number | null>(null);
  const [comparisonResult, setComparisonResult] = useState<ComparisonResponse | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);

  const maxClassCount = useMemo(
    () => Math.max(1, ...Object.values(dataset?.class_counts ?? {})),
    [dataset],
  );

  useEffect(() => {
    let active = true;
    setDatasetLoading(true);
    apiRequest<DatasetStatistics>('/dataset/stats')
      .then((statistics) => {
        if (active) {
          setDataset(statistics);
          setDatasetError('');
        }
      })
      .catch((error: Error) => {
        if (active) setDatasetError(error.message);
      })
      .finally(() => {
        if (active) setDatasetLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const loadModelsComparison = async () => {
    setModelsLoading(true);
    try {
      const data = await apiRequest<ModelsComparisonData>('/analysis/models');
      setModelsData(data);
    } catch {
      // Ignored if models endpoint unreachable
    } finally {
      setModelsLoading(false);
    }
  };

  useEffect(() => {
    void loadModelsComparison();
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl('');
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const showApiError = (error: unknown, requestToken?: string) => {
    if (error instanceof ApiError && error.status === 401 && (token || requestToken)) {
      setToken(null);
      setCases([]);
      setSelectedCase(null);
      setPage('dashboard');
      setNotice({ kind: 'error', text: error.message });
      return;
    }
    setNotice({
      kind: 'error',
      text: error instanceof Error ? error.message : 'An unexpected error occurred.',
    });
  };

  const loadCases = async (accessToken: string, isDoctor: boolean = userRole === 'doctor') => {
    setCasesLoading(true);
    try {
      const url = isDoctor ? '/cases?doctor_view=true' : '/cases';
      const records = await apiRequest<CaseRecord[]>(url, {}, accessToken);
      setCases(records);
    } catch (error) {
      showApiError(error, accessToken);
    } finally {
      setCasesLoading(false);
    }
  };

  const submitAuth = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setNotice(null);
    setAuthBusy(true);
    try {
      if (authMode === 'register') {
        await apiRequest('/auth/register', {
          method: 'POST',
          body: JSON.stringify({
            email: authForm.email.trim(),
            password: authForm.password,
            full_name: authForm.fullName.trim(),
            role: authRole,
          }),
        });
        setAuthMode('login');
        setAuthForm((current) => ({ ...current, password: '' }));
        setNotice({
          kind: 'success',
          text: `Account created for ${authRole === 'doctor' ? 'Doctor / Dermatologist' : 'Patient'}. Sign in with your new credentials.`,
        });
        return;
      }

      const result = await apiRequest<{ access_token: string; role?: string; full_name?: string }>('/auth/login', {
        method: 'POST',
        body: JSON.stringify({ email: authForm.email.trim(), password: authForm.password }),
      });
      setToken(result.access_token);
      const assignedRole = (result.role?.toLowerCase() === 'doctor') ? 'doctor' : 'patient';
      setUserRole(assignedRole);
      setUserName(result.full_name || authForm.email.trim());
      setAuthForm((current) => ({ ...current, password: '' }));
      setNotice({
        kind: 'success',
        text: `Signed in successfully as ${assignedRole === 'doctor' ? 'Dr. Arjun (Doctor)' : 'Varun (Patient)'}.`,
      });
      setPage('dashboard');
      await loadCases(result.access_token, assignedRole === 'doctor');
    } catch (error) {
      showApiError(error);
    } finally {
      setAuthBusy(false);
    }
  };

  const submitClinicalReview = async (event: FormEvent<HTMLFormElement>, caseId: number) => {
    event.preventDefault();
    if (!token) return;
    setReviewBusy(true);
    setNotice(null);
    try {
      const updatedCase = await apiRequest<CaseRecord>(
        `/cases/${caseId}/review`,
        {
          method: 'POST',
          body: JSON.stringify({
            review_status: reviewDecision,
            doctor_notes: doctorNotes.trim(),
          }),
        },
        token,
      );
      setSelectedCase(updatedCase);
      setNotice({ kind: 'success', text: `Clinical review recorded: ${reviewDecision.replace('_', ' ')}.` });
      await loadCases(token, true);
    } catch (error) {
      showApiError(error);
    } finally {
      setReviewBusy(false);
    }
  };

  const signOut = () => {
    setToken(null);
    setUserRole('patient');
    setUserName('');
    setCases([]);
    setSelectedCase(null);
    setActiveAnalysisImage(null);
    setUploadResult(null);
    setSelectedFile(null);
    setComparisonResult(null);
    setPage('dashboard');
    setNotice({ kind: 'info', text: 'You have been signed out.' });
  };

  const createCase = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!token) {
      setNotice({ kind: 'error', text: 'Sign in before creating a case.' });
      return;
    }
    setCaseBusy(true);
    setNotice(null);
    try {
      const created = await apiRequest<CaseRecord>(
        '/cases',
        {
          method: 'POST',
          body: JSON.stringify({ title: caseTitle.trim(), description: caseDescription.trim() || null }),
        },
        token,
      );
      setCases((current) => [created, ...current]);
      setCaseTitle('');
      setCaseDescription('');
      setNotice({ kind: 'success', text: `Case "${created.title}" was created.` });
    } catch (error) {
      showApiError(error);
    } finally {
      setCaseBusy(false);
    }
  };

  const openCase = async (caseId: number) => {
    if (!token) {
      setNotice({ kind: 'error', text: 'Sign in to view case details.' });
      return;
    }
    setPage('case-detail');
    setSelectedCase(null);
    setActiveAnalysisImage(null);
    setComparisonResult(null);
    setCaseDetailLoading(true);
    setNotice(null);
    try {
      const record = await apiRequest<CaseRecord>(`/cases/${caseId}`, {}, token);
      setSelectedCase(record);
      setDoctorNotes(record.doctor_review_notes || '');
      if (record.review_status) {
        setReviewDecision(record.review_status as any);
      }
      if (record.images && record.images.length > 0) {
        setActiveAnalysisImage(record.images[0]);
      }
    } catch (error) {
      showApiError(error);
    } finally {
      setCaseDetailLoading(false);
    }
  };

  const runSessionComparison = async () => {
    if (!token || !compBaseId || !compFollId) return;
    setCompareLoading(true);
    setComparisonResult(null);
    try {
      const res = await apiRequest<ComparisonResponse>(
        '/analysis/compare',
        {
          method: 'POST',
          body: JSON.stringify({ baseline_image_id: compBaseId, followup_image_id: compFollId }),
        },
        token,
      );
      setComparisonResult(res);
    } catch (error) {
      showApiError(error);
    } finally {
      setCompareLoading(false);
    }
  };

  const downloadReport = async (caseId: number, format: 'pdf' | 'csv' | 'json') => {
    if (!token) {
      setNotice({ kind: 'error', text: 'Sign in to export clinical reports.' });
      return;
    }
    try {
      setNotice({ kind: 'info', text: `Preparing ${format.toUpperCase()} report…` });
      const response = await fetch(`${apiBase}/reports/cases/${caseId}?format=${format}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (!response.ok) {
        throw new Error(`Report export failed with status ${response.status}`);
      }
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `DermAI_Report_Case_${caseId}.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      a.remove();
      setNotice({ kind: 'success', text: `Downloaded ${format.toUpperCase()} report for Case #${caseId}.` });
    } catch (error: any) {
      setNotice({ kind: 'error', text: error.message || 'Report download failed.' });
    }
  };

  const selectImage = (file?: File) => {
    setUploadResult(null);
    setFileError('');
    setNotice(null);
    if (!file) {
      setSelectedFile(null);
      return;
    }
    if (!acceptedTypes.has(file.type)) {
      setSelectedFile(null);
      setFileError('Choose a JPEG or PNG image.');
      return;
    }
    if (file.size === 0) {
      setSelectedFile(null);
      setFileError('The selected file is empty.');
      return;
    }
    if (file.size > maxUploadBytes) {
      setSelectedFile(null);
      setFileError(`The selected file is larger than the ${formatBytes(maxUploadBytes)} upload limit.`);
      return;
    }
    setSelectedFile(file);
  };

  const uploadImage = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!token) {
      setNotice({ kind: 'error', text: 'Sign in before uploading an image.' });
      return;
    }
    if (!uploadCaseId) {
      setNotice({ kind: 'error', text: 'Select one of your cases for this upload.' });
      return;
    }
    if (!selectedFile) {
      setFileError('Select a valid JPEG or PNG image before uploading.');
      return;
    }
    setUploadBusy(true);
    setUploadResult(null);
    setNotice(null);
    try {
      const data = new FormData();
      data.append('file', selectedFile);
      const url = `/upload/${uploadCaseId}?architecture=${encodeURIComponent(selectedArch)}`;
      const uploaded = await apiRequest<UploadedImage>(url, { method: 'POST', body: data }, token);
      setUploadResult(uploaded);
      setNotice({
        kind: 'success',
        text: `${uploaded.filename} analyzed successfully with ${selectedArch.toUpperCase()} architecture.`,
      });
      // Refresh cases list
      await loadCases(token);
    } catch (error) {
      showApiError(error);
    } finally {
      setUploadBusy(false);
    }
  };

  const navItems: { id: Page; label: string }[] = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'cases', label: 'Cases' },
    { id: 'upload', label: 'Image Analysis' },
    { id: 'models', label: 'Models & XAI' },
    { id: 'dataset', label: 'Dataset' },
  ];

  const titleForPage: Record<Page, string> = {
    dashboard: 'Clinical Decision Support Dashboard',
    cases: 'Patient & Case Management',
    upload: 'Medical Image Analysis & Upload',
    models: 'AI Architectures & Model Benchmarks',
    dataset: 'HAM10000 Dataset Distribution',
    'case-detail': selectedCase?.title ?? 'Case Details & History',
  };

  // Render Medical Image Visual Inspection Panel (Section 14)
  const renderVisualInspection = (
    imageSrc: string,
    maskUrl?: string,
    heatmapUrl?: string,
    overlayUrl?: string,
    boundaryOverlayUrl?: string,
  ) => {
    return (
      <div>
        <div className="view-mode-tabs" role="tablist" aria-label="Image inspection views">
          <button
            type="button"
            className={`view-mode-tab ${viewMode === 'all' ? 'active' : ''}`}
            onClick={() => setViewMode('all')}
          >
            Multi-View Comparison
          </button>
          <button
            type="button"
            className={`view-mode-tab ${viewMode === 'original' ? 'active' : ''}`}
            onClick={() => setViewMode('original')}
          >
            Original
          </button>
          {maskUrl && (
            <button
              type="button"
              className={`view-mode-tab ${viewMode === 'mask' ? 'active' : ''}`}
              onClick={() => setViewMode('mask')}
            >
              U-Net Lesion Mask
            </button>
          )}
          {heatmapUrl && (
            <button
              type="button"
              className={`view-mode-tab ${viewMode === 'heatmap' ? 'active' : ''}`}
              onClick={() => setViewMode('heatmap')}
            >
              Grad-CAM Heatmap
            </button>
          )}
          {overlayUrl && (
            <button
              type="button"
              className={`view-mode-tab ${viewMode === 'overlay' ? 'active' : ''}`}
              onClick={() => setViewMode('overlay')}
            >
              AI Attention Overlay
            </button>
          )}
        </div>

        <div className="analysis-display-grid">
          {(viewMode === 'all' || viewMode === 'original') && (
            <figure className="analysis-figure">
              <img src={imageSrc} alt="Original dermoscopic lesion" />
              <figcaption>
                <span>Original Skin Image</span>
                <span className="tag">Dermoscopic</span>
              </figcaption>
            </figure>
          )}

          {(viewMode === 'all' || viewMode === 'mask') && (maskUrl || boundaryOverlayUrl) && (
            <figure className="analysis-figure">
              <img src={maskUrl || boundaryOverlayUrl} alt="U-Net predicted segmentation mask" />
              <figcaption>
                <span>Lesion Region / Mask</span>
                <span className="tag">ISBI U-Net</span>
              </figcaption>
            </figure>
          )}

          {(viewMode === 'all' || viewMode === 'heatmap') && heatmapUrl && (
            <figure className="analysis-figure">
              <img src={heatmapUrl} alt="Grad-CAM activation heatmap" />
              <figcaption>
                <span>AI Attention Map</span>
                <span className="tag">Grad-CAM (JET)</span>
              </figcaption>
            </figure>
          )}

          {(viewMode === 'all' || viewMode === 'overlay') && overlayUrl && (
            <figure className="analysis-figure">
              <img src={overlayUrl} alt="Attention blended on original image" />
              <figcaption>
                <span>XAI Attention Overlay</span>
                <span className="tag">Feature Focus</span>
              </figcaption>
            </figure>
          )}
        </div>
      </div>
    );
  };

  const renderDataset = () => (
    <section className="panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Dataset Ground Truth</p>
          <h2>HAM10000 Dataset</h2>
          <p className="muted">Standardized benchmark of 10,015 dermatoscopic images across 7 diagnostic categories.</p>
        </div>
        <span className="tag">Dermoscopic Dataset</span>
      </div>
      {datasetLoading && <p className="state-message" role="status">Loading dataset statistics…</p>}
      {datasetError && <p className="state-message error" role="alert">{datasetError}</p>}
      {dataset && (
        <>
          <p className="dataset-note">{dataset.image_type}</p>
          <div className="metric-grid">
            <Metric label="Images in metadata" value={dataset.image_count.toLocaleString()} />
            <Metric label="Unique lesion IDs" value={dataset.unique_lesion_count.toLocaleString()} />
            <Metric label="Metadata classes" value={String(dataset.class_count)} />
          </div>
          <h3 className="subheading">Class distribution</h3>
          <div className="chart" aria-label="HAM10000 image counts by diagnosis class">
            {Object.entries(dataset.class_counts).map(([label, count]) => (
              <div className="chart-row" key={label}>
                <span className="chart-label">{label}</span>
                <div
                  className="chart-track"
                  role="img"
                  aria-label={`${label}: ${count.toLocaleString()} images`}
                >
                  <div className="chart-fill" style={{ width: `${(count / maxClassCount) * 100}%` }} />
                </div>
                <span className="chart-count">{count.toLocaleString()}</span>
              </div>
            ))}
          </div>
          <p className="source-note">Source: {dataset.source}. Grouped by lesion_id to ensure zero data leakage.</p>
        </>
      )}
    </section>
  );

  const renderModels = () => (
    <div className="panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Deep Learning Benchmarking & Multi-Model Evaluation</p>
          <h2>Multi-Model Architectures & Comparison</h2>
          <p className="muted">
            Evaluation across CNNs (MobileNetV2, EfficientNet-B0, ResNet-50) and Transformer-based architectures (ViT-B/16), plus Multi-Model Ensemble.
          </p>
        </div>
        <span className="tag">Engine Operational & Validated</span>
      </div>

      {modelsLoading && <p className="state-message">Loading model benchmarks…</p>}
      {modelsData && (
        <>
          <div className="models-table-wrapper">
            <table className="models-table">
              <thead>
                <tr>
                  <th>Architecture</th>
                  <th>Family</th>
                  <th>Params</th>
                  <th>Val Accuracy</th>
                  <th>Val Macro F1</th>
                  <th>Test Accuracy</th>
                  <th>Test Macro F1</th>
                  <th>AUC-ROC</th>
                  <th>Inference</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {modelsData.models.map((m) => (
                  <tr key={m.id} style={{ background: m.id === modelsData.selected_default ? '#f0faf8' : undefined }}>
                    <td>
                      <strong>{m.name}</strong>
                      {m.id === modelsData.selected_default && <span style={{ marginLeft: 6, fontSize: 10, color: '#087f71' }}>★ Default</span>}
                    </td>
                    <td>{m.family}</td>
                    <td>{m.parameters_m}M</td>
                    <td>{(m.validation_accuracy * 100).toFixed(1)}%</td>
                    <td>{(m.validation_macro_f1 * 100).toFixed(1)}%</td>
                    <td>{(m.test_accuracy * 100).toFixed(1)}%</td>
                    <td>{(m.test_macro_f1 * 100).toFixed(1)}%</td>
                    <td>{m.auc_roc.toFixed(3)}</td>
                    <td>{m.inference_latency_ms} ms</td>
                    <td>
                      <span className="tag" style={{ background: '#ecfdf5', color: '#059669', fontSize: 10 }}>
                        {m.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ marginTop: 24 }}>
            <h3 className="subheading">Architectural Insights & Strengths</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 14 }}>
              {modelsData.models.map((m) => (
                <div key={m.id} style={{ border: '1px solid #e1ecec', borderRadius: 10, padding: 14, background: '#fcfdfd' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                    <strong>{m.name}</strong>
                    <span className="tag" style={{ fontSize: 10 }}>{m.family.split(' ')[0]}</span>
                  </div>
                  <p style={{ margin: 0, fontSize: 12, color: '#556c75', lineHeight: 1.5 }}>{m.strengths}</p>
                </div>
              ))}
            </div>
          </div>

          <div style={{ marginTop: 20, padding: 12, background: '#f8fafb', borderRadius: 10, border: '1px solid #e2ecec', fontSize: 12, color: '#5a6e76' }}>
            <strong>Evaluation Protocol Note:</strong> All models evaluated on strict StratifiedGroupKFold split (60% Train, 20% Val, 20% Held-Out Test).
            Neither lesion ID nor duplicate hashes cross splits.
          </div>
        </>
      )}
    </div>
  );

  const renderCases = () => (
    <div className="content-grid">
      <section className="panel">
        <p className="eyebrow">Case Management</p>
        <h2>Create a Case</h2>
        {!token ? (
          <p className="state-message">Sign in to create and manage cases.</p>
        ) : (
          <form onSubmit={createCase} className="form-stack">
            <label htmlFor="case-title">Case Title / Reference</label>
            <input
              id="case-title"
              value={caseTitle}
              onChange={(event) => setCaseTitle(event.target.value)}
              placeholder="e.g., Forearm pigmented lesion evaluation"
              minLength={2}
              maxLength={255}
              required
              autoComplete="off"
            />
            <label htmlFor="case-description">
              Clinical Notes <span className="muted">(De-identified research records only)</span>
            </label>
            <textarea
              id="case-description"
              value={caseDescription}
              onChange={(event) => setCaseDescription(event.target.value)}
              placeholder="Brief observation, patient age/gender de-identified, anatomical location..."
              rows={4}
            />
            <button className="button primary" type="submit" disabled={caseBusy}>
              {caseBusy ? 'Creating…' : 'Create Case'}
            </button>
          </form>
        )}
      </section>
      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">{userRole === 'doctor' ? 'Clinical Triage & Review' : 'Your Records'}</p>
            <h2>{userRole === 'doctor' ? 'Patient Cases for Clinical Review' : 'Cases & History'}</h2>
          </div>
          {token && (
            <button className="button secondary compact" type="button" onClick={() => void loadCases(token, userRole === 'doctor')} disabled={casesLoading}>
              Refresh
            </button>
          )}
        </div>
        {token && userRole === 'doctor' && (
          <div style={{ marginBottom: 14, padding: '8px 12px', background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: 8, fontSize: 12, color: '#1e40af' }}>
            👨‍⚕️ <strong>Doctor RBAC:</strong> You have clinical review permissions to examine patient cases, inspect Grad-CAM activations, evaluate ISBI U-Net masks, and document diagnosis reviews.
          </div>
        )}
        {!token && <p className="state-message">Sign in to see your cases.</p>}
        {token && casesLoading && <p className="state-message" role="status">Loading cases…</p>}
        {token && !casesLoading && cases.length === 0 && (
          <p className="state-message">{userRole === 'doctor' ? 'No cases awaiting review.' : 'No cases yet. Create a case to begin.'}</p>
        )}
        {token && !casesLoading && cases.length > 0 && (
          <ul className="case-list">
            {cases.map((record) => (
              <li key={record.id} className="case-list-item">
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                    <h3 style={{ margin: 0 }}>{record.title}</h3>
                    {record.review_status && (
                      <span
                        className="tag"
                        style={{
                          fontSize: 10,
                          padding: '2px 8px',
                          textTransform: 'uppercase',
                          fontWeight: 700,
                          background: record.review_status === 'reviewed' ? '#dcfce7' : record.review_status === 'biopsy_recommended' ? '#fee2e2' : '#fef3c7',
                          color: record.review_status === 'reviewed' ? '#15803d' : record.review_status === 'biopsy_recommended' ? '#b91c1c' : '#b45309',
                        }}
                      >
                        {record.review_status.replace(/_/g, ' ')}
                      </span>
                    )}
                  </div>
                  <p>{record.description || 'No notes provided.'}</p>
                  <span className="case-date">Created {formatDate(record.created_at)}</span>
                </div>
                <button className="button secondary" type="button" onClick={() => void openCase(record.id)}>
                  {userRole === 'doctor' ? 'Review Case →' : 'View Case & History →'}
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );

  const renderUpload = () => (
    <section className="panel narrow-panel">
      <p className="eyebrow">Image Analysis Engine</p>
      <h2>Upload & Analyze Skin Lesion</h2>
      <p className="muted">
        Upload a dermoscopic or mobile skin lesion photograph. The pipeline performs automated quality validation, disease classification, ISBI U-Net lesion segmentation, Grad-CAM explainability, and severity risk profiling.
      </p>

      {!token ? (
        <p className="state-message">Sign in to upload and analyze images.</p>
      ) : cases.length === 0 ? (
        <div className="state-message">
          No cases available. <button className="text-button" type="button" onClick={() => setPage('cases')}>Create a case first</button>.
        </div>
      ) : (
        <form className="form-stack" onSubmit={(event) => void uploadImage(event)}>
          <label htmlFor="upload-case">Associate with Case</label>
          <select id="upload-case" value={uploadCaseId} onChange={(event) => setUploadCaseId(event.target.value)} required>
            <option value="">Select a case</option>
            {cases.map((record) => (
              <option key={record.id} value={record.id}>
                #{record.id} — {record.title}
              </option>
            ))}
          </select>

          <label htmlFor="model-select">AI Architecture Selection</label>
          <select id="model-select" value={selectedArch} onChange={(event) => setSelectedArch(event.target.value)}>
            <option value="mobilenet_v2">MobileNetV2 (Default Lightweight CNN)</option>
            <option value="resnet50">ResNet-50 (Advanced Deep CNN Architecture)</option>
            <option value="vit_b_16">Vision Transformer ViT-B/16 (Self-Attention Transformer)</option>
            <option value="efficientnet_b0">EfficientNet-B0 (Compound Scaling CNN)</option>
            <option value="ensemble">Multi-Model Ensemble (Soft-Voting CNN + ViT)</option>
          </select>

          <label htmlFor="image-file">JPEG or PNG Image</label>
          <input
            id="image-file"
            type="file"
            accept="image/jpeg,image/png"
            onChange={(event) => selectImage(event.currentTarget.files?.[0])}
          />
          {fileError && <p className="inline-error" role="alert">{fileError}</p>}

          {selectedFile && (
            <div className="preview-card">
              <img src={previewUrl} alt={`Preview of ${selectedFile.name}`} className="image-preview" />
              <div className="preview-meta">
                <strong>{selectedFile.name}</strong>
                <span>{formatBytes(selectedFile.size)}</span>
                <span>{uploadBusy ? 'Analyzing with AI…' : uploadResult ? 'Analyzed' : 'Ready for analysis'}</span>
              </div>
              <button
                type="button"
                className="button secondary compact"
                onClick={() => {
                  setSelectedFile(null);
                  setUploadResult(null);
                  setFileError('');
                }}
                disabled={uploadBusy}
              >
                Cancel
              </button>
            </div>
          )}

          <button className="button primary" type="submit" disabled={uploadBusy || !selectedFile}>
            {uploadBusy ? 'Running Deep Learning Inference & XAI…' : 'Run Full AI Analysis'}
          </button>

          {uploadResult && (
            <div className="upload-receipt" role="status">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <strong>Server Analysis Confirmed · Case #{uploadResult.case_id}</strong>
                <span className="tag" style={{ background: '#dcfce7', color: '#166534', fontSize: 10 }}>
                  Model: {uploadResult.classification?.architecture || selectedArch}
                </span>
              </div>

              {/* Severity & Risk Assessment Card */}
              {uploadResult.risk_assessment && (
                <div className={getRiskCardClass(uploadResult.risk_assessment.risk_level)}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <span className={getRiskBadgeClass(uploadResult.risk_assessment.risk_level)}>
                      ● {uploadResult.risk_assessment.risk_level}
                    </span>
                    <span style={{ fontSize: 12, fontWeight: 700, color: '#374151' }}>
                      Severity Index: {uploadResult.risk_assessment.severity_score}/100 ({uploadResult.risk_assessment.urgency} Urgency)
                    </span>
                  </div>
                  <h3 style={{ margin: '4px 0', fontSize: 16 }}>{uploadResult.risk_assessment.clinical_name}</h3>
                  <p style={{ margin: '4px 0 10px', fontSize: 12, color: '#4b5563', lineHeight: 1.5 }}>
                    {uploadResult.risk_assessment.description}
                  </p>
                  <div style={{ background: '#ffffff', padding: 10, borderRadius: 8, border: '1px solid rgba(0,0,0,0.06)' }}>
                    <strong style={{ display: 'block', fontSize: 11, textTransform: 'uppercase', color: '#6b7280', marginBottom: 3 }}>
                      Recommended Action:
                    </strong>
                    <span style={{ fontSize: 12, color: '#1f2937', fontWeight: 600 }}>
                      {uploadResult.risk_assessment.recommended_action}
                    </span>
                  </div>
                </div>
              )}

              {/* Classification Predictions & Confidence */}
              <div className="analysis-result">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <strong>Disease Classification</strong>
                  <span style={{ fontSize: 11, color: '#6b7280' }}>
                    Entropy: {uploadResult.classification?.uncertainty_entropy ?? 'N/A'} bits
                  </span>
                </div>

                {uploadResult.classification?.status === 'available' ? (
                  <>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 4 }}>
                      <span className="primary-prediction" style={{ fontSize: 15 }}>
                        {diagnosisLabels[uploadResult.classification.predicted_class || ''] || uploadResult.classification.predicted_class}
                      </span>
                      <span className="tag" style={{ background: '#e0f2fe', color: '#0369a1' }}>
                        {((uploadResult.classification.confidence ?? 0) * 100).toFixed(1)}% Confidence
                      </span>
                      <span className="tag" style={{ background: '#f3f4f6', color: '#374151' }}>
                        {uploadResult.classification.confidence_level}
                      </span>
                    </div>

                    {uploadResult.classification.top_predictions && (
                      <div className="prob-bar-container">
                        <strong style={{ fontSize: 11, color: '#6b7280', marginTop: 8 }}>Class Probability Distribution:</strong>
                        {uploadResult.classification.top_predictions.map((pred, i) => (
                          <div key={pred.class_name} className="prob-bar-row">
                            <span style={{ fontWeight: i === 0 ? 700 : 500, color: i === 0 ? '#0f766e' : '#4b5563' }}>
                              {pred.label || diagnosisLabels[pred.class_name] || pred.class_name}
                            </span>
                            <div className="prob-bar-track">
                              <div
                                className={`prob-bar-fill ${i === 0 ? 'highlight' : ''}`}
                                style={{ width: `${pred.confidence * 100}%` }}
                              />
                            </div>
                            <span style={{ textAlign: 'right', fontWeight: 600 }}>
                              {(pred.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                ) : (
                  <span>{uploadResult.classification?.reason || 'No trained model available.'}</span>
                )}
              </div>

              {/* Explainable AI Visualizations (Section 14 & Milestone 3) */}
              <div className="analysis-result">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <strong>Explainable AI (XAI) & Lesion Region</strong>
                  <span className="tag" style={{ fontSize: 10 }}>
                    {uploadResult.explainability?.method || 'Grad-CAM'}
                  </span>
                </div>

                {uploadResult.explainability?.interpretation && (
                  <p style={{ margin: '6px 0', fontSize: 12, color: '#4b5563' }}>
                    {uploadResult.explainability.interpretation}
                  </p>
                )}

                {renderVisualInspection(
                  previewUrl,
                  uploadResult.lesion_detection?.mask_data_url,
                  uploadResult.explainability?.heatmap_data_url,
                  uploadResult.explainability?.overlay_data_url,
                  uploadResult.lesion_detection?.overlay_data_url,
                )}
              </div>

              <div style={{ padding: 8, background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 11, color: '#64748b' }}>
                <strong>Disclaimer:</strong> {uploadResult.risk_assessment?.disclaimer || 'DermAI provides decision support and does not replace medical diagnosis.'}
              </div>

              <div style={{ marginTop: 14, padding: '12px 14px', background: '#f0fdfa', border: '1px solid #99f6e4', borderRadius: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
                <div>
                  <strong style={{ fontSize: 13, color: '#0f766e', display: 'block' }}>Clinical Analysis Report Ready</strong>
                  <span style={{ fontSize: 11, color: '#115e59' }}>Export verified diagnostic report with predictions, U-Net mask, and Grad-CAM data.</span>
                </div>
                <div style={{ display: 'flex', gap: 6 }}>
                  <button
                    className="button secondary compact"
                    type="button"
                    onClick={() => void downloadReport(uploadResult.case_id, 'pdf')}
                  >
                    📄 PDF Report
                  </button>
                  <button
                    className="button secondary compact"
                    type="button"
                    onClick={() => void downloadReport(uploadResult.case_id, 'csv')}
                  >
                    📊 CSV Data
                  </button>
                  <button
                    className="button secondary compact"
                    type="button"
                    onClick={() => void downloadReport(uploadResult.case_id, 'json')}
                  >
                    💾 JSON
                  </button>
                </div>
              </div>
            </div>
          )}
        </form>
      )}
    </section>
  );

  const renderCaseDetail = () => {
    if (caseDetailLoading) return <p className="state-message" role="status">Loading case details…</p>;
    if (!selectedCase) return <p className="state-message">Case details are unavailable.</p>;

    const caseImages = selectedCase.images ?? [];
    const casePredictions = selectedCase.predictions ?? [];

    const activeAnalysis = activeAnalysisImage?.analysis;
    const activeClf = activeAnalysis?.classification;
    const activeRisk = activeAnalysis?.risk_assessment;
    const activeExpl = activeAnalysis?.explainability;
    const activeSeg = activeAnalysis?.lesion_detection;

    return (
      <section className="panel">
        <button className="text-button back-button" type="button" onClick={() => setPage('cases')}>
          ← Back to Cases List
        </button>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12 }}>
          <div>
            <p className="eyebrow">Case #{selectedCase.id}</p>
            <h2>{selectedCase.title}</h2>
            <p className="detail-copy">{selectedCase.description || 'No clinical notes provided.'}</p>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <button
              className="button secondary compact"
              type="button"
              onClick={() => void downloadReport(selectedCase.id, 'pdf')}
              title="Download formatted clinical report (PDF)"
            >
              📄 Export PDF
            </button>
            <button
              className="button secondary compact"
              type="button"
              onClick={() => void downloadReport(selectedCase.id, 'csv')}
              title="Export structured tabular analysis data (CSV)"
            >
              📊 Export CSV
            </button>
            <button
              className="button secondary compact"
              type="button"
              onClick={() => void downloadReport(selectedCase.id, 'json')}
              title="Export complete case and prediction payload (JSON)"
            >
              💾 Export JSON
            </button>
            <button
              className="button primary"
              type="button"
              onClick={() => {
                setUploadCaseId(String(selectedCase.id));
                setPage('upload');
              }}
            >
              + Upload Image to Case
            </button>
          </div>
        </div>

        <dl className="detail-list">
          <div>
            <dt>Status</dt>
            <dd>{selectedCase.status}</dd>
          </div>
          <div>
            <dt>Total Uploaded Images</dt>
            <dd>{caseImages.length} image{caseImages.length === 1 ? '' : 's'}</dd>
          </div>
          <div>
            <dt>Prediction Records</dt>
            <dd>{casePredictions.length} record{casePredictions.length === 1 ? '' : 's'}</dd>
          </div>
        </dl>

        {/* Doctor Clinical Review & Triage Section (Doctor RBAC) */}
        <div style={{ margin: '20px 0', padding: 18, background: '#f8fafc', border: '1px solid #cbd5e1', borderRadius: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <div>
              <p className="eyebrow" style={{ margin: 0 }}>Clinical Review & Triage Status</p>
              <h3 style={{ margin: '4px 0' }}>Dermatologist Assessment</h3>
            </div>
            <span
              style={{
                padding: '4px 12px',
                borderRadius: '16px',
                fontSize: '12px',
                fontWeight: 700,
                textTransform: 'uppercase',
                background: selectedCase.review_status === 'reviewed' ? '#dcfce7' : selectedCase.review_status === 'biopsy_recommended' ? '#fee2e2' : '#fef3c7',
                color: selectedCase.review_status === 'reviewed' ? '#15803d' : selectedCase.review_status === 'biopsy_recommended' ? '#b91c1c' : '#b45309',
              }}
            >
              {selectedCase.review_status ? selectedCase.review_status.replace(/_/g, ' ') : 'Pending Review'}
            </span>
          </div>

          {selectedCase.doctor_review_notes ? (
            <div style={{ padding: 12, background: '#ffffff', borderRadius: 8, border: '1px solid #e2e8f0', marginBottom: 12 }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', marginBottom: 4 }}>
                Specialist Notes ({selectedCase.reviewed_by_id ? `Reviewer ID #${selectedCase.reviewed_by_id}` : 'Clinician'}):
              </div>
              <p style={{ margin: 0, fontSize: 13, color: '#1e293b', whiteSpace: 'pre-wrap' }}>
                {selectedCase.doctor_review_notes}
              </p>
            </div>
          ) : (
            <p style={{ fontSize: 12, color: '#64748b', margin: '4px 0 12px' }}>
              No specialist review notes have been documented for this case yet.
            </p>
          )}

          {userRole === 'doctor' && (
            <form onSubmit={(e) => void submitClinicalReview(e, selectedCase.id)} style={{ marginTop: 12, borderTop: '1px dashed #cbd5e1', paddingTop: 12 }}>
              <p style={{ fontSize: 12, fontWeight: 700, color: '#0369a1', margin: '0 0 8px' }}>
                👨‍⚕️ Clinician Review Form (Doctor RBAC)
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) auto', gap: 12, alignItems: 'flex-start' }}>
                <div>
                  <label htmlFor="review-decision" style={{ display: 'block', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>
                    Clinical Action Decision
                  </label>
                  <select
                    id="review-decision"
                    value={reviewDecision}
                    onChange={(e) => setReviewDecision(e.target.value as any)}
                    style={{ width: '100%', padding: '8px 10px', borderRadius: 8, border: '1px solid #cbd5e1', marginBottom: 10 }}
                  >
                    <option value="reviewed">Reviewed & Routine Monitoring</option>
                    <option value="action_required">Action Required / In-Clinic Visit</option>
                    <option value="biopsy_recommended">Biopsy / Histopathology Recommended</option>
                    <option value="second_opinion_requested">Second Opinion Requested</option>
                    <option value="dismissed">Dismissed / Benign Lesion</option>
                  </select>

                  <label htmlFor="review-notes" style={{ display: 'block', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>
                    Dermatologist Assessment Notes & Instructions
                  </label>
                  <textarea
                    id="review-notes"
                    rows={3}
                    value={doctorNotes}
                    onChange={(e) => setDoctorNotes(e.target.value)}
                    placeholder="Enter clinical assessment, ABCD observation notes, follow-up timeline, or referral recommendations..."
                    style={{ width: '100%', padding: 8, borderRadius: 8, border: '1px solid #cbd5e1' }}
                  />
                </div>
                <div style={{ alignSelf: 'flex-end' }}>
                  <button type="submit" className="button primary" disabled={reviewBusy}>
                    {reviewBusy ? 'Saving…' : 'Record Clinical Review'}
                  </button>
                </div>
              </div>
            </form>
          )}
        </div>

        {/* Longitudinal Session Comparison Tool (Section 16) */}
        {caseImages.length >= 2 && (
          <div style={{ margin: '24px 0', padding: 18, background: '#f8fbfb', border: '1px solid #d8e5e6', borderRadius: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <p className="eyebrow" style={{ margin: 0 }}>Multi-Session Tracking (Section 16)</p>
                <h3 style={{ margin: '4px 0' }}>Longitudinal Comparison Tool</h3>
                <p className="muted" style={{ margin: 0 }}>
                  Compare a baseline image session with a follow-up image session to detect lesion evolution.
                </p>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr) auto', gap: 12, marginTop: 12, alignItems: 'flex-end' }}>
              <div>
                <label style={{ display: 'block', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>Baseline Session</label>
                <select
                  value={compBaseId ?? ''}
                  onChange={(e) => setCompBaseId(e.target.value ? Number(e.target.value) : null)}
                  style={{ width: '100%', padding: 8, borderRadius: 8, border: '1px solid #ccdcdc' }}
                >
                  <option value="">Select baseline image...</option>
                  {caseImages.map((img) => (
                    <option key={img.id} value={img.id}>
                      #{img.id} - {img.filename} ({formatDate(img.created_at)})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>Follow-up Session</label>
                <select
                  value={compFollId ?? ''}
                  onChange={(e) => setCompFollId(e.target.value ? Number(e.target.value) : null)}
                  style={{ width: '100%', padding: 8, borderRadius: 8, border: '1px solid #ccdcdc' }}
                >
                  <option value="">Select follow-up image...</option>
                  {caseImages.map((img) => (
                    <option key={img.id} value={img.id}>
                      #{img.id} - {img.filename} ({formatDate(img.created_at)})
                    </option>
                  ))}
                </select>
              </div>

              <button
                type="button"
                className="button primary"
                onClick={() => void runSessionComparison()}
                disabled={compareLoading || !compBaseId || !compFollId || compBaseId === compFollId}
              >
                {compareLoading ? 'Comparing…' : 'Compare Sessions'}
              </button>
            </div>

            {comparisonResult && (
              <div className="compare-box">
                <div className="compare-column">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <strong>Baseline: {comparisonResult.baseline.filename}</strong>
                    <span className={getRiskBadgeClass(comparisonResult.baseline.risk_level)}>
                      {comparisonResult.baseline.risk_level}
                    </span>
                  </div>
                  <p style={{ fontSize: 11, color: '#6b7280', margin: '4px 0' }}>
                    Date: {formatDate(comparisonResult.baseline.date)}
                  </p>
                  <p style={{ margin: '8px 0', fontSize: 13, fontWeight: 700 }}>
                    Condition: {diagnosisLabels[comparisonResult.baseline.predicted_class || ''] || comparisonResult.baseline.predicted_class}
                  </p>
                  <p style={{ margin: '4px 0', fontSize: 12 }}>
                    Confidence: {(comparisonResult.baseline.confidence * 100).toFixed(1)}% | Severity Index: {comparisonResult.baseline.severity_score}/100
                  </p>
                </div>

                <div className="compare-column">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <strong>Follow-up: {comparisonResult.followup.filename}</strong>
                    <span className={getRiskBadgeClass(comparisonResult.followup.risk_level)}>
                      {comparisonResult.followup.risk_level}
                    </span>
                  </div>
                  <p style={{ fontSize: 11, color: '#6b7280', margin: '4px 0' }}>
                    Date: {formatDate(comparisonResult.followup.date)}
                  </p>
                  <p style={{ margin: '8px 0', fontSize: 13, fontWeight: 700 }}>
                    Condition: {diagnosisLabels[comparisonResult.followup.predicted_class || ''] || comparisonResult.followup.predicted_class}
                  </p>
                  <p style={{ margin: '4px 0', fontSize: 12 }}>
                    Confidence: {(comparisonResult.followup.confidence * 100).toFixed(1)}% | Severity Index: {comparisonResult.followup.severity_score}/100
                  </p>
                </div>

                <div style={{ gridColumn: '1 / -1', background: '#ffffff', padding: 12, borderRadius: 8, border: '1px solid #d5e4e4' }}>
                  <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
                    <span style={{ fontWeight: 700, color: '#0d8f7e' }}>
                      Trajectory: {comparisonResult.comparison.trajectory}
                    </span>
                    <span style={{ fontSize: 12, color: '#4b5563' }}>
                      Confidence Shift: {(comparisonResult.comparison.confidence_delta * 100).toFixed(1)}%
                    </span>
                    <span style={{ fontSize: 12, color: '#4b5563' }}>
                      Severity Shift: {comparisonResult.comparison.severity_score_delta > 0 ? '+' : ''}{comparisonResult.comparison.severity_score_delta} pts
                    </span>
                  </div>
                  <p style={{ margin: '6px 0 0', fontSize: 12, color: '#374151' }}>
                    <strong>Clinical Follow-up Recommendation:</strong> {comparisonResult.comparison.clinical_recommendation}
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Gallery of Uploaded Images in this Case */}
        <div>
          <h3 className="subheading">Case Images & Prediction History</h3>
          {caseImages.length === 0 ? (
            <p className="state-message">No images uploaded for this case yet.</p>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 14 }}>
              {caseImages.map((img) => {
                const imgClf = img.analysis?.classification;
                const imgRisk = img.analysis?.risk_assessment;
                const isSelected = activeAnalysisImage?.id === img.id;
                const imgUrl = `${apiBase}/upload/images/${img.id}/file`;

                return (
                  <div
                    key={img.id}
                    onClick={() => setActiveAnalysisImage(img)}
                    style={{
                      cursor: 'pointer',
                      border: isSelected ? '2px solid #0d8f7e' : '1px solid #e1ecec',
                      borderRadius: 12,
                      padding: 10,
                      background: isSelected ? '#f2fbf9' : '#fff',
                      transition: 'all 120ms ease',
                    }}
                  >
                    <div style={{ width: '100%', height: 130, background: '#111', borderRadius: 8, overflow: 'hidden', marginBottom: 8 }}>
                      <img
                        src={token ? `${imgUrl}?token=${token}` : imgUrl}
                        alt={img.filename}
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                        onError={(e) => {
                          // Fallback placeholder if token url requires bearer
                          (e.target as HTMLImageElement).src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><rect width="100" height="100" fill="%2322333b"/><text x="50" y="55" fill="%238899a6" font-size="12" text-anchor="middle">Skin Image</text></svg>';
                        }}
                      />
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <strong style={{ fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {img.filename}
                      </strong>
                    </div>
                    {imgRisk && (
                      <div style={{ marginTop: 4 }}>
                        <span className={getRiskBadgeClass(imgRisk.risk_level)} style={{ fontSize: 10, padding: '2px 8px' }}>
                          {imgRisk.risk_level}
                        </span>
                      </div>
                    )}
                    {imgClf && imgClf.predicted_class && (
                      <p style={{ margin: '4px 0 0', fontSize: 11, color: '#374151', fontWeight: 600 }}>
                        {diagnosisLabels[imgClf.predicted_class] || imgClf.predicted_class}
                      </p>
                    )}
                    <span style={{ fontSize: 10, color: '#9ca3af' }}>{formatDate(img.created_at)}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Selected Image Detailed Medical Analysis Screen (Section 14) */}
        {activeAnalysisImage && activeAnalysis && (
          <div style={{ marginTop: 28, borderTop: '2px solid #e5eded', paddingTop: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <p className="eyebrow" style={{ margin: 0 }}>Medical Image Analysis Screen (Section 14)</p>
                <h3 style={{ margin: '4px 0' }}>Detailed Analysis: {activeAnalysisImage.filename}</h3>
                <span style={{ fontSize: 11, color: '#6b7280' }}>
                  Analyzed with {activeClf?.architecture || 'Deep Learning Model'} on {formatDate(activeAnalysisImage.created_at)}
                </span>
              </div>
              {activeRisk && (
                <span className={getRiskBadgeClass(activeRisk.risk_level)}>
                  ● {activeRisk.risk_level}
                </span>
              )}
            </div>

            {/* Risk Card */}
            {activeRisk && (
              <div className={getRiskCardClass(activeRisk.risk_level)}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                  <strong style={{ fontSize: 14 }}>{activeRisk.clinical_name}</strong>
                  <span style={{ fontSize: 12, fontWeight: 700 }}>
                    Severity: {activeRisk.severity_score}/100 · {activeRisk.urgency} Action
                  </span>
                </div>
                <p style={{ margin: '4px 0 10px', fontSize: 12, color: '#4b5563', lineHeight: 1.5 }}>
                  {activeRisk.description}
                </p>
                <div style={{ background: '#ffffff', padding: 10, borderRadius: 8, border: '1px solid rgba(0,0,0,0.06)' }}>
                  <strong style={{ display: 'block', fontSize: 11, textTransform: 'uppercase', color: '#6b7280' }}>
                    Clinical Guidance:
                  </strong>
                  <span style={{ fontSize: 12, fontWeight: 600, color: '#1f2937' }}>
                    {activeRisk.recommended_action}
                  </span>
                </div>
              </div>
            )}

            {/* Probability Breakdown */}
            {activeClf && activeClf.top_predictions && (
              <div className="analysis-result" style={{ margin: '14px 0' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <strong>Prediction Confidence & Top Classes</strong>
                  <span className="tag">{((activeClf.confidence ?? 0) * 100).toFixed(1)}% {activeClf.confidence_level}</span>
                </div>
                <div className="prob-bar-container">
                  {activeClf.top_predictions.map((pred, i) => (
                    <div key={pred.class_name} className="prob-bar-row">
                      <span style={{ fontWeight: i === 0 ? 700 : 500, color: i === 0 ? '#0f766e' : '#4b5563' }}>
                        {pred.label || diagnosisLabels[pred.class_name] || pred.class_name}
                      </span>
                      <div className="prob-bar-track">
                        <div
                          className={`prob-bar-fill ${i === 0 ? 'highlight' : ''}`}
                          style={{ width: `${pred.confidence * 100}%` }}
                        />
                      </div>
                      <span style={{ textAlign: 'right', fontWeight: 600 }}>
                        {(pred.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Visual Inspection with Grad-CAM and ISBI Mask */}
            <div className="analysis-result" style={{ margin: '14px 0' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <strong>Explainable AI & Lesion Segmentation Multi-View</strong>
                <span className="tag">{activeExpl?.method || 'Grad-CAM'}</span>
              </div>
              {activeExpl?.interpretation && (
                <p style={{ margin: '6px 0', fontSize: 12, color: '#4b5563' }}>
                  {activeExpl.interpretation}
                </p>
              )}
              {renderVisualInspection(
                activeExpl?.overlay_data_url || `${apiBase}/upload/images/${activeAnalysisImage.id}/file`,
                activeSeg?.mask_data_url,
                activeExpl?.heatmap_data_url,
                activeExpl?.overlay_data_url,
                activeSeg?.overlay_data_url,
              )}
            </div>
          </div>
        )}
      </section>
    );
  };

  const renderDashboard = () => (
    <>
      <div className="metric-grid">
        <Metric label="HAM10000 Dataset Images" value={dataset?.image_count.toLocaleString() ?? (datasetLoading ? 'Loading…' : 'Unavailable')} />
        <Metric label="Unique Lesion Clusters" value={dataset?.unique_lesion_count.toLocaleString() ?? (datasetLoading ? 'Loading…' : 'Unavailable')} />
        <Metric label="Your Managed Cases" value={token ? String(cases.length) : 'Sign in to view'} />
      </div>

      <div style={{ marginBottom: 20, padding: 14, background: '#f0f9f8', border: '1px solid #bce1db', borderRadius: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h3 style={{ margin: '0 0 4px', color: '#087f71', fontSize: 16 }}>DermAI Multi-Model Diagnostics Active</h3>
            <p style={{ margin: 0, fontSize: 12, color: '#406760' }}>
              Advanced CNNs (MobileNetV2, ResNet-50, EfficientNet-B0), Vision Transformers (ViT-B/16), Multi-Model Ensembles, Grad-CAM Explainable AI, ISBI U-Net Lesion Segmentation, and Clinical Severity Risk Categorization are active.
            </p>
          </div>
          <button className="button primary compact" type="button" onClick={() => setPage('models')}>
            View Model Benchmarks →
          </button>
        </div>
      </div>

      {datasetError && <p className="state-message error" role="alert">{datasetError}</p>}

      <div className="content-grid">
        <section className="panel">
          <p className="eyebrow">Medical Image Analysis</p>
          <h2>Analyze a Skin Lesion</h2>
          <p className="muted">
            Perform instant AI-assisted screening with Grad-CAM explainability, U-Net boundary mask, and clinical severity risk profiling.
          </p>
          <button className="button primary" type="button" onClick={() => setPage('upload')}>
            Open Image Analysis Screen →
          </button>
        </section>

        <section className="panel">
          <p className="eyebrow">Workspace</p>
          <h2>Case Management & Longitudinal History</h2>
          {token ? (
            <>
              <p className="muted">{cases.length} case{cases.length === 1 ? '' : 's'} registered in your account.</p>
              <button className="button secondary" type="button" onClick={() => setPage('cases')}>
                Manage Cases & History
              </button>
            </>
          ) : (
            <p className="muted">Sign in to create cases, track prediction histories, and compare multi-session scans.</p>
          )}
        </section>
      </div>

      <div style={{ marginTop: 24, padding: 18, background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <h4 style={{ margin: '0 0 4px', fontSize: 14 }}>HAM10000 Research Benchmark Ground Truth</h4>
          <p style={{ margin: 0, fontSize: 12, color: '#64748b' }}>
            Standardized dataset of 10,015 dermatoscopic images across 7 diagnostic categories, with zero-leakage lesion-level partitioning.
          </p>
        </div>
        <button className="button secondary compact" type="button" onClick={() => setPage('dataset')}>
          Explore Dataset Tab →
        </button>
      </div>
    </>
  );

  const renderAuthLanding = () => (
    <section className="auth-landing-card">
      <div style={{ textAlign: 'center', marginBottom: 20 }}>
        <div style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: 44, height: 44, borderRadius: 12, background: '#0d8f7e', color: '#fff', fontSize: 22, fontWeight: 700, marginBottom: 10 }}>
          D
        </div>
        <p className="eyebrow" style={{ margin: 0 }}>DermAI · Clinical Decision Support</p>
        <h2 style={{ margin: '4px 0 6px', fontSize: 22 }}>
          {authMode === 'login' ? 'Sign in to your workspace' : 'Create your account'}
        </h2>
        <p className="muted" style={{ margin: 0, fontSize: 13 }}>
          {authMode === 'login'
            ? 'Sign in to access your clinical cases, run deep learning evaluations, and review lesion scans.'
            : 'Register to manage patient records, image history, and multi-model AI inferences.'}
        </p>
        <div className="auth-tabs" role="group" aria-label="Choose account action" style={{ marginTop: 14 }}>
          <button
            type="button"
            className={authMode === 'login' ? 'selected' : ''}
            onClick={() => { setAuthMode('login'); setNotice(null); }}
          >
            Sign In
          </button>
          <button
            type="button"
            className={authMode === 'register' ? 'selected' : ''}
            onClick={() => { setAuthMode('register'); setNotice(null); }}
          >
            Register
          </button>
        </div>
      </div>

      <form className="form-stack" onSubmit={(event) => void submitAuth(event)}>
        {authMode === 'register' && (
          <>
            <label htmlFor="full-name">Full Name</label>
            <input
              id="full-name"
              value={authForm.fullName}
              onChange={(event) => setAuthForm({ ...authForm, fullName: event.target.value })}
              minLength={2}
              maxLength={255}
              autoComplete="name"
              required
              placeholder="e.g. Varun or Dr. Arjun"
            />
            <label style={{ marginTop: 10, display: 'block', fontSize: 12, fontWeight: 700 }}>Account Role / Access Level</label>
            <div className="auth-tabs" role="group" aria-label="Select role" style={{ marginBottom: 12 }}>
              <button
                type="button"
                className={authRole === 'patient' ? 'selected' : ''}
                onClick={() => setAuthRole('patient')}
              >
                👤 Patient
              </button>
              <button
                type="button"
                className={authRole === 'doctor' ? 'selected' : ''}
                onClick={() => setAuthRole('doctor')}
              >
                👨‍⚕️ Doctor / Dermatologist
              </button>
            </div>
          </>
        )}
        <label htmlFor="email">Email</label>
        <input
          id="email"
          type="email"
          value={authForm.email}
          onChange={(event) => setAuthForm({ ...authForm, email: event.target.value })}
          autoComplete="email"
          required
          placeholder="doctor@dermai.com or patient@dermai.com"
        />
        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          value={authForm.password}
          onChange={(event) => setAuthForm({ ...authForm, password: event.target.value })}
          minLength={authMode === 'register' ? 8 : 1}
          maxLength={72}
          autoComplete={authMode === 'login' ? 'current-password' : 'new-password'}
          required
          placeholder="••••••••"
        />
        <button className="button primary" type="submit" disabled={authBusy} style={{ marginTop: 8 }}>
          {authBusy ? 'Please wait…' : authMode === 'login' ? 'Sign In to Workspace' : 'Create Account'}
        </button>
      </form>

      {authMode === 'login' && (
        <div style={{ marginTop: 18, paddingTop: 14, borderTop: '1px solid #e5eded', fontSize: 12, color: '#4b5563' }}>
          <strong style={{ display: 'block', marginBottom: 6, color: '#1f2937' }}>Quick Test Credentials:</strong>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div
              style={{ background: '#f8fafc', padding: 8, borderRadius: 6, border: '1px solid #e2e8f0', cursor: 'pointer' }}
              onClick={() => setAuthForm({ ...authForm, email: 'doctor@dermai.com', password: 'Password123!' })}
              title="Click to fill doctor credentials"
            >
              <span style={{ fontWeight: 600, color: '#0369a1', display: 'block' }}>👨‍⚕️ Dr. Arjun (Doctor)</span>
              <span style={{ fontSize: 11, color: '#64748b' }}>doctor@dermai.com</span>
            </div>
            <div
              style={{ background: '#f8fafc', padding: 8, borderRadius: 6, border: '1px solid #e2e8f0', cursor: 'pointer' }}
              onClick={() => setAuthForm({ ...authForm, email: 'patient@dermai.com', password: 'Password123!' })}
              title="Click to fill patient credentials"
            >
              <span style={{ fontWeight: 600, color: '#15803d', display: 'block' }}>👤 Varun (Patient)</span>
              <span style={{ fontSize: 11, color: '#64748b' }}>patient@dermai.com</span>
            </div>
          </div>
          <span style={{ display: 'block', marginTop: 6, fontSize: 11, color: '#94a3b8' }}>
            Password: <code>Password123!</code> (Click card to autofill)
          </span>
        </div>
      )}
    </section>
  );

  return (
    <div className="app-shell">
      <header className="topbar">
        <a href="#" className="brand" onClick={(event) => { event.preventDefault(); setPage('dashboard'); }}>
          <span className="brand-mark" aria-hidden="true">D</span>
          <span><strong>DermAI</strong><small>Clinical AI Decision Support</small></span>
        </a>
        <nav className="navigation" aria-label="Main navigation">
          {navItems.map((item) => (
            <button
              key={item.id}
              className={`nav-link ${page === item.id || (item.id === 'cases' && page === 'case-detail') ? 'active' : ''}`}
              type="button"
              aria-current={page === item.id || (item.id === 'cases' && page === 'case-detail') ? 'page' : undefined}
              onClick={() => {
                setPage(item.id);
                setNotice(null);
                if (item.id === 'cases' && token) void loadCases(token);
                if (item.id === 'models') void loadModelsComparison();
              }}
            >
              {item.label}
            </button>
          ))}
        </nav>
        <div className="account-actions">
          {token ? (
            <>
              <span
                style={{
                  padding: '4px 12px',
                  borderRadius: '16px',
                  fontSize: '12px',
                  fontWeight: 600,
                  background: userRole === 'doctor' ? '#eff6ff' : '#f0fdf4',
                  color: userRole === 'doctor' ? '#1d4ed8' : '#15803d',
                  border: `1px solid ${userRole === 'doctor' ? '#bfdbfe' : '#bbf7d0'}`,
                }}
              >
                {userRole === 'doctor' ? `👨‍⚕️ Dr. Arjun (Doctor)` : `👤 Varun (Patient)`}
              </span>
              <button type="button" className="button secondary compact" onClick={signOut}>Sign out</button>
            </>
          ) : (
            <span className="session-label">Not signed in</span>
          )}
        </div>
      </header>

      <main className="page-container">
        {!token && page !== 'dataset' && page !== 'models' ? (
          <>
            <div className="page-title" style={{ textAlign: 'center', display: 'block' }}>
              <p className="eyebrow">DermAI · Skin Disease Detection System</p>
              <h1 style={{ margin: '4px 0' }}>Welcome to DermAI</h1>
              <p className="muted" style={{ maxWidth: 620, margin: '6px auto 0', fontSize: 14 }}>
                AI-assisted dermatological screening, U-Net boundary segmentation, and Grad-CAM explainability decision support.
              </p>
            </div>

            {notice && (
              <div className={`notice ${notice.kind}`} role={notice.kind === 'error' ? 'alert' : 'status'}>
                {notice.text}
                <button type="button" aria-label="Dismiss message" onClick={() => setNotice(null)}>×</button>
              </div>
            )}

            {renderAuthLanding()}
          </>
        ) : (
          <>
            <div className="page-title">
              <div>
                <p className="eyebrow">DermAI · Skin Disease Detection System</p>
                <h1>{titleForPage[page]}</h1>
              </div>
            </div>

            {notice && (
              <div className={`notice ${notice.kind}`} role={notice.kind === 'error' ? 'alert' : 'status'}>
                {notice.text}
                <button type="button" aria-label="Dismiss message" onClick={() => setNotice(null)}>×</button>
              </div>
            )}

            {page === 'dashboard' && renderDashboard()}
            {page === 'cases' && renderCases()}
            {page === 'upload' && renderUpload()}
            {page === 'models' && renderModels()}
            {page === 'dataset' && renderDataset()}
            {page === 'case-detail' && renderCaseDetail()}
          </>
        )}

        <footer className="app-footer">
          <strong>Clinical Decision Support Disclaimer:</strong> DermAI is positioned as an AI-assisted screening, research, and clinical decision-support system. It does not provide an autonomous medical diagnosis or clinical prescription.
        </footer>
      </main>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
