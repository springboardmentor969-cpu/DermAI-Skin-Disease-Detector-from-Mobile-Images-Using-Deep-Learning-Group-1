import { FormEvent, useEffect, useMemo, useState } from 'react';

type Page = 'dashboard' | 'cases' | 'upload' | 'dataset' | 'case-detail';

type CaseRecord = {
  id: number;
  title: string;
  description: string | null;
  status: string;
  created_at: string;
  updated_at: string;
  user_id: number;
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
  created_at: string;
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
      const body = await response.json() as {
        detail?: string | Array<{ msg?: string; loc?: Array<string | number> }>;
      };
      if (typeof body.detail === 'string') {
        detail = body.detail;
      } else if (Array.isArray(body.detail)) {
        detail = body.detail.map((issue) => issue.msg || 'Invalid input').join(' ');
      }
    } catch {
      // Use the HTTP status message when the server does not return JSON.
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

export default function App() {
  const [page, setPage] = useState<Page>('dashboard');
  const [token, setToken] = useState<string | null>(null);
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const [authBusy, setAuthBusy] = useState(false);
  const [authForm, setAuthForm] = useState({ email: '', password: '', fullName: '' });
  const [cases, setCases] = useState<CaseRecord[]>([]);
  const [casesLoading, setCasesLoading] = useState(false);
  const [caseBusy, setCaseBusy] = useState(false);
  const [caseTitle, setCaseTitle] = useState('');
  const [caseDescription, setCaseDescription] = useState('');
  const [selectedCase, setSelectedCase] = useState<CaseRecord | null>(null);
  const [caseDetailLoading, setCaseDetailLoading] = useState(false);
  const [dataset, setDataset] = useState<DatasetStatistics | null>(null);
  const [datasetLoading, setDatasetLoading] = useState(true);
  const [datasetError, setDatasetError] = useState('');
  const [notice, setNotice] = useState<Notice | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [uploadCaseId, setUploadCaseId] = useState('');
  const [uploadBusy, setUploadBusy] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadedImage | null>(null);
  const [fileError, setFileError] = useState('');

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

  const loadCases = async (accessToken: string) => {
    setCasesLoading(true);
    try {
      const records = await apiRequest<CaseRecord[]>('/cases', {}, accessToken);
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
          }),
        });
        setAuthMode('login');
        setAuthForm((current) => ({ ...current, password: '' }));
        setNotice({ kind: 'success', text: 'Account created. Sign in with your new credentials.' });
        return;
      }

      const result = await apiRequest<{ access_token: string }>('/auth/login', {
        method: 'POST',
        body: JSON.stringify({ email: authForm.email.trim(), password: authForm.password }),
      });
      setToken(result.access_token);
      setAuthForm((current) => ({ ...current, password: '' }));
      setNotice({ kind: 'success', text: 'Signed in successfully.' });
      await loadCases(result.access_token);
    } catch (error) {
      showApiError(error);
    } finally {
      setAuthBusy(false);
    }
  };

  const signOut = () => {
    setToken(null);
    setCases([]);
    setSelectedCase(null);
    setUploadResult(null);
    setSelectedFile(null);
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
      const created = await apiRequest<CaseRecord>('/cases', {
        method: 'POST',
        body: JSON.stringify({ title: caseTitle.trim(), description: caseDescription.trim() || null }),
      }, token);
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
    setCaseDetailLoading(true);
    setNotice(null);
    try {
      const record = await apiRequest<CaseRecord>(`/cases/${caseId}`, {}, token);
      setSelectedCase(record);
    } catch (error) {
      showApiError(error);
    } finally {
      setCaseDetailLoading(false);
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
      const uploaded = await apiRequest<UploadedImage>(`/upload/${uploadCaseId}`, {
        method: 'POST',
        body: data,
      }, token);
      setUploadResult(uploaded);
      setNotice({ kind: 'success', text: `${uploaded.filename} was uploaded and validated by the server.` });
    } catch (error) {
      showApiError(error);
    } finally {
      setUploadBusy(false);
    }
  };

  const navItems: { id: Page; label: string }[] = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'cases', label: 'Cases' },
    { id: 'upload', label: 'Image upload' },
    { id: 'dataset', label: 'Dataset' },
  ];

  const titleForPage: Record<Page, string> = {
    dashboard: 'Research dashboard',
    cases: 'Case management',
    upload: 'Upload an image',
    dataset: 'Dataset overview',
    'case-detail': selectedCase?.title ?? 'Case details',
  };

  const renderDataset = () => (
    <section className="panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Dataset statistics</p>
          <h2>HAM10000</h2>
          <p className="muted">These are dataset records, not user cases or model predictions.</p>
        </div>
        <span className="tag">Dermoscopic dataset</span>
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
          <p className="source-note">Source: {dataset.source}. Regenerate the dataset report after changing its source data.</p>
        </>
      )}
    </section>
  );

  const renderCases = () => (
    <div className="content-grid">
      <section className="panel">
        <p className="eyebrow">Case management</p>
        <h2>Create a case</h2>
        {!token ? (
          <p className="state-message">Sign in to create and manage cases.</p>
        ) : (
          <form onSubmit={createCase} className="form-stack">
            <label htmlFor="case-title">Case title</label>
            <input
              id="case-title"
              value={caseTitle}
              onChange={(event) => setCaseTitle(event.target.value)}
              minLength={2}
              maxLength={255}
              required
              autoComplete="off"
            />
            <label htmlFor="case-description">Notes <span className="muted">(optional; do not include identifying information)</span></label>
            <textarea
              id="case-description"
              value={caseDescription}
              onChange={(event) => setCaseDescription(event.target.value)}
              rows={4}
            />
            <button className="button primary" type="submit" disabled={caseBusy}>
              {caseBusy ? 'Creating…' : 'Create case'}
            </button>
          </form>
        )}
      </section>
      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Your records</p>
            <h2>Cases</h2>
          </div>
          {token && <button className="button secondary compact" type="button" onClick={() => void loadCases(token)} disabled={casesLoading}>Refresh</button>}
        </div>
        {!token && <p className="state-message">Sign in to see your cases.</p>}
        {token && casesLoading && <p className="state-message" role="status">Loading cases…</p>}
        {token && !casesLoading && cases.length === 0 && (
          <p className="state-message">No cases yet. Create a case to begin.</p>
        )}
        {token && !casesLoading && cases.length > 0 && (
          <ul className="case-list">
            {cases.map((record) => (
              <li key={record.id} className="case-list-item">
                <div>
                  <h3>{record.title}</h3>
                  <p>{record.description || 'No notes provided.'}</p>
                  <span className="case-date">Created {formatDate(record.created_at)}</span>
                </div>
                <button className="button secondary" type="button" onClick={() => void openCase(record.id)}>
                  View details
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
      <p className="eyebrow">Image handling</p>
      <h2>Upload a case image</h2>
      <p className="muted">Uploads are stored separately from the HAM10000 dataset. This application does not perform diagnosis or prediction.</p>
      {!token ? (
        <p className="state-message">Sign in to upload images.</p>
      ) : cases.length === 0 ? (
        <div className="state-message">
          No cases available. <button className="text-button" type="button" onClick={() => setPage('cases')}>Create a case first</button>.
        </div>
      ) : (
        <form className="form-stack" onSubmit={(event) => void uploadImage(event)}>
          <label htmlFor="upload-case">Associate with case</label>
          <select id="upload-case" value={uploadCaseId} onChange={(event) => setUploadCaseId(event.target.value)} required>
            <option value="">Select a case</option>
            {cases.map((record) => <option key={record.id} value={record.id}>{record.title}</option>)}
          </select>
          <label htmlFor="image-file">JPEG or PNG image</label>
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
                <span>{uploadBusy ? 'Uploading…' : uploadResult ? 'Uploaded' : 'Ready to upload'}</span>
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
                Cancel selection
              </button>
            </div>
          )}
          <button className="button primary" type="submit" disabled={uploadBusy || !selectedFile}>
            {uploadBusy ? 'Uploading…' : 'Upload image'}
          </button>
          {uploadResult && (
            <div className="upload-receipt" role="status">
              <strong>Server-confirmed upload</strong>
              <span>{uploadResult.filename} · {formatBytes(uploadResult.file_size)}</span>
              <span>{uploadResult.width} × {uploadResult.height} px · {uploadResult.content_type}</span>
              <span>Status: {uploadResult.status} · Case #{uploadResult.case_id}</span>
            </div>
          )}
        </form>
      )}
    </section>
  );

  const renderDashboard = () => (
    <>
      <div className="metric-grid">
        <Metric label="HAM10000 images" value={dataset?.image_count.toLocaleString() ?? (datasetLoading ? 'Loading…' : 'Unavailable')} />
        <Metric label="HAM10000 lesion IDs" value={dataset?.unique_lesion_count.toLocaleString() ?? (datasetLoading ? 'Loading…' : 'Unavailable')} />
        <Metric label="Your cases" value={token ? String(cases.length) : 'Sign in to view'} />
      </div>
      {datasetError && <p className="state-message error" role="alert">{datasetError}</p>}
      <div className="content-grid">
        <section className="panel">
          <p className="eyebrow">HAM10000 dataset</p>
          <h2>Dermoscopic research images</h2>
          <p className="dataset-note">{dataset?.image_type ?? 'Dataset statistics are loaded from the backend inspection report.'}</p>
          <button className="button secondary" type="button" onClick={() => setPage('dataset')}>View dataset distribution</button>
        </section>
        <section className="panel">
          <p className="eyebrow">Your workspace</p>
          <h2>Cases and predictions</h2>
          {token ? (
            <>
              <p className="muted">{cases.length} case{cases.length === 1 ? '' : 's'} belonging to your account.</p>
              <p className="muted">No disease predictions are generated by this application.</p>
              <button className="button secondary" type="button" onClick={() => setPage('cases')}>Manage cases</button>
            </>
          ) : (
            <p className="muted">Sign in to view your case records. Dataset counts shown here are separate from user case data.</p>
          )}
        </section>
      </div>
      {dataset && renderDataset()}
    </>
  );

  const renderCaseDetail = () => (
    <section className="panel">
      <button className="text-button back-button" type="button" onClick={() => setPage('cases')}>← Back to cases</button>
      {caseDetailLoading && <p className="state-message" role="status">Loading case details…</p>}
      {!caseDetailLoading && !selectedCase && <p className="state-message">Case details are unavailable.</p>}
      {selectedCase && (
        <>
          <p className="eyebrow">Case #{selectedCase.id}</p>
          <h2>{selectedCase.title}</h2>
          <p className="detail-copy">{selectedCase.description || 'No notes provided.'}</p>
          <dl className="detail-list">
            <div><dt>Status</dt><dd>{selectedCase.status}</dd></div>
            <div><dt>Created</dt><dd>{formatDate(selectedCase.created_at)}</dd></div>
            <div><dt>Last updated</dt><dd>{formatDate(selectedCase.updated_at)}</dd></div>
          </dl>
          <p className="muted">No prediction or diagnosis is available for this case.</p>
          <button className="button primary" type="button" onClick={() => {
            setUploadCaseId(String(selectedCase.id));
            setPage('upload');
          }}>Upload an image to this case</button>
        </>
      )}
    </section>
  );

  return (
    <div className="app-shell">
      <header className="topbar">
        <a href="#" className="brand" onClick={(event) => { event.preventDefault(); setPage('dashboard'); }}>
          <span className="brand-mark" aria-hidden="true">D</span>
          <span><strong>DermAI</strong><small>Research workspace</small></span>
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
              }}
            >
              {item.label}
            </button>
          ))}
        </nav>
        <div className="account-actions">
          {token ? (
            <>
              <span className="session-label">Signed in · memory-only session</span>
              <button type="button" className="button secondary compact" onClick={signOut}>Sign out</button>
            </>
          ) : <span className="session-label">Not signed in</span>}
        </div>
      </header>

      <main className="page-container">
        <div className="page-title">
          <div>
            <p className="eyebrow">DermAI · HAM10000 research</p>
            <h1>{titleForPage[page]}</h1>
          </div>
          {!token && (
            <button className="button primary" type="button" onClick={() => document.getElementById('auth-panel')?.scrollIntoView({ behavior: 'smooth' })}>
              Sign in
            </button>
          )}
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
        {page === 'dataset' && renderDataset()}
        {page === 'case-detail' && renderCaseDetail()}

        {!token && (
          <section className="auth-panel panel" id="auth-panel">
            <div>
              <p className="eyebrow">Account access</p>
              <h2>{authMode === 'login' ? 'Sign in to your workspace' : 'Create your account'}</h2>
              <p className="muted">Access tokens stay in memory only and are cleared when you sign out or reload this page. Sign in again after a refresh.</p>
              <div className="auth-tabs" role="group" aria-label="Choose account action">
                <button type="button" className={authMode === 'login' ? 'selected' : ''} onClick={() => { setAuthMode('login'); setNotice(null); }}>Login</button>
                <button type="button" className={authMode === 'register' ? 'selected' : ''} onClick={() => { setAuthMode('register'); setNotice(null); }}>Register</button>
              </div>
            </div>
            <form className="form-stack" onSubmit={(event) => void submitAuth(event)}>
              {authMode === 'register' && (
                <>
                  <label htmlFor="full-name">Full name</label>
                  <input
                    id="full-name"
                    value={authForm.fullName}
                    onChange={(event) => setAuthForm({ ...authForm, fullName: event.target.value })}
                    minLength={2}
                    maxLength={255}
                    autoComplete="name"
                    required
                  />
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
              />
              {authMode === 'register' && <p className="muted">Use at least 8 characters; the maximum is 72 UTF-8 bytes.</p>}
              <button className="button primary" type="submit" disabled={authBusy}>
                {authBusy ? 'Please wait…' : authMode === 'login' ? 'Sign in' : 'Create account'}
              </button>
            </form>
          </section>
        )}
        <footer className="app-footer">Research support only. DermAI does not provide diagnosis or medical advice.</footer>
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
