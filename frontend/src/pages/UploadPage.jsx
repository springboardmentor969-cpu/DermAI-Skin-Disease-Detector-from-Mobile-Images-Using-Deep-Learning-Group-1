import React, { useState, useRef, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import ReactCrop, { centerCrop, makeAspectCrop } from 'react-image-crop';
import { casesApi } from '../api/client';
import { QualityBadge } from '../components/QualityBadge';
import {
  Camera,
  Upload,
  Crop as CropIcon,
  CheckCircle,
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  RefreshCw,
  Image as ImageIcon,
  Sparkles,
  Info,
} from 'lucide-react';

function getInitialCrop(mediaWidth, mediaHeight) {
  return centerCrop(
    makeAspectCrop(
      {
        unit: '%',
        width: 80,
      },
      1, // 1:1 Aspect Ratio
      mediaWidth,
      mediaHeight
    ),
    mediaWidth,
    mediaHeight
  );
}

export const UploadPage = () => {
  const { caseId } = useParams();
  const navigate = useNavigate();

  const [caseData, setCaseData] = useState(null);
  const [source, setSource] = useState('camera'); // 'camera' or 'gallery'
  const [imageSrc, setImageSrc] = useState(null);
  const [crop, setCrop] = useState(null);
  const [completedCrop, setCompletedCrop] = useState(null);
  const [originalFile, setOriginalFile] = useState(null);

  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [error, setError] = useState('');

  const imgRef = useRef(null);
  const galleryInputRef = useRef(null);
  const cameraInputRef = useRef(null);

  // Load case metadata
  useEffect(() => {
    const fetchCase = async () => {
      try {
        const data = await casesApi.getCase(caseId);
        setCaseData(data);
      } catch (err) {
        setError('Failed to fetch case information.');
      }
    };
    fetchCase();
  }, [caseId]);

  // Handle file selection
  const handleFileChange = (e, captureSource) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];

      // Client-side quick checks
      if (!['image/jpeg', 'image/png', 'image/jpg'].includes(file.type)) {
        setError('Only JPG and PNG images are supported.');
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        setError('File size exceeds the 10 MB limit.');
        return;
      }

      setError('');
      setUploadResult(null);
      setOriginalFile(file);
      setSource(captureSource);

      const reader = new FileReader();
      reader.addEventListener('load', () => {
        setImageSrc(reader.result?.toString() || null);
      });
      reader.readAsDataURL(file);
    }
  };

  const onImageLoad = (e) => {
    const { width, height } = e.currentTarget;
    setCrop(getInitialCrop(width, height));
  };

  // Convert cropped area from canvas to a Blob/File
  const getCroppedBlob = async (image, cropConfig) => {
    const canvas = document.createElement('canvas');
    const scaleX = image.naturalWidth / image.width;
    const scaleY = image.naturalHeight / image.height;

    const pixelCrop = {
      x: cropConfig.x * scaleX,
      y: cropConfig.y * scaleY,
      width: cropConfig.width * scaleX,
      height: cropConfig.height * scaleY,
    };

    canvas.width = Math.floor(pixelCrop.width);
    canvas.height = Math.floor(pixelCrop.height);

    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    ctx.drawImage(
      image,
      pixelCrop.x,
      pixelCrop.y,
      pixelCrop.width,
      pixelCrop.height,
      0,
      0,
      pixelCrop.width,
      pixelCrop.height
    );

    return new Promise((resolve) => {
      canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.95);
    });
  };

  const handleUpload = async () => {
    if (!imageSrc || !originalFile) {
      setError('Please select or capture an image first.');
      return;
    }

    setUploading(true);
    setError('');

    try {
      let fileToUpload = originalFile;

      // If user performed a crop, upload the cropped version
      if (completedCrop && imgRef.current && completedCrop.width > 0 && completedCrop.height > 0) {
        const croppedBlob = await getCroppedBlob(imgRef.current, completedCrop);
        if (croppedBlob) {
          fileToUpload = new File([croppedBlob], originalFile.name.replace(/\.[^/.]+$/, '') + '_cropped.jpg', {
            type: 'image/jpeg',
          });
        }
      }

      const formData = new FormData();
      formData.append('file', fileToUpload);
      formData.append('source', source);

      const result = await casesApi.uploadImage(caseId, formData);
      setUploadResult(result);
    } catch (err) {
      const detail = err.response?.data?.detail || 'Failed to upload image. Please try again.';
      setError(detail);
    } finally {
      setUploading(false);
    }
  };

  const resetSelection = () => {
    setImageSrc(null);
    setCrop(null);
    setCompletedCrop(null);
    setOriginalFile(null);
    setUploadResult(null);
    setError('');
    if (galleryInputRef.current) galleryInputRef.current.value = '';
    if (cameraInputRef.current) cameraInputRef.current.value = '';
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-10">
      {/* Navigation Header */}
      <div className="flex items-center justify-between mb-6">
        <Link
          to={`/cases/${caseId}`}
          className="inline-flex items-center space-x-1.5 text-xs font-semibold text-slate-500 hover:text-slate-900 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Case #{caseId} Details</span>
        </Link>
      </div>

      {/* Case Context Card */}
      {caseData && (
        <div className="bg-white rounded-xl border border-slate-200/80 px-5 py-4 mb-6 shadow-xs flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="text-xs uppercase tracking-wider font-semibold text-slate-400">
              Active Case Target
            </div>
            <div className="text-base font-bold text-slate-900 flex items-center space-x-2 mt-0.5">
              <span>{caseData.body_location}</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-sky-50 text-sky-700 font-medium">
                Patient #{caseData.patient_id}
              </span>
            </div>
          </div>
          <div className="text-xs text-slate-500 sm:text-right">
            <div>Age: {caseData.patient?.age || 'N/A'} &bull; Gender: {caseData.patient?.gender || 'N/A'}</div>
            <div className="text-slate-400">{caseData.images?.length || 0} existing scans</div>
          </div>
        </div>
      )}

      {/* Main Upload Card */}
      <div className="bg-white rounded-2xl shadow-xl shadow-slate-200/60 border border-slate-200/80 p-6 sm:p-8 transition-all">
        {/* Error Alert */}
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-50 border border-rose-200 flex items-start space-x-3 text-sm text-rose-800 animate-fadeIn">
            <AlertTriangle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
            <div className="flex-1">{error}</div>
          </div>
        )}

        {/* POST-UPLOAD QUALITY RESULT VIEW */}
        {uploadResult ? (
          <div className="space-y-6 animate-fadeIn">
            {/* Header Status */}
            <div className="text-center py-4">
              <div className="w-16 h-16 rounded-full bg-emerald-50 text-emerald-600 flex items-center justify-center mx-auto mb-3 shadow-sm border border-emerald-100">
                <CheckCircle className="w-8 h-8" />
              </div>
              <h2 className="text-2xl font-bold text-slate-900">Upload Successful</h2>
              <p className="text-sm text-slate-500 mt-1">
                Skin lesion photograph stored securely in MinIO storage.
              </p>
            </div>

            {/* Quality Assessment Card */}
            <div className="bg-slate-50/80 rounded-2xl p-6 border border-slate-200/80">
              <div className="flex items-center justify-between pb-4 border-b border-slate-200/60 mb-4">
                <div className="flex items-center space-x-2">
                  <Sparkles className="w-5 h-5 text-sky-600" />
                  <span className="font-bold text-slate-900 text-sm">Automated Image Quality Assessment</span>
                </div>
                <QualityBadge label={uploadResult.quality_label} size="lg" />
              </div>

              {/* Warning Notice if Poor Quality */}
              {uploadResult.warning && (
                <div className="mb-5 p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-sm flex items-start space-x-3">
                  <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
                  <div className="leading-relaxed font-medium">{uploadResult.warning}</div>
                </div>
              )}

              {/* Quality Metrics Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                <div className="bg-white p-3.5 rounded-xl border border-slate-200 shadow-xs">
                  <div className="text-xs text-slate-500 font-medium">Blur Score</div>
                  <div className="text-lg font-bold text-slate-900 mt-0.5">
                    {uploadResult.quality_details?.blur_score ?? 'N/A'}
                  </div>
                  <div className="text-[10px] text-slate-400">Laplacian var (higher is sharper)</div>
                </div>

                <div className="bg-white p-3.5 rounded-xl border border-slate-200 shadow-xs">
                  <div className="text-xs text-slate-500 font-medium">Brightness</div>
                  <div className="text-lg font-bold text-slate-900 mt-0.5">
                    {uploadResult.quality_details?.brightness ?? 'N/A'}
                  </div>
                  <div className="text-[10px] text-slate-400">Mean scale (0-255)</div>
                </div>

                <div className="bg-white p-3.5 rounded-xl border border-slate-200 shadow-xs">
                  <div className="text-xs text-slate-500 font-medium">Contrast</div>
                  <div className="text-lg font-bold text-slate-900 mt-0.5">
                    {uploadResult.quality_details?.contrast ?? 'N/A'}
                  </div>
                  <div className="text-[10px] text-slate-400">RMS intensity dev</div>
                </div>

                <div className="bg-white p-3.5 rounded-xl border border-slate-200 shadow-xs">
                  <div className="text-xs text-slate-500 font-medium">Resolution</div>
                  <div className="text-lg font-bold text-slate-900 mt-0.5">
                    {uploadResult.quality_details?.resolution?.width || uploadResult.quality_details?.width}x
                    {uploadResult.quality_details?.resolution?.height || uploadResult.quality_details?.height}
                  </div>
                  <div className="text-[10px] text-slate-400">Target: &ge; 224x224</div>
                </div>
              </div>

              {/* Detected Issues List */}
              {uploadResult.quality_details?.issues && uploadResult.quality_details.issues.length > 0 && (
                <div className="mt-4 pt-3 border-t border-slate-200/60 text-xs text-slate-600">
                  <span className="font-semibold text-slate-700">Noted conditions: </span>
                  {uploadResult.quality_details.issues.join(' • ')}
                </div>
              )}
            </div>

            {/* Next Steps Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-end gap-3 pt-4 border-t border-slate-100">
              <button
                onClick={resetSelection}
                className="w-full sm:w-auto px-4 py-2.5 rounded-xl border border-slate-200 text-slate-700 hover:bg-slate-50 font-semibold text-sm transition-colors flex items-center justify-center space-x-2"
              >
                <RefreshCw className="w-4 h-4" />
                <span>Upload Another Photo</span>
              </button>

              <button
                onClick={() => navigate(`/cases/${caseId}`)}
                className="w-full sm:w-auto px-5 py-2.5 rounded-xl bg-sky-600 hover:bg-sky-700 text-white font-semibold text-sm shadow-md transition-all flex items-center justify-center space-x-2"
              >
                <span>View Case Details</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        ) : !imageSrc ? (
          /* STEP 1: CHOOSE PHOTO (CAMERA OR GALLERY) */
          <div className="space-y-6">
            <div className="text-center max-w-md mx-auto mb-6">
              <h2 className="text-xl font-bold text-slate-900">Upload Skin Lesion Photograph</h2>
              <p className="text-xs text-slate-500 mt-1">
                Choose an image from your device gallery or snap a direct camera photo. JPG or PNG, max 10MB, min 224x224.
              </p>
            </div>

            {/* Hidden Input Elements */}
            <input
              id="gallery-file-input"
              data-testid="gallery-file-input"
              type="file"
              ref={galleryInputRef}
              accept="image/jpeg,image/png,image/jpg"
              onChange={(e) => handleFileChange(e, 'gallery')}
              className="hidden"
            />
            <input
              id="camera-file-input"
              data-testid="camera-file-input"
              type="file"
              ref={cameraInputRef}
              accept="image/*"
              capture="environment"
              onChange={(e) => handleFileChange(e, 'camera')}
              className="hidden"
            />

            {/* Action Cards with Drag and Drop */}
            <div
              className="grid grid-cols-1 sm:grid-cols-2 gap-4"
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                  handleFileChange({ target: { files: e.dataTransfer.files } }, 'gallery');
                }
              }}
            >
              {/* Option A: Camera */}
              <button
                type="button"
                id="camera-upload-button"
                data-testid="camera-upload-button"
                onClick={() => cameraInputRef.current?.click()}
                className="p-8 rounded-2xl border-2 border-dashed border-sky-200 hover:border-sky-500 bg-sky-50/40 hover:bg-sky-50/80 transition-all text-center flex flex-col items-center justify-center space-y-3 cursor-pointer group"
              >
                <div className="w-14 h-14 rounded-2xl bg-sky-600 text-white flex items-center justify-center shadow-lg shadow-sky-600/20 group-hover:scale-110 transition-transform">
                  <Camera className="w-7 h-7" />
                </div>
                <div>
                  <div className="font-bold text-slate-900 text-base">Capture with Camera</div>
                  <div className="text-xs text-slate-500 mt-0.5">Use smartphone or webcam</div>
                </div>
              </button>

              {/* Option B: Gallery */}
              <button
                type="button"
                id="gallery-upload-button"
                data-testid="gallery-upload-button"
                onClick={() => galleryInputRef.current?.click()}
                className="p-8 rounded-2xl border-2 border-dashed border-slate-200 hover:border-slate-400 bg-slate-50/40 hover:bg-slate-50 transition-all text-center flex flex-col items-center justify-center space-y-3 cursor-pointer group"
              >
                <div className="w-14 h-14 rounded-2xl bg-slate-800 text-white flex items-center justify-center shadow-lg shadow-slate-800/20 group-hover:scale-110 transition-transform">
                  <Upload className="w-7 h-7" />
                </div>
                <div>
                  <div className="font-bold text-slate-900 text-base">Upload from Gallery</div>
                  <div className="text-xs text-slate-500 mt-0.5">Select existing JPG / PNG scan</div>
                </div>
              </button>
            </div>

            {/* Clinical Photo Tips */}
            <div className="p-4 rounded-xl bg-sky-50/60 border border-sky-100 flex items-start space-x-3 text-xs text-slate-600">
              <Info className="w-4 h-4 text-sky-600 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold text-sky-900">Best results tip:</span> Ensure the skin lesion is centered, well-illuminated by uniform white light, in sharp focus, and free of glare or shadows.
              </div>
            </div>
          </div>
        ) : (
          /* STEP 2: PREVIEW & INTERACTIVE CROP (react-image-crop) */
          <div className="space-y-6">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div>
                <h3 className="text-base font-bold text-slate-900 flex items-center space-x-2">
                  <CropIcon className="w-4 h-4 text-sky-600" />
                  <span>Adjust Framing &amp; Crop</span>
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Center the lesion within the 1:1 square boundary for optimal neural network inference.
                </p>
              </div>

              <button
                onClick={resetSelection}
                className="text-xs text-slate-500 hover:text-slate-900 px-2 py-1 rounded-md hover:bg-slate-100 transition-colors"
              >
                Choose Different
              </button>
            </div>

            {/* Crop Canvas Area */}
            <div className="flex justify-center bg-slate-950/5 rounded-2xl p-4 overflow-hidden border border-slate-200">
              <ReactCrop
                crop={crop}
                onChange={(_, percentCrop) => setCrop(percentCrop)}
                onComplete={(c) => setCompletedCrop(c)}
                aspect={1}
                className="max-h-[500px]"
              >
                <img
                  ref={imgRef}
                  alt="Lesion preview"
                  src={imageSrc}
                  onLoad={onImageLoad}
                  className="max-h-[500px] w-auto object-contain rounded-lg"
                />
              </ReactCrop>
            </div>

            {/* Upload Action Button */}
            <div className="flex items-center justify-end space-x-3 pt-2">
              <button
                type="button"
                onClick={resetSelection}
                disabled={uploading}
                className="px-4 py-2.5 rounded-xl border border-slate-200 text-slate-600 hover:bg-slate-50 text-sm font-semibold transition-colors"
              >
                Cancel
              </button>

              <button
                type="button"
                onClick={handleUpload}
                disabled={uploading}
                className="py-2.5 px-6 rounded-xl bg-sky-600 hover:bg-sky-700 active:bg-sky-800 text-white font-medium text-sm shadow-md shadow-sky-600/20 hover:shadow-lg hover:shadow-sky-600/30 transition-all flex items-center space-x-2 disabled:opacity-60 cursor-pointer"
              >
                {uploading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    <span>Uploading &amp; Assessing Quality...</span>
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4" />
                    <span>Upload &amp; Analyze Quality</span>
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
