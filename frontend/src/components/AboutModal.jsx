import React from 'react';
import {
  X,
  ShieldAlert,
  Info,
  Layers,
  Database,
  Cpu,
  CheckCircle2,
  ExternalLink,
} from 'lucide-react';

export const DISEASE_DEFINITIONS = [
  {
    code: 'akiec',
    name: 'Actinic Keratoses / Intraepithelial Carcinoma',
    category: 'Pre-Malignant',
    description:
      'Common non-invasive variants of squamous cell carcinoma that can be locally invasive if untreated.',
    badgeClass: 'bg-amber-100 text-amber-800 border-amber-300',
  },
  {
    code: 'bcc',
    name: 'Basal Cell Carcinoma',
    category: 'Malignant (Common)',
    description:
      'The most frequent epithelial skin cancer. Rarely metastasizes but causes significant local tissue destruction.',
    badgeClass: 'bg-rose-100 text-rose-800 border-rose-300',
  },
  {
    code: 'bkl',
    name: 'Benign Keratosis',
    category: 'Benign',
    description:
      'Generic grouping containing solar lentigines, seborrheic keratoses, and lichen-planus like keratoses.',
    badgeClass: 'bg-emerald-100 text-emerald-800 border-emerald-300',
  },
  {
    code: 'df',
    name: 'Dermatofibroma',
    category: 'Benign',
    description:
      'Benign dermal proliferation or reaction to minor injury, characterized by firm dermal nodules.',
    badgeClass: 'bg-emerald-100 text-emerald-800 border-emerald-300',
  },
  {
    code: 'mel',
    name: 'Melanoma',
    category: 'Malignant (High Urgency)',
    description:
      'Malignant neoplasm of melanocytes. Highly lethal if not diagnosed at an early, thin surgical stage.',
    badgeClass: 'bg-red-100 text-red-900 border-red-400 font-bold',
  },
  {
    code: 'nv',
    name: 'Melanocytic Nevus',
    category: 'Benign (Common)',
    description:
      'Benign melanocytic nevi (moles) presenting with dense network, globular, or homogenous pigmentation patterns.',
    badgeClass: 'bg-emerald-100 text-emerald-800 border-emerald-300',
  },
  {
    code: 'vasc',
    name: 'Vascular Lesions',
    category: 'Benign',
    description:
      'Angiomas, cherry hemangiomas, pyogenic granulomas, and related benign vascular proliferations.',
    badgeClass: 'bg-emerald-100 text-emerald-800 border-emerald-300',
  },
];

export const AboutModal = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="about-modal-title"
      className="fixed inset-0 z-50 overflow-y-auto bg-slate-900/60 backdrop-blur-sm flex items-center justify-center p-4 sm:p-6 animate-fadeIn"
      onClick={onClose}
    >
      <div
        className="relative bg-white rounded-2xl shadow-2xl max-w-3xl w-full max-h-[90vh] flex flex-col border border-slate-200 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b border-slate-200 bg-slate-50/50">
          <div className="flex items-center space-x-2.5">
            <div className="w-9 h-9 rounded-xl bg-sky-100 text-sky-700 flex items-center justify-center shadow-xs">
              <Info className="w-5 h-5" />
            </div>
            <div>
              <h2 id="about-modal-title" className="text-lg font-bold text-slate-900">
                About DermAI
              </h2>
              <p className="text-xs text-slate-500">
                Clinical Decision-Support &amp; Skin Lesion Screening System
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-200/60 transition-colors"
            aria-label="Close dialog"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Scrollable Body */}
        <div className="p-6 space-y-6 overflow-y-auto text-sm text-slate-600">
          {/* Mandatory Medical Disclaimer */}
          <div className="p-4 rounded-xl bg-amber-50 border border-amber-300/80 text-amber-900 flex items-start space-x-3 shadow-xs">
            <ShieldAlert className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
            <div className="text-xs space-y-1">
              <span className="font-bold uppercase tracking-wider block text-amber-950">
                Clinical Disclaimer — Educational Decision Support Only
              </span>
              <p className="leading-relaxed">
                <strong>Not a substitute for medical advice.</strong> DermAI is an educational research and clinical triage decision-support tool. It does <strong>NOT</strong> provide definitive medical diagnosis or replace consultation, dermoscopic inspection, or histopathological biopsy by a licensed physician or board-certified dermatologist.
              </p>
            </div>
          </div>

          {/* Project Summary */}
          <div>
            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider mb-2 flex items-center space-x-2">
              <Layers className="w-4 h-4 text-sky-600" />
              <span>Project Mission &amp; Overview</span>
            </h3>
            <p className="leading-relaxed">
              DermAI is an end-to-end, privacy-preserving mobile dermatology screening and decision-support system. It enables non-invasive capture of dermoscopic skin lesions, assesses photographic visual quality in real-time, segments lesion margins, and computes calibrated multi-class disease predictions with transparent Grad-CAM saliency visualizations.
            </p>
          </div>

          {/* 7 Disease Classes */}
          <div>
            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider mb-3 flex items-center space-x-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              <span>7 Diagnostic Skin Lesion Categories</span>
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {DISEASE_DEFINITIONS.map((d) => (
                <div
                  key={d.code}
                  className="p-3 rounded-xl bg-slate-50 border border-slate-200/80 space-y-1"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-slate-900 text-xs">
                      {d.name} ({d.code.toUpperCase()})
                    </span>
                    <span className={`text-[10px] px-2 py-0.5 rounded-full border ${d.badgeClass}`}>
                      {d.category}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-500 leading-snug">
                    {d.description}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Model Architecture & Calibration */}
          <div>
            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider mb-2 flex items-center space-x-2">
              <Cpu className="w-4 h-4 text-indigo-600" />
              <span>Deep Learning Architecture &amp; Calibration</span>
            </h3>
            <ul className="list-disc pl-5 space-y-1.5 text-xs text-slate-600">
              <li>
                <strong>Multi-Model Soft-Voting Ensemble:</strong> Blends prediction logits across convolutional (ResNet-50, DenseNet-121, EfficientNet-B0) and vision transformer (Swin Transformer) models.
              </li>
              <li>
                <strong>Temperature Scaling Probability Calibration:</strong> Optimizes validation NLL to reduce overconfidence, guaranteeing an Expected Calibration Error (ECE) strictly under 10% (&lt; 0.10).
              </li>
              <li>
                <strong>Grad-CAM Visual Explainability:</strong> Computes gradients of the target class score with respect to deep convolutional feature maps to highlight the physical lesion markers influencing the AI prediction.
              </li>
            </ul>
          </div>

          {/* Dataset Attribution */}
          <div className="pt-3 border-t border-slate-200/80">
            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider mb-1.5 flex items-center space-x-2">
              <Database className="w-4 h-4 text-slate-500" />
              <span>Dataset Attribution</span>
            </h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              <strong>Dataset: HAM10000 by Tschandl et al. (2018)</strong> — <em>The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions</em>, Scientific Data 5:180161 (2018). DOI: 10.1038/sdata.2018.161.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-slate-50 border-t border-slate-200 flex items-center justify-between">
          <span className="text-xs text-slate-500">
            DermAI &bull; Final Production Milestone
          </span>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold shadow-xs transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
