import React, { useEffect } from 'react';
import { Link, Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
  ShieldAlert,
  Sparkles,
  ArrowRight,
  Activity,
  Layers,
  CheckCircle,
  FileText,
  UserCheck,
  Stethoscope,
  Microscope,
  Brain,
  Zap,
} from 'lucide-react';

export const LandingPage = () => {
  const { isAuthenticated, user, loading } = useAuth();

  useEffect(() => {
    document.title = 'DermAI — AI-Powered Skin Lesion Detection';
  }, []);

  // Redirect authenticated users directly to their dashboard
  if (!loading && isAuthenticated) {
    if (user?.role === 'clinician' || user?.role === 'admin') {
      return <Navigate to="/clinician" replace />;
    }
    return <Navigate to="/dashboard" replace />;
  }

  return (
    <div className="min-h-[calc(100vh-4rem)] flex flex-col justify-between">
      {/* Hero Section */}
      <section className="relative overflow-hidden pt-12 pb-16 sm:pt-20 sm:pb-24 lg:pt-24">
        {/* Subtle background glow */}
        <div
          aria-hidden="true"
          className="absolute inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80"
        >
          <div
            style={{
              clipPath:
                'polygon(74.1% 44.1%, 100% 61.6%, 97.5% 26.9%, 85.5% 0.1%, 80.7% 2%, 72.5% 32.5%, 60.2% 62.4%, 52.4% 68.1%, 47.5% 58.3%, 45.2% 34.5%, 27.5% 76.7%, 0.1% 64.9%, 17.9% 100%, 27.6% 76.8%, 76.1% 97.7%, 74.1% 44.1%)',
            }}
            className="relative left-[calc(50%-11rem)] aspect-1155/678 w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-sky-400 to-indigo-500 opacity-20 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]"
          />
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          {/* Badge */}
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-semibold bg-sky-50 text-sky-700 border border-sky-200 mb-6 shadow-xs">
            <Sparkles className="w-3.5 h-3.5 text-sky-600" />
            <span>Clinical Screening & Decision Support System</span>
          </div>

          {/* Headline */}
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-slate-900 tracking-tight max-w-4xl mx-auto leading-[1.15]">
            AI-Powered Skin Lesion Detection
          </h1>

          {/* Subtitle */}
          <p className="mt-5 text-lg sm:text-xl text-slate-600 max-w-2xl mx-auto leading-relaxed">
            Upload a dermoscopy image — get instant analysis with explainable AI
          </p>

          {/* CTA Buttons */}
          <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-4 max-w-md mx-auto">
            <Link
              to="/register"
              className="w-full sm:w-auto inline-flex items-center justify-center space-x-2 px-6 py-3.5 rounded-xl bg-sky-600 hover:bg-sky-700 active:bg-sky-800 text-white font-semibold text-base shadow-lg shadow-sky-600/25 hover:shadow-sky-600/40 transition-all cursor-pointer"
            >
              <span>Get Started (Patient)</span>
              <ArrowRight className="w-4 h-4" />
            </Link>

            <Link
              to="/login?role=clinician"
              className="w-full sm:w-auto inline-flex items-center justify-center space-x-2 px-6 py-3.5 rounded-xl bg-white hover:bg-slate-50 text-slate-700 font-semibold text-base border border-slate-300 shadow-xs hover:border-slate-400 transition-all cursor-pointer"
            >
              <Stethoscope className="w-4 h-4 text-sky-600" />
              <span>Clinician Login</span>
            </Link>
          </div>

          {/* Medical Disclaimer Banner */}
          <div className="mt-10 max-w-2xl mx-auto p-3.5 rounded-xl bg-amber-50/80 border border-amber-200/80 text-amber-900 text-xs text-center flex items-center justify-center space-x-2 shadow-xs">
            <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0" />
            <span>
              <strong>Clinical Research Aid:</strong> DermAI provides educational screening decision-support, not a definitive medical diagnosis.
            </span>
          </div>
        </div>
      </section>

      {/* 3 Feature Cards */}
      <section className="py-12 bg-white border-y border-slate-200/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-10">
            <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight">
              State-of-the-Art Dermatological AI
            </h2>
            <p className="text-sm text-slate-500 mt-2 max-w-xl mx-auto">
              Built on peer-reviewed clinical benchmarks and modern computer vision architectures.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8">
            {/* Feature 1 */}
            <div className="p-6 rounded-2xl bg-slate-50 border border-slate-200/80 hover:border-sky-300 hover:shadow-md transition-all flex flex-col justify-between">
              <div>
                <div className="w-12 h-12 rounded-xl bg-sky-100 text-sky-600 flex items-center justify-center mb-4 text-2xl">
                  🔬
                </div>
                <h3 className="text-lg font-bold text-slate-900 mb-2">
                  7-Class Detection
                </h3>
                <p className="text-sm text-slate-600 leading-relaxed">
                  Evaluates across 7 common pigmented skin lesion categories: <strong>akiec, bcc, bkl, df, mel, nv, vasc</strong> with multi-model ensemble consensus.
                </p>
              </div>
              <div className="mt-4 pt-3 border-t border-slate-200/60 text-xs font-semibold text-sky-700 flex items-center space-x-1">
                <span>Calibrated with Temperature Scaling</span>
              </div>
            </div>

            {/* Feature 2 */}
            <div className="p-6 rounded-2xl bg-slate-50 border border-slate-200/80 hover:border-sky-300 hover:shadow-md transition-all flex flex-col justify-between">
              <div>
                <div className="w-12 h-12 rounded-xl bg-indigo-100 text-indigo-600 flex items-center justify-center mb-4 text-2xl">
                  🧠
                </div>
                <h3 className="text-lg font-bold text-slate-900 mb-2">
                  Grad-CAM Explainability
                </h3>
                <p className="text-sm text-slate-600 leading-relaxed">
                  See what the AI focuses on. Visual attention saliency maps generate heatmap overlays directly on dermoscopic structures for transparent inspection.
                </p>
              </div>
              <div className="mt-4 pt-3 border-t border-slate-200/60 text-xs font-semibold text-indigo-700 flex items-center space-x-1">
                <span>Pixel-Level Feature Attribution</span>
              </div>
            </div>

            {/* Feature 3 */}
            <div className="p-6 rounded-2xl bg-slate-50 border border-slate-200/80 hover:border-sky-300 hover:shadow-md transition-all flex flex-col justify-between">
              <div>
                <div className="w-12 h-12 rounded-xl bg-rose-100 text-rose-600 flex items-center justify-center mb-4 text-2xl">
                  ⚡
                </div>
                <h3 className="text-lg font-bold text-slate-900 mb-2">
                  Instant Risk Assessment
                </h3>
                <p className="text-sm text-slate-600 leading-relaxed">
                  Clear triage recommendations categorizing lesions into <strong>HIGH / MEDIUM / LOW</strong> risk with actionable clinical urgency guidance.
                </p>
              </div>
              <div className="mt-4 pt-3 border-t border-slate-200/60 text-xs font-semibold text-rose-700 flex items-center space-x-1">
                <span>Triage Protocol Aligned</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Row */}
      <section className="py-10 bg-slate-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 text-center divide-y sm:divide-y-0 sm:divide-x divide-slate-800">
            <div className="py-3 sm:py-0">
              <div className="text-3xl sm:text-4xl font-extrabold text-sky-400 tracking-tight">
                10,015
              </div>
              <div className="text-xs uppercase tracking-wider text-slate-400 mt-1 font-medium">
                Training Images (HAM10000)
              </div>
            </div>

            <div className="py-3 sm:py-0">
              <div className="text-3xl sm:text-4xl font-extrabold text-emerald-400 tracking-tight">
                7
              </div>
              <div className="text-xs uppercase tracking-wider text-slate-400 mt-1 font-medium">
                Skin Lesion Classes
              </div>
            </div>

            <div className="py-3 sm:py-0">
              <div className="text-3xl sm:text-4xl font-extrabold text-indigo-400 tracking-tight">
                6
              </div>
              <div className="text-xs uppercase tracking-wider text-slate-400 mt-1 font-medium">
                Deep Learning Models Compared
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
