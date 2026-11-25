import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { casesApi } from '../api/client';
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  AlertCircle,
  MapPin,
  User,
  FileText,
  Sparkles,
} from 'lucide-react';

const COMMON_LOCATIONS = [
  'Left Forearm',
  'Right Forearm',
  'Upper Back',
  'Lower Back',
  'Cheek / Face',
  'Neck',
  'Shoulder',
  'Chest',
  'Abdomen',
  'Scalp',
  'Thigh',
];

export const NewCasePage = () => {
  useEffect(() => {
    document.title = 'New Case | DermAI';
  }, []);

  const [formData, setFormData] = useState({
    bodyLocation: '',
    age: '',
    gender: 'female',
    notes: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  const handleSelectLocation = (loc) => {
    setFormData((prev) => ({
      ...prev,
      bodyLocation: loc,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.bodyLocation.trim()) {
      setError('Body location is required.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const payload = {
        body_location: formData.bodyLocation.trim(),
        patient_age: formData.age ? parseInt(formData.age, 10) : null,
        patient_gender: formData.gender,
        clinical_notes: formData.notes.trim() || null,
        status: 'submitted',
      };

      const newCase = await casesApi.createCase(payload);
      // Navigate directly to image upload for the newly created case
      navigate(`/cases/${newCase.id}/upload`);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to create case. Please verify the information.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8 sm:py-10">
      {/* Top Navigation */}
      <div className="mb-6">
        <Link
          to="/dashboard"
          className="inline-flex items-center space-x-1.5 text-xs font-semibold text-slate-500 hover:text-slate-900 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Dashboard</span>
        </Link>
      </div>

      <div className="bg-white rounded-2xl shadow-xl shadow-slate-200/60 border border-slate-200/80 p-6 sm:p-10 transition-all">
        {/* Header */}
        <div className="flex items-center space-x-3 mb-8 pb-6 border-b border-slate-100">
          <div className="w-12 h-12 rounded-2xl bg-sky-50 text-sky-600 flex items-center justify-center shrink-0 border border-sky-100">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Create Screening Case</h1>
            <p className="text-sm text-slate-500 mt-0.5">
              Enter non-identifying patient demographics and anatomical lesion location
            </p>
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-50 border border-rose-200 flex items-start space-x-3 text-sm text-rose-800 animate-fadeIn">
            <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
            <div className="flex-1">{error}</div>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Body Location */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
              Anatomical Body Location <span className="text-rose-500">*</span>
            </label>
            <div className="relative mb-2.5">
              <MapPin className="w-5 h-5 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                type="text"
                name="bodyLocation"
                required
                value={formData.bodyLocation}
                onChange={handleChange}
                placeholder="e.g. Left Forearm, Upper Back"
                className="w-full pl-11 pr-4 py-2.5 rounded-xl border border-slate-300 focus:border-sky-500 focus:ring-4 focus:ring-sky-500/10 outline-none text-slate-800 placeholder-slate-400 transition-all text-sm"
              />
            </div>

            {/* Quick Location Chips */}
            <div className="flex flex-wrap gap-1.5 pt-1">
              {COMMON_LOCATIONS.map((loc) => (
                <button
                  type="button"
                  key={loc}
                  onClick={() => handleSelectLocation(loc)}
                  className={`text-xs px-2.5 py-1 rounded-lg border transition-colors cursor-pointer ${
                    formData.bodyLocation === loc
                      ? 'bg-sky-50 border-sky-300 text-sky-700 font-semibold'
                      : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100'
                  }`}
                >
                  {loc}
                </button>
              ))}
            </div>
          </div>

          {/* Demographics Row (Age & Gender) */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
                Patient Age (Years)
              </label>
              <div className="relative">
                <User className="w-5 h-5 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                <input
                  type="number"
                  name="age"
                  min="0"
                  max="125"
                  value={formData.age}
                  onChange={handleChange}
                  placeholder="e.g. 45"
                  className="w-full pl-11 pr-4 py-2.5 rounded-xl border border-slate-300 focus:border-sky-500 focus:ring-4 focus:ring-sky-500/10 outline-none text-slate-800 placeholder-slate-400 transition-all text-sm"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
                Gender
              </label>
              <select
                name="gender"
                value={formData.gender}
                onChange={handleChange}
                className="w-full px-4 py-2.5 rounded-xl border border-slate-300 focus:border-sky-500 focus:ring-4 focus:ring-sky-500/10 outline-none text-slate-800 bg-white transition-all text-sm cursor-pointer"
              >
                <option value="female">Female</option>
                <option value="male">Male</option>
                <option value="other">Other / Non-binary</option>
                <option value="prefer_not_to_say">Prefer not to say</option>
              </select>
            </div>
          </div>

          {/* Clinical Notes */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
              Clinical Observations / Symptoms
            </label>
            <div className="relative">
              <textarea
                name="notes"
                rows={3}
                value={formData.notes}
                onChange={handleChange}
                placeholder="Mention any evolution, itching, bleeding, or relevant medical history..."
                className="w-full p-3.5 rounded-xl border border-slate-300 focus:border-sky-500 focus:ring-4 focus:ring-sky-500/10 outline-none text-slate-800 placeholder-slate-400 transition-all text-sm"
              ></textarea>
            </div>
            <p className="text-[11px] text-slate-400 mt-1">
              Strictly non-identifying: please do not include patient names, addresses, or identifiers.
            </p>
          </div>

          {/* Form Actions */}
          <div className="pt-4 flex items-center justify-end space-x-3 border-t border-slate-100">
            <Link
              to="/dashboard"
              className="px-4 py-2.5 rounded-xl border border-slate-200 text-slate-600 hover:bg-slate-50 text-sm font-semibold transition-colors"
            >
              Cancel
            </Link>

            <button
              type="submit"
              disabled={loading}
              className="py-2.5 px-5 rounded-xl bg-sky-600 hover:bg-sky-700 active:bg-sky-800 text-white font-medium text-sm shadow-md shadow-sky-600/20 hover:shadow-lg hover:shadow-sky-600/30 transition-all flex items-center space-x-2 disabled:opacity-60 cursor-pointer"
            >
              {loading ? (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
              ) : (
                <>
                  <span>Save &amp; Proceed to Upload</span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
