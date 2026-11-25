import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Mail, Lock, User as UserIcon, Shield, ArrowRight, AlertCircle } from 'lucide-react';
import { DermoscopeLogo } from '../components/DermoscopeLogo';

export const RegisterPage = () => {
  useEffect(() => {
    document.title = 'Register | DermAI';
  }, []);

  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    password: '',
    role: 'patient',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const { register } = useAuth();
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await register({
        email: formData.email,
        password: formData.password,
        full_name: formData.fullName,
        role: formData.role,
      });
      navigate('/dashboard');
    } catch (err) {
      const detail = err.response?.data?.detail || 'Registration failed. Please check your information.';
      setError(detail);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-[calc(100vh-8rem)] flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-lg bg-white rounded-2xl shadow-xl shadow-slate-200/60 border border-slate-200/80 p-8 sm:p-10 transition-all">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="mx-auto w-12 h-12 rounded-2xl bg-gradient-to-tr from-sky-600 to-cyan-500 flex items-center justify-center shadow-lg shadow-sky-500/25 mb-4">
            <DermoscopeLogo className="w-7 h-7 text-white" size={28} />
          </div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Create your DermAI account</h1>
          <p className="text-sm text-slate-500 mt-1.5">Join the decision-support network for clinical dermatological screening</p>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-50 border border-rose-200 flex items-start space-x-3 text-sm text-rose-800 animate-fadeIn">
            <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">{error}</div>
          </div>
        )}

        {/* Register Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
              Full Name
            </label>
            <div className="relative">
              <UserIcon className="w-5 h-5 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                type="text"
                name="fullName"
                required
                value={formData.fullName}
                onChange={handleChange}
                placeholder="Dr. Jane Doe"
                className="w-full pl-11 pr-4 py-2.5 rounded-xl border border-slate-300 focus:border-sky-500 focus:ring-4 focus:ring-sky-500/10 outline-none text-slate-800 placeholder-slate-400 transition-all text-sm"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
              Email Address
            </label>
            <div className="relative">
              <Mail className="w-5 h-5 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                type="email"
                name="email"
                required
                value={formData.email}
                onChange={handleChange}
                placeholder="clinician@example.com"
                className="w-full pl-11 pr-4 py-2.5 rounded-xl border border-slate-300 focus:border-sky-500 focus:ring-4 focus:ring-sky-500/10 outline-none text-slate-800 placeholder-slate-400 transition-all text-sm"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
              Password
            </label>
            <div className="relative">
              <Lock className="w-5 h-5 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                type="password"
                name="password"
                required
                minLength={6}
                value={formData.password}
                onChange={handleChange}
                placeholder="At least 6 characters"
                className="w-full pl-11 pr-4 py-2.5 rounded-xl border border-slate-300 focus:border-sky-500 focus:ring-4 focus:ring-sky-500/10 outline-none text-slate-800 placeholder-slate-400 transition-all text-sm"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
              Role
            </label>
            <div className="relative">
              <Shield className="w-5 h-5 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
              <select
                name="role"
                value={formData.role}
                onChange={handleChange}
                className="w-full pl-11 pr-4 py-2.5 rounded-xl border border-slate-300 focus:border-sky-500 focus:ring-4 focus:ring-sky-500/10 outline-none text-slate-800 bg-white transition-all text-sm appearance-none cursor-pointer"
              >
                <option value="patient">Patient (View own screening cases)</option>
                <option value="clinician">Clinician (Full clinical review & screening)</option>
                <option value="researcher">Researcher (Clinical datasets & trials)</option>
                <option value="ml_engineer">ML Engineer (Model performance & benchmarks)</option>
              </select>
            </div>
            <p className="text-[11px] text-slate-400 mt-1.5">
              *Administrator access is assigned internally and cannot be self-registered.
            </p>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full mt-4 py-3 px-4 rounded-xl bg-sky-600 hover:bg-sky-700 active:bg-sky-800 text-white font-medium text-sm shadow-md shadow-sky-600/20 hover:shadow-lg hover:shadow-sky-600/30 transition-all flex items-center justify-center space-x-2 disabled:opacity-60 cursor-pointer"
          >
            {loading ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
            ) : (
              <>
                <span>Complete Registration</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </form>

        {/* Footer Link */}
        <div className="mt-8 pt-6 border-t border-slate-100 text-center text-sm text-slate-500">
          Already registered?{' '}
          <Link to="/login" className="font-semibold text-sky-600 hover:text-sky-700 transition-colors">
            Sign In here
          </Link>
        </div>
      </div>
    </div>
  );
};
