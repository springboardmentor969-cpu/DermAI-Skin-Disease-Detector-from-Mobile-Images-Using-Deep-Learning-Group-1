import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { FileQuestion, ArrowLeft, Home, LayoutDashboard } from 'lucide-react';

export const NotFoundPage = () => {
  const { isAuthenticated, user } = useAuth();

  useEffect(() => {
    document.title = '404 — Page Not Found | DermAI';
  }, []);

  const destination = !isAuthenticated
    ? '/'
    : user?.role === 'clinician' || user?.role === 'admin'
    ? '/clinician'
    : '/dashboard';

  const destinationLabel = !isAuthenticated
    ? 'Back to Homepage'
    : user?.role === 'clinician' || user?.role === 'admin'
    ? 'Back to Clinician Dashboard'
    : 'Back to Dashboard';

  return (
    <div className="min-h-[calc(100vh-12rem)] flex items-center justify-center px-4 sm:px-6 lg:px-8 py-16">
      <div className="max-w-md w-full text-center space-y-6 bg-white p-8 sm:p-10 rounded-2xl border border-slate-200 shadow-sm">
        <div className="w-20 h-20 rounded-2xl bg-sky-50 text-sky-600 flex items-center justify-center mx-auto shadow-inner">
          <FileQuestion className="w-10 h-10" />
        </div>

        <div className="space-y-2">
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
            404 — Page Not Found
          </h1>
          <p className="text-sm text-slate-500 leading-relaxed">
            The page you are looking for does not exist, has been removed, or is temporarily unavailable.
          </p>
        </div>

        <div className="pt-2">
          <Link
            to={destination}
            className="inline-flex items-center justify-center space-x-2 w-full py-3 px-5 rounded-xl bg-sky-600 hover:bg-sky-700 active:bg-sky-800 text-white font-semibold text-sm shadow-md shadow-sky-600/20 hover:shadow-lg transition-all"
          >
            {isAuthenticated ? (
              <LayoutDashboard className="w-4 h-4" />
            ) : (
              <Home className="w-4 h-4" />
            )}
            <span>{destinationLabel}</span>
          </Link>
        </div>
      </div>
    </div>
  );
};
