import React from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Activity, PlusCircle, LayoutDashboard, LogOut, User as UserIcon, Clock } from 'lucide-react';

export const Navbar = () => {
  const { user, isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const isActive = (path) => location.pathname === path;

  return (
    <header className="bg-white/90 backdrop-blur-md border-b border-slate-200 sticky top-0 z-30 transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand Logo */}
        <Link to="/" className="flex items-center space-x-3 group">
          {/* Inline Dermoscope SVG (32x32px) */}
          <svg
            width="32"
            height="32"
            viewBox="0 0 32 32"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            className="w-8 h-8 shrink-0 text-[#0EA5E9] dark:text-sky-400 group-hover:scale-105 transition-transform"
            aria-label="DermAI Dermoscope Logo"
          >
            {/* Outer circle stroke */}
            <circle
              cx="16"
              cy="16"
              r="14"
              stroke="#0EA5E9"
              strokeWidth="2.25"
              className="stroke-[#0EA5E9] dark:stroke-sky-400"
            />
            {/* Inner circle */}
            <circle
              cx="16"
              cy="16"
              r="8.5"
              stroke="#0EA5E9"
              strokeWidth="1.5"
              className="stroke-[#0EA5E9] dark:stroke-sky-400"
            />
            {/* 4 small dots in a 2x2 grid pattern inside (dermoscopy pattern) */}
            <circle cx="13" cy="13" r="1.5" fill="#0EA5E9" className="fill-[#0EA5E9] dark:fill-sky-400" />
            <circle cx="19" cy="13" r="1.5" fill="#0EA5E9" className="fill-[#0EA5E9] dark:fill-sky-400" />
            <circle cx="13" cy="19" r="1.5" fill="#0EA5E9" className="fill-[#0EA5E9] dark:fill-sky-400" />
            <circle cx="19" cy="19" r="1.5" fill="#0EA5E9" className="fill-[#0EA5E9] dark:fill-sky-400" />
          </svg>

          {/* Brand Text */}
          <div className="flex flex-col">
            <div className="flex items-center space-x-2">
              <span className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">
                Derm<span className="text-[#0EA5E9]">AI</span>
              </span>
              <span className="hidden sm:inline-block px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider bg-sky-50 text-sky-700 dark:bg-sky-950/60 dark:text-sky-300 border border-sky-200 dark:border-sky-800 rounded-full">
                Clinical Support
              </span>
            </div>
            <span className="text-[10px] font-medium tracking-wide text-slate-500 dark:text-slate-400 -mt-0.5">
              Skin Analysis
            </span>
          </div>
        </Link>

        {/* Navigation Links */}
        {isAuthenticated ? (
          <div className="flex items-center space-x-2 sm:space-x-4">
            {/* Dashboard (Clinician and Admin roles only) */}
            {(user?.role === 'clinician' || user?.role === 'admin') && (
              <Link
                to="/dashboard"
                className={`flex items-center space-x-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive('/dashboard')
                    ? 'bg-sky-50 text-sky-700'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <LayoutDashboard className="w-4 h-4" />
                <span className="hidden sm:inline">Dashboard</span>
              </Link>
            )}

            {/* My History (Patient role only) */}
            {user?.role === 'patient' && (
              <Link
                to="/history"
                className={`flex items-center space-x-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive('/history')
                    ? 'bg-sky-50 text-sky-700'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <Clock className="w-4 h-4" />
                <span>My History</span>
              </Link>
            )}

            <Link
              to="/cases/new"
              className={`flex items-center space-x-1.5 px-3.5 py-2 rounded-lg text-sm font-medium transition-all shadow-sm ${
                isActive('/cases/new')
                  ? 'bg-sky-700 text-white'
                  : 'bg-sky-600 hover:bg-sky-700 text-white hover:shadow-sky-500/25'
              }`}
            >
              <PlusCircle className="w-4 h-4" />
              <span>New Case</span>
            </Link>

            <div className="h-6 w-px bg-slate-200 mx-1 hidden sm:block"></div>

            {/* User Profile Badge */}
            <div className="hidden md:flex items-center space-x-2 pl-2">
              <div className="w-8 h-8 rounded-full bg-slate-100 border border-slate-300 flex items-center justify-center text-slate-600">
                <UserIcon className="w-4 h-4" />
              </div>
              <div className="text-left text-xs">
                <div className="font-semibold text-slate-800 leading-tight">
                  {user?.full_name || user?.email?.split('@')[0]}
                </div>
                <div className="text-slate-500 capitalize">{user?.role || 'User'}</div>
              </div>
            </div>

            {/* Logout button */}
            <button
              onClick={handleLogout}
              title="Sign Out"
              className="p-2 text-slate-500 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
            >
              <LogOut className="w-5 h-5" />
            </button>
          </div>
        ) : (
          <div className="flex items-center space-x-3">
            <Link
              to="/login"
              className="text-sm font-medium text-slate-600 hover:text-slate-900 px-3 py-2 rounded-lg transition-colors"
            >
              Sign In
            </Link>
            <Link
              to="/register"
              className="text-sm font-medium bg-sky-600 hover:bg-sky-700 text-white px-4 py-2 rounded-lg shadow-sm hover:shadow transition-all"
            >
              Register
            </Link>
          </div>
        )}
      </div>
    </header>
  );
};
