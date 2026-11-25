import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
  Activity,
  PlusCircle,
  LayoutDashboard,
  LogOut,
  User as UserIcon,
  Clock,
  FileCheck,
  Info,
  Menu,
  X,
} from 'lucide-react';
import { AboutModal } from './AboutModal';

export const Navbar = () => {
  const { user, isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [aboutOpen, setAboutOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleLogout = () => {
    logout();
    setMobileMenuOpen(false);
    navigate('/login');
  };

  const isActive = (path) => location.pathname === path;

  const closeMobileMenu = () => setMobileMenuOpen(false);

  return (
    <>
      <header className="bg-white/95 backdrop-blur-md border-b border-slate-200 sticky top-0 z-30 transition-all">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          {/* Brand Logo */}
          <Link
            to="/"
            onClick={closeMobileMenu}
            className="flex items-center space-x-3 group"
          >
            {/* Inline Dermoscope SVG (32x32px) */}
            <svg
              width="32"
              height="32"
              viewBox="0 0 32 32"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              className="w-8 h-8 shrink-0 text-[#0EA5E9] group-hover:scale-105 transition-transform"
              aria-label="DermAI Dermoscope Logo"
            >
              <circle
                cx="16"
                cy="16"
                r="14"
                stroke="#0EA5E9"
                strokeWidth="2.25"
              />
              <circle
                cx="16"
                cy="16"
                r="8.5"
                stroke="#0EA5E9"
                strokeWidth="1.5"
              />
              <circle cx="13" cy="13" r="1.5" fill="#0EA5E9" />
              <circle cx="19" cy="13" r="1.5" fill="#0EA5E9" />
              <circle cx="13" cy="19" r="1.5" fill="#0EA5E9" />
              <circle cx="19" cy="19" r="1.5" fill="#0EA5E9" />
            </svg>

            {/* Brand Text */}
            <div className="flex flex-col">
              <div className="flex items-center space-x-2">
                <span className="text-xl font-bold tracking-tight text-slate-900">
                  Derm<span className="text-[#0EA5E9]">AI</span>
                </span>
                <span className="hidden sm:inline-block px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider bg-sky-50 text-sky-700 border border-sky-200 rounded-full">
                  Clinical Support
                </span>
              </div>
              <span className="text-[10px] font-medium tracking-wide text-slate-500 -mt-0.5">
                Skin Analysis
              </span>
            </div>
          </Link>

          {/* Desktop Navigation Links */}
          <div className="hidden md:flex items-center space-x-2 lg:space-x-4">
            {/* About Modal Link */}
            <button
              type="button"
              onClick={() => setAboutOpen(true)}
              className="flex items-center space-x-1.5 px-3 py-2 rounded-lg text-sm font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-colors cursor-pointer"
            >
              <Info className="w-4 h-4 text-sky-600" />
              <span>About DermAI</span>
            </button>

            {isAuthenticated ? (
              <>
                {/* Clinical Triage & Analytics (Clinician/Admin) */}
                {(user?.role === 'clinician' || user?.role === 'admin') && (
                  <>
                    <Link
                      to="/clinician"
                      className={`flex items-center space-x-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                        isActive('/clinician')
                          ? 'bg-sky-50 text-sky-700'
                          : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                      }`}
                    >
                      <FileCheck className="w-4 h-4" />
                      <span>Clinical Triage</span>
                    </Link>
                    <Link
                      to="/dashboard"
                      className={`flex items-center space-x-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                        isActive('/dashboard')
                          ? 'bg-sky-50 text-sky-700'
                          : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                      }`}
                    >
                      <LayoutDashboard className="w-4 h-4" />
                      <span>Analytics</span>
                    </Link>
                  </>
                )}

                {/* My History (Patient role) */}
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

                {/* New Case button */}
                <Link
                  to="/cases/new"
                  className={`flex items-center space-x-1.5 px-3.5 py-2 rounded-lg text-sm font-medium transition-all shadow-xs ${
                    isActive('/cases/new')
                      ? 'bg-sky-700 text-white'
                      : 'bg-sky-600 hover:bg-sky-700 text-white hover:shadow-sky-500/25'
                  }`}
                >
                  <PlusCircle className="w-4 h-4" />
                  <span>New Case</span>
                </Link>

                <div className="h-6 w-px bg-slate-200 mx-1"></div>

                {/* User Profile Badge */}
                <div className="flex items-center space-x-2 pl-1">
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
                  className="p-2 text-slate-500 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors cursor-pointer"
                >
                  <LogOut className="w-5 h-5" />
                </button>
              </>
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
                  className="text-sm font-medium bg-sky-600 hover:bg-sky-700 text-white px-4 py-2 rounded-lg shadow-xs hover:shadow transition-all"
                >
                  Register
                </Link>
              </div>
            )}
          </div>

          {/* Mobile Hamburger Toggle Button */}
          <div className="flex items-center md:hidden space-x-2">
            <button
              type="button"
              onClick={() => setAboutOpen(true)}
              className="p-2 text-slate-600 hover:text-slate-900 rounded-lg"
              title="About DermAI"
            >
              <Info className="w-5 h-5 text-sky-600" />
            </button>
            <button
              type="button"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition-colors cursor-pointer"
              aria-label="Toggle navigation menu"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>

        {/* Mobile Dropdown Menu (< 768px) */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-slate-200 bg-white/95 px-4 pt-3 pb-5 space-y-3 shadow-lg animate-fadeIn">
            {isAuthenticated ? (
              <>
                {/* User info bar */}
                <div className="flex items-center space-x-3 py-2 px-3 bg-slate-50 rounded-xl border border-slate-200">
                  <div className="w-9 h-9 rounded-full bg-sky-100 text-sky-700 flex items-center justify-center font-bold text-sm">
                    {(user?.full_name || user?.email || 'U')[0].toUpperCase()}
                  </div>
                  <div className="text-xs">
                    <div className="font-bold text-slate-800">
                      {user?.full_name || user?.email}
                    </div>
                    <div className="text-slate-500 capitalize">{user?.role || 'User'}</div>
                  </div>
                </div>

                <div className="space-y-1">
                  {(user?.role === 'clinician' || user?.role === 'admin') && (
                    <>
                      <Link
                        to="/clinician"
                        onClick={closeMobileMenu}
                        className={`flex items-center space-x-2.5 px-3 py-2.5 rounded-xl text-sm font-medium ${
                          isActive('/clinician') ? 'bg-sky-50 text-sky-700' : 'text-slate-700 hover:bg-slate-50'
                        }`}
                      >
                        <FileCheck className="w-4 h-4 text-sky-600" />
                        <span>Clinical Triage</span>
                      </Link>
                      <Link
                        to="/dashboard"
                        onClick={closeMobileMenu}
                        className={`flex items-center space-x-2.5 px-3 py-2.5 rounded-xl text-sm font-medium ${
                          isActive('/dashboard') ? 'bg-sky-50 text-sky-700' : 'text-slate-700 hover:bg-slate-50'
                        }`}
                      >
                        <LayoutDashboard className="w-4 h-4 text-sky-600" />
                        <span>Analytics Dashboard</span>
                      </Link>
                    </>
                  )}

                  {user?.role === 'patient' && (
                    <Link
                      to="/history"
                      onClick={closeMobileMenu}
                      className={`flex items-center space-x-2.5 px-3 py-2.5 rounded-xl text-sm font-medium ${
                        isActive('/history') ? 'bg-sky-50 text-sky-700' : 'text-slate-700 hover:bg-slate-50'
                      }`}
                    >
                      <Clock className="w-4 h-4 text-sky-600" />
                      <span>My History</span>
                    </Link>
                  )}

                  <Link
                    to="/cases/new"
                    onClick={closeMobileMenu}
                    className="flex items-center space-x-2.5 px-3 py-2.5 rounded-xl text-sm font-semibold bg-sky-600 text-white shadow-xs"
                  >
                    <PlusCircle className="w-4 h-4" />
                    <span>Create New Case</span>
                  </Link>
                </div>

                <div className="pt-2 border-t border-slate-100 flex items-center justify-between">
                  <button
                    type="button"
                    onClick={() => {
                      setMobileMenuOpen(false);
                      setAboutOpen(true);
                    }}
                    className="flex items-center space-x-1.5 text-xs font-medium text-slate-600 hover:text-slate-900 py-1"
                  >
                    <Info className="w-4 h-4 text-sky-600" />
                    <span>About DermAI</span>
                  </button>

                  <button
                    type="button"
                    onClick={handleLogout}
                    className="flex items-center space-x-1.5 text-xs font-semibold text-rose-600 hover:text-rose-700 py-1"
                  >
                    <LogOut className="w-4 h-4" />
                    <span>Sign Out</span>
                  </button>
                </div>
              </>
            ) : (
              <div className="space-y-2">
                <Link
                  to="/login"
                  onClick={closeMobileMenu}
                  className="block w-full py-2.5 px-4 text-center rounded-xl border border-slate-200 text-sm font-semibold text-slate-700 hover:bg-slate-50"
                >
                  Sign In
                </Link>
                <Link
                  to="/register"
                  onClick={closeMobileMenu}
                  className="block w-full py-2.5 px-4 text-center rounded-xl bg-sky-600 text-white text-sm font-semibold shadow-xs hover:bg-sky-700"
                >
                  Register
                </Link>
                <button
                  type="button"
                  onClick={() => {
                    setMobileMenuOpen(false);
                    setAboutOpen(true);
                  }}
                  className="w-full flex items-center justify-center space-x-2 py-2 text-xs font-medium text-slate-500 hover:text-slate-800"
                >
                  <Info className="w-4 h-4 text-sky-600" />
                  <span>About DermAI Clinical Support</span>
                </button>
              </div>
            )}
          </div>
        )}
      </header>

      {/* About Modal */}
      <AboutModal isOpen={aboutOpen} onClose={() => setAboutOpen(false)} />
    </>
  );
};
