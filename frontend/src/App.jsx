import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { AuthProvider } from './context/AuthContext';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { ProtectedRoute } from './components/ProtectedRoute';

import { LandingPage } from './pages/LandingPage';
import { NotFoundPage } from './pages/NotFoundPage';
import { LoginPage } from './pages/LoginPage';
import { RegisterPage } from './pages/RegisterPage';
import { DashboardPage } from './pages/DashboardPage';
import { NewCasePage } from './pages/NewCasePage';
import { UploadPage } from './pages/UploadPage';
import { CaseDetailPage } from './pages/CaseDetailPage';
import { PredictionHistoryPage } from './pages/PredictionHistoryPage';
import { ClinicalReviewPage } from './pages/ClinicalReviewPage';
import { CliniciandashboardPage } from './pages/CliniciandashboardPage';

export const App = () => {
  return (
    <AuthProvider>
      <BrowserRouter>
        <div className="min-h-screen flex flex-col bg-slate-50 text-slate-800">
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#0f172a',
                color: '#f8fafc',
                fontSize: '13px',
                borderRadius: '12px',
                padding: '12px 16px',
                boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.2), 0 8px 10px -6px rgba(0, 0, 0, 0.2)',
              },
              success: {
                iconTheme: {
                  primary: '#10b981',
                  secondary: '#ffffff',
                },
              },
              error: {
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#ffffff',
                },
              },
            }}
          />
          <Navbar />
          <main className="flex-1">
            <Routes>
              {/* Landing Homepage (at /) */}
              <Route path="/" element={<LandingPage />} />

              {/* Public Auth Routes */}
              <Route path="/login" element={<LoginPage />} />
              <Route path="/register" element={<RegisterPage />} />

              {/* Protected Routes */}
              <Route
                path="/dashboard"
                element={
                  <ProtectedRoute>
                    <DashboardPage />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/cases/new"
                element={
                  <ProtectedRoute>
                    <NewCasePage />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/cases/:caseId"
                element={
                  <ProtectedRoute>
                    <CaseDetailPage />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/cases/:caseId/upload"
                element={
                  <ProtectedRoute>
                    <UploadPage />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/history"
                element={
                  <ProtectedRoute>
                    <PredictionHistoryPage />
                  </ProtectedRoute>
                }
              />

              {/* Clinician Only Routes */}
              <Route
                path="/review/:image_id"
                element={
                  <ProtectedRoute allowedRoles={['clinician', 'admin']}>
                    <ClinicalReviewPage />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/clinician"
                element={
                  <ProtectedRoute allowedRoles={['clinician', 'admin']}>
                    <CliniciandashboardPage />
                  </ProtectedRoute>
                }
              />

              {/* 404 Not Found Catch-All */}
              <Route path="*" element={<NotFoundPage />} />
            </Routes>
          </main>
          <Footer />
        </div>
      </BrowserRouter>
    </AuthProvider>
  );
};

export default App;
