import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { ProtectedRoute } from './components/ProtectedRoute';

import { LoginPage } from './pages/LoginPage';
import { RegisterPage } from './pages/RegisterPage';
import { DashboardPage } from './pages/DashboardPage';
import { NewCasePage } from './pages/NewCasePage';
import { UploadPage } from './pages/UploadPage';
import { CaseDetailPage } from './pages/CaseDetailPage';
import { PredictionHistoryPage } from './pages/PredictionHistoryPage';
import { ClinicalReviewPage } from './pages/ClinicalReviewPage';
import { CliniciandashboardPage } from './pages/CliniciandashboardPage';

const RootRedirect = () => {
  const { isAuthenticated, loading } = useAuth();
  if (loading) return null;
  return isAuthenticated ? <Navigate to="/dashboard" replace /> : <Navigate to="/login" replace />;
};

export const App = () => {
  return (
    <AuthProvider>
      <BrowserRouter>
        <div className="min-h-screen flex flex-col bg-slate-50 text-slate-800">
          <Navbar />
          <main className="flex-1">
            <Routes>
              {/* Public Routes */}
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

              {/* Root and Fallback redirects */}
              <Route path="/" element={<RootRedirect />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
          <Footer />
        </div>
      </BrowserRouter>
    </AuthProvider>
  );
};

export default App;
