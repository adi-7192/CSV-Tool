import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import { lightTheme } from '@/styles/antdTheme';
import dayjs from 'dayjs';
import 'dayjs/locale/en';
import locale from 'antd/locale/en_US';
import { useAuthStore } from '@/store/authStore';

// Public pages
import PublicLayout from '@/layouts/PublicLayout';
import LandingPage from '@/pages/public/LandingPage';
import PricingPage from '@/pages/public/PricingPage';
import LoginPage from '@/pages/public/LoginPage';
import SignupPage from '@/pages/public/SignupPage';
import ForgotPasswordPage from '@/pages/public/ForgotPasswordPage';
import ResetPasswordPage from '@/pages/public/ResetPasswordPage';

// App pages (protected)
import AppLayout from '@/components/Layout/AppLayout';
import ProtectedRoute from '@/components/ProtectedRoute';
import Dashboard from '@/pages/Dashboard';
import Workspace from '@/pages/Workspace';
import AIAnalyst from '@/pages/AIAnalyst';
import DataManagement from '@/pages/DataManagement';
import Settings from '@/pages/Settings';
import Profile from '@/pages/Profile';
import Onboarding from '@/pages/Onboarding';
import DataAwareRedirect from '@/components/DataAwareRedirect';

// Admin pages
import AdminLayout from '@/layouts/AdminLayout';
import AdminRoute from '@/components/AdminRoute';
import AdminUsersPage from '@/pages/admin/AdminUsersPage';
import AdminTenantsPage from '@/pages/admin/AdminTenantsPage';
import AdminUsagePage from '@/pages/admin/AdminUsagePage';
import AdminSystemPage from '@/pages/admin/AdminSystemPage';
import AdminMonitoringPage from '@/pages/admin/AdminMonitoringPage';

dayjs.locale('en');

function AppContent() {
  const { initFromStorage } = useAuthStore();
  
  // Initialize auth from storage on app load
  React.useEffect(() => {
    initFromStorage();
  }, [initFromStorage]);

  return (
    <Routes>
          {/* Public Routes */}
          <Route
            path="/"
            element={
              <PublicLayout>
                <LandingPage />
              </PublicLayout>
            }
          />
          <Route
            path="/pricing"
            element={
              <PublicLayout>
                <PricingPage />
              </PublicLayout>
            }
          />
          <Route
            path="/login"
            element={
              <PublicLayout>
                <LoginPage />
              </PublicLayout>
            }
          />
          <Route
            path="/signup"
            element={
              <PublicLayout>
                <SignupPage />
              </PublicLayout>
            }
          />
          <Route
            path="/forgot-password"
            element={
              <PublicLayout>
                <ForgotPasswordPage />
              </PublicLayout>
            }
          />
          <Route
            path="/reset-password"
            element={
              <PublicLayout>
                <ResetPasswordPage />
              </PublicLayout>
            }
          />

          {/* Protected App Routes */}
          <Route
            path="/app/*"
            element={
              <ProtectedRoute allowedRoles={['user', 'admin']}>
                <AppLayout>
                  <Routes>
                    <Route path="onboarding" element={<Onboarding />} />
                    <Route path="dashboard" element={<Dashboard />} />
                    <Route path="workspace" element={<Workspace />} />
                    <Route path="analyst" element={<AIAnalyst />} />
                    <Route path="data-management" element={<DataManagement />} />
                    <Route path="profile" element={<Profile />} />
                    <Route path="settings" element={<Settings />} />
                    <Route path="" element={<DataAwareRedirect />} />
                  </Routes>
                </AppLayout>
              </ProtectedRoute>
            }
          />

          {/* Admin Routes */}
          <Route
            path="/admin/*"
            element={
              <AdminRoute>
                <AdminLayout>
                  <Routes>
                    <Route path="" element={<Navigate to="/admin/users" replace />} />
                    <Route path="users" element={<AdminUsersPage />} />
                    <Route path="tenants" element={<AdminTenantsPage />} />
                    <Route path="usage" element={<AdminUsagePage />} />
                    <Route path="system" element={<AdminSystemPage />} />
                    <Route path="monitoring" element={<AdminMonitoringPage />} />
                  </Routes>
                </AdminLayout>
              </AdminRoute>
            }
          />

          {/* Legacy route redirects for backward compatibility */}
          <Route path="/dashboard" element={<Navigate to="/app/dashboard" replace />} />
          <Route path="/workspace" element={<Navigate to="/app/workspace" replace />} />
          <Route path="/analyst" element={<Navigate to="/app/analyst" replace />} />
          <Route path="/data-management" element={<Navigate to="/app/data-management" replace />} />
          <Route path="/settings" element={<Navigate to="/app/settings" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <ConfigProvider theme={lightTheme} locale={locale}>
      <BrowserRouter>
        <AppContent />
      </BrowserRouter>
    </ConfigProvider>
  );
}
