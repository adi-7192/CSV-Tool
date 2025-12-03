import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import { lightTheme } from '@/styles/antdTheme';
import TopBar from '@/components/TopBar';
import Dashboard from '@/pages/Dashboard';
import Workspace from '@/pages/Workspace';
import AIAnalyst from '@/pages/AIAnalyst';
import DataManagement from '@/pages/DataManagement';
import Settings from '@/pages/Settings';
import dayjs from 'dayjs';
import 'dayjs/locale/en';
import locale from 'antd/locale/en_US';

dayjs.locale('en');

export default function App() {
  return (
    <ConfigProvider theme={lightTheme} locale={locale}>
      <BrowserRouter>
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
          <TopBar />
          <div style={{ flex: 1, overflow: 'auto', backgroundColor: '#FFFFFF' }}>
            <Routes>
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/workspace" element={<Workspace />} />
              <Route path="/analyst" element={<AIAnalyst />} />
              <Route path="/data-management" element={<DataManagement />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </div>
        </div>
      </BrowserRouter>
    </ConfigProvider>
  );
}
