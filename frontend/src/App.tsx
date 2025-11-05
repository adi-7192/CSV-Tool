import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import { lightTheme } from '@/styles/antdTheme';
import TopBar from '@/components/TopBar';
import Dashboard from '@/pages/Dashboard';
import DataWorkspace from '@/pages/DataWorkspace';
import AIAnalyst from '@/pages/AIAnalyst';
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
              <Route path="/workspace" element={<DataWorkspace />} />
              <Route path="/analyst" element={<AIAnalyst />} />
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </div>
        </div>
      </BrowserRouter>
    </ConfigProvider>
  );
}
