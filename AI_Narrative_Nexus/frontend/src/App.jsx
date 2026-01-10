import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Layout, Menu } from 'antd';
import {
  HomeOutlined,
  UploadOutlined,
  FileTextOutlined,
  SmileOutlined,
  BulbOutlined,
  BarChartOutlined,
  FileSearchOutlined,
  ReadOutlined,
} from '@ant-design/icons';

// Pages
import HomePage from './pages/HomePage';
import UploadPage from './pages/UploadPage';
import PreprocessingPage from './pages/PreprocessingPage';
import SentimentPage from './pages/SentimentPage';
import TopicsPage from './pages/TopicsPage';
import InsightsPage from './pages/InsightsPage';
import ReportsPage from './pages/ReportsPage';
import SummaryPage from './pages/SummaryPage';

const { Sider, Content, Footer } = Layout;

function AppContent() {
  const location = useLocation();

  const menuItems = [
    {
      key: '/',
      icon: <HomeOutlined />,
      label: <Link to="/">Home</Link>,
    },
    {
      key: '/upload',
      icon: <UploadOutlined />,
      label: <Link to="/upload">Upload</Link>,
    },
    {
      key: '/preprocessing',
      icon: <FileTextOutlined />,
      label: <Link to="/preprocessing">Preprocessing</Link>,
    },
    {
      key: '/sentiment',
      icon: <SmileOutlined />,
      label: <Link to="/sentiment">Sentiment Analysis</Link>,
    },
    {
      key: '/topics',
      icon: <BulbOutlined />,
      label: <Link to="/topics">Topic Modeling</Link>,
    },
    {
      key: '/insights',
      icon: <BarChartOutlined />,
      label: <Link to="/insights">Insights</Link>,
    },
    {
      key: '/summary',
      icon: <ReadOutlined />,
      label: <Link to="/summary">Summaries</Link>,
    },
    {
      key: '/reports',
      icon: <FileSearchOutlined />,
      label: <Link to="/reports">Reports</Link>,
    },
  ];

  return (
    <Layout style={{ minHeight: '100vh' }}>
      {/* Left Sidebar */}
      <Sider
        width={250}
        style={{
          background: 'linear-gradient(180deg, #4facfe 0%, #00f2fe 100%)',
          boxShadow: '2px 0 12px rgba(0,0,0,0.1)',
        }}
      >
        {/* Logo/Brand */}
        <div style={{ 
          padding: '24px 16px', 
          textAlign: 'center',
          borderBottom: '1px solid rgba(255,255,255,0.1)',
          marginBottom: '8px'
        }}>
          <div style={{ 
            color: 'white', 
            fontSize: '24px', 
            fontWeight: 'bold',
            marginBottom: '8px'
          }}>
            📊 Narrative Nexus
          </div>
          <div style={{ 
            color: 'rgba(255,255,255,0.8)', 
            fontSize: '12px',
            fontWeight: '300'
          }}>
            Text Analysis Platform
          </div>
        </div>

        {/* Navigation Menu */}
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          style={{ 
            background: 'transparent',
            borderRight: 'none',
            fontSize: '15px'
          }}
        />
      </Sider>

      {/* Main Content Area */}
      <Layout>
        <Content style={{ padding: '24px', background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)' }}>
          <div style={{ 
            maxWidth: '1400px', 
            margin: '0 auto',
            background: '#fff',
            padding: '32px',
            borderRadius: '16px',
            boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
            minHeight: 'calc(100vh - 112px)'
          }}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/upload" element={<UploadPage />} />
              <Route path="/preprocessing" element={<PreprocessingPage />} />
              <Route path="/sentiment" element={<SentimentPage />} />
              <Route path="/topics" element={<TopicsPage />} />
              <Route path="/insights" element={<InsightsPage />} />
              <Route path="/summary" element={<SummaryPage />} />
              <Route path="/reports" element={<ReportsPage />} />
            </Routes>
          </div>
        </Content>
        
        <Footer style={{ textAlign: 'center', background: 'linear-gradient(90deg, #4facfe 0%, #00f2fe 100%)', color: 'white', padding: '16px', fontWeight: '500' }}>
          AI Narrative Nexus ©2025 - Dynamic Text Analysis Platform
        </Footer>
      </Layout>
    </Layout>
  );
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;
