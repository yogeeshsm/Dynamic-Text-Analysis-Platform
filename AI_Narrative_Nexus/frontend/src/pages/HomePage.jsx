import React, { useState, useEffect } from 'react';
import { Button, Card, Row, Col, Statistic, Alert, Spin, Typography, Divider } from 'antd';
import {
  RocketOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  FileTextOutlined,
} from '@ant-design/icons';
import { getDatasetInfo, runFullAnalysis, getAnalysisStatus } from '../api';

const { Title, Paragraph } = Typography;

function HomePage() {
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadDatasetInfo();
  }, []);

  const loadDatasetInfo = async () => {
    try {
      setLoading(true);
      const response = await getDatasetInfo();
      console.log('Dataset info received:', response.data);
      setDatasetInfo(response.data);
      setError(null);
    } catch (err) {
      console.error('Error loading dataset:', err);
      const errorMessage = err.response?.data?.message 
        || err.response?.data?.error 
        || err.message 
        || 'Failed to load dataset. Please make sure the backend server is running.';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleStartAnalysis = async () => {
    try {
      setAnalyzing(true);
      setError(null);
      
      // Start full analysis
      await runFullAnalysis({
        use_svm: true,
        use_distilbert: false,
        n_topics: 7,
        topic_method: 'lda'
      });
      
      // Poll for status updates
      const interval = setInterval(async () => {
        const statusResponse = await getAnalysisStatus();
        setAnalysisStatus(statusResponse.data);
        
        if (statusResponse.data.status === 'completed' || statusResponse.data.status === 'error') {
          clearInterval(interval);
          setAnalyzing(false);
        }
      }, 2000);
      
    } catch (err) {
      setError(err.response?.data?.error || 'Analysis failed');
      setAnalyzing(false);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '100px' }}>
        <Spin size="large" />
        <p style={{ marginTop: '20px' }}>Loading dataset information...</p>
      </div>
    );
  }

  return (
    <div>
      <div style={{ textAlign: 'center', marginBottom: '40px' }}>
        <Title level={1} style={{ 
          background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          fontSize: '48px',
          fontWeight: 'bold'
        }}>
          📊 AI Narrative Nexus
        </Title>
        <Title level={3} style={{ color: '#555', fontWeight: 'normal' }}>
          Airline Sentiment Analysis Platform
        </Title>
        <Paragraph style={{ fontSize: '16px', maxWidth: '800px', margin: '20px auto', color: '#666' }}>
          Welcome to the Dynamic Text Analysis Platform. This system performs comprehensive 
          sentiment analysis using SVM classifiers, topic modeling, and insight generation on the Twitter US Airline 
          Sentiment Dataset.
        </Paragraph>
      </div>

      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          showIcon
          closable
          style={{ marginBottom: '24px' }}
        />
      )}

      {analysisStatus && (
        <Alert
          message={analysisStatus.message}
          description={`Progress: ${analysisStatus.progress}% - ${analysisStatus.current_step}`}
          type={analysisStatus.status === 'error' ? 'error' : 'info'}
          showIcon
          style={{ marginBottom: '24px' }}
        />
      )}

      {datasetInfo && datasetInfo.exists && (
        <>
          <Card title="📊 Dataset Overview" style={{ marginBottom: '24px' }}>
            <Row gutter={[16, 16]}>
              <Col xs={24} sm={12} md={6}>
                <Card>
                  <Statistic
                    title="Total Records"
                    value={datasetInfo.total_records}
                    prefix={<FileTextOutlined />}
                    valueStyle={{ color: '#3f8600' }}
                  />
                </Card>
              </Col>
              <Col xs={24} sm={12} md={6}>
                <Card>
                  <Statistic
                    title="Columns"
                    value={datasetInfo.columns?.length || 0}
                    valueStyle={{ color: '#1890ff' }}
                  />
                </Card>
              </Col>
              {datasetInfo.airline_distribution && (
                <Col xs={24} sm={12} md={6}>
                  <Card>
                    <Statistic
                      title="Airlines"
                      value={Object.keys(datasetInfo.airline_distribution).length}
                      valueStyle={{ color: '#cf1322' }}
                    />
                  </Card>
                </Col>
              )}
              {datasetInfo.sentiment_distribution && (
                <Col xs={24} sm={12} md={6}>
                  <Card>
                    <Statistic
                      title="Sentiment Categories"
                      value={Object.keys(datasetInfo.sentiment_distribution).length}
                      valueStyle={{ color: '#722ed1' }}
                    />
                  </Card>
                </Col>
              )}
            </Row>
          </Card>

          <Card title="🚀 Quick Start" style={{ marginBottom: '24px' }}>
            <Paragraph>
              Run the complete analysis pipeline with one click. This will perform:
            </Paragraph>
            <ul style={{ marginLeft: '20px', marginBottom: '20px' }}>
              <li>✅ Data Preprocessing (text cleaning, tokenization, lemmatization)</li>
              <li>✅ Sentiment Analysis (VADER + TextBlob)</li>
              <li>✅ Topic Modeling (LDA with 7 topics)</li>
              <li>✅ Insights Generation (word clouds, summaries)</li>
              <li>✅ Report Generation (PDF export)</li>
            </ul>
            <Button
              type="primary"
              size="large"
              icon={<RocketOutlined />}
              onClick={handleStartAnalysis}
              loading={analyzing}
              disabled={analyzing}
              style={{ 
                marginRight: '16px',
                background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                border: 'none',
                height: '48px',
                fontSize: '16px',
                fontWeight: '600',
                borderRadius: '8px',
                boxShadow: '0 4px 15px rgba(79, 172, 254, 0.4)',
                transition: 'all 0.3s ease',
                position: 'relative',
                overflow: 'hidden'
              }}
              onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
              onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
              onMouseDown={(e) => e.currentTarget.style.transform = 'scale(0.95)'}
              onMouseUp={(e) => e.currentTarget.style.transform = 'scale(1)'}
            >
              {analyzing ? 'Analysis in Progress...' : 'Start Full Analysis'}
            </Button>
            <Button
              size="large"
              onClick={loadDatasetInfo}
              disabled={analyzing}
              style={{
                height: '48px',
                fontSize: '16px',
                borderRadius: '8px',
                borderColor: '#4facfe',
                color: '#4facfe',
                transition: 'all 0.3s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(79, 172, 254, 0.3)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = 'none';
              }}
              onMouseDown={(e) => e.currentTarget.style.transform = 'scale(0.95)'}
              onMouseUp={(e) => e.currentTarget.style.transform = 'scale(1)'}
            >
              Refresh Dataset Info
            </Button>
          </Card>

          <Card title="📋 Analysis Features" style={{ borderRadius: '12px', boxShadow: '0 2px 12px rgba(0,0,0,0.08)' }}>
            <Row gutter={[16, 16]}>
              <Col xs={24} md={8}>
                <Card hoverable style={{ 
                  borderRadius: '12px', 
                  border: '2px solid #e8f4ff',
                  transition: 'all 0.3s'
                }}>
                  <div style={{ textAlign: 'center' }}>
                    <FileTextOutlined style={{ fontSize: '48px', color: '#4facfe' }} />
                    <Title level={4}>Text Preprocessing</Title>
                    <Paragraph style={{ color: '#666' }}>
                      Advanced text cleaning with URL removal, tokenization, 
                      stopword filtering, and lemmatization.
                    </Paragraph>
                  </div>
                </Card>
              </Col>
              <Col xs={24} md={8}>
                <Card hoverable style={{ 
                  borderRadius: '12px', 
                  border: '2px solid #f0ffe8',
                  transition: 'all 0.3s'
                }}>
                  <div style={{ textAlign: 'center' }}>
                    <CheckCircleOutlined style={{ fontSize: '48px', color: '#52c41a' }} />
                    <Title level={4}>Sentiment Analysis</Title>
                    <Paragraph style={{ color: '#666' }}>
                      Multi-method sentiment classification using VADER, TextBlob, 
                      and fast SVM classifier for accurate predictions.
                    </Paragraph>
                  </div>
                </Card>
              </Col>
              <Col xs={24} md={8}>
                <Card hoverable style={{ 
                  borderRadius: '12px', 
                  border: '2px solid #f9f0ff',
                  transition: 'all 0.3s'
                }}>
                  <div style={{ textAlign: 'center' }}>
                    <ClockCircleOutlined style={{ fontSize: '48px', color: '#9254de' }} />
                    <Title level={4}>Topic Modeling</Title>
                    <Paragraph>
                      Discover hidden themes using LDA/NMF algorithms with 
                      automatic topic labeling and visualization.
                    </Paragraph>
                  </div>
                </Card>
              </Col>
            </Row>
          </Card>
        </>
      )}

      {datasetInfo && !datasetInfo.exists && (
        <Card 
          style={{ 
            marginTop: '40px',
            background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
            border: 'none',
            minHeight: '400px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderRadius: '20px',
            boxShadow: '0 8px 32px rgba(79, 172, 254, 0.3)'
          }}
        >
          <div style={{ 
            textAlign: 'center', 
            color: 'white',
            padding: '60px 40px'
          }}>
            <div style={{ 
              fontSize: '120px', 
              marginBottom: '30px',
              opacity: 0.95,
              filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.2))',
              animation: 'float 3s ease-in-out infinite'
            }}>
              📊
            </div>
            <Title level={2} style={{ color: 'white', marginBottom: '20px', fontWeight: '600' }}>
              Ready to Analyze Airline Sentiments?
            </Title>
            <Paragraph style={{ 
              fontSize: '18px', 
              color: 'rgba(255,255,255,0.95)',
              maxWidth: '600px',
              margin: '0 auto 30px'
            }}>
              Upload your dataset to get started with comprehensive sentiment analysis, 
              topic modeling, and automated insights generation.
            </Paragraph>
            <div style={{ marginTop: '40px' }}>
              <Button 
                type="default" 
                size="large"
                style={{ 
                  marginRight: '15px',
                  background: 'white',
                  color: '#4facfe',
                  borderColor: 'white',
                  fontWeight: 'bold',
                  height: '45px',
                  fontSize: '16px',
                  borderRadius: '8px',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                  transition: 'all 0.3s ease'
                }}
                onClick={() => window.location.href = '/upload'}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-3px)';
                  e.currentTarget.style.boxShadow = '0 6px 20px rgba(0,0,0,0.2)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
                }}
                onMouseDown={(e) => e.currentTarget.style.transform = 'translateY(0) scale(0.98)'}
                onMouseUp={(e) => e.currentTarget.style.transform = 'translateY(-3px) scale(1)'}
              >
                Upload Dataset
              </Button>
              <Button 
                size="large"
                style={{ 
                  background: 'rgba(255,255,255,0.25)',
                  color: 'white',
                  borderColor: 'rgba(255,255,255,0.5)',
                  height: '45px',
                  fontSize: '16px',
                  borderRadius: '8px',
                  transition: 'all 0.3s ease'
                }}
                onClick={loadDatasetInfo}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(255,255,255,0.35)';
                  e.currentTarget.style.transform = 'translateY(-2px)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(255,255,255,0.25)';
                  e.currentTarget.style.transform = 'translateY(0)';
                }}
                onMouseDown={(e) => e.currentTarget.style.transform = 'scale(0.95)'}
                onMouseUp={(e) => e.currentTarget.style.transform = 'scale(1)'}
              >
                Refresh
              </Button>
            </div>
            <Divider style={{ 
              borderColor: 'rgba(255,255,255,0.3)',
              margin: '50px 0 30px'
            }} />
            <Paragraph style={{ 
              color: 'rgba(255,255,255,0.9)',
              fontSize: '14px'
            }}>
              💡 Place <strong>cleaned_dataset.csv.csv</strong> in the <strong>data/</strong> directory
              or use the upload feature above
            </Paragraph>
          </div>
        </Card>
      )}
    </div>
  );
}

export default HomePage;
