import React, { useState } from 'react';
import { Button, Card, Alert, Typography, Row, Col, Divider, Space } from 'antd';
import { DownloadOutlined, FilePdfOutlined, FileExcelOutlined } from '@ant-design/icons';
import { generateReport, downloadReport, downloadData } from '../api';

const { Title, Paragraph } = Typography;

function ReportsPage() {
  const [generating, setGenerating] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState(null);

  const handleGenerateReport = async () => {
    try {
      setGenerating(true);
      setError(null);
      setSuccess(false);
      
      await generateReport();
      setSuccess(true);
    } catch (err) {
      setError(err.response?.data?.error || 'Report generation failed');
    } finally {
      setGenerating(false);
    }
  };

  const handleDownloadReport = () => {
    downloadReport();
  };

  const handleDownloadCSV = (filename) => {
    downloadData(filename);
  };

  return (
    <div>
      <Title level={2}>📄 Reports & Export</Title>
      <Paragraph>
        Generate comprehensive PDF reports and download analysis results in various formats.
      </Paragraph>

      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          showIcon
          closable
          style={{ marginBottom: '24px' }}
          onClose={() => setError(null)}
        />
      )}

      {success && (
        <Alert
          message="Success"
          description="Report generated successfully! Click the download button below to get your PDF report."
          type="success"
          showIcon
          closable
          style={{ marginBottom: '24px' }}
          onClose={() => setSuccess(false)}
        />
      )}

      <Card title="📊 PDF Report Generation" style={{ marginBottom: '24px' }}>
        <Paragraph>
          Generate a comprehensive PDF report including:
        </Paragraph>
        <ul style={{ marginLeft: '20px', marginBottom: '20px' }}>
          <li>Executive Summary with key statistics</li>
          <li>Airline Performance Rankings</li>
          <li>Topic Analysis and Keywords</li>
          <li>Sentiment Distribution Charts</li>
          <li>Word Cloud Visualizations</li>
          <li>Key Insights and Recommendations</li>
          <li><strong style={{ color: '#4facfe' }}>✨ Text Summaries</strong> - Extractive and Abstractive summaries by sentiment (if generated)</li>
        </ul>
        <Space>
          <Button
            type="primary"
            size="large"
            icon={<FilePdfOutlined />}
            onClick={handleGenerateReport}
            loading={generating}
            disabled={generating}
          >
            {generating ? 'Generating Report...' : 'Generate PDF Report'}
          </Button>
          
          {success && (
            <Button
              type="default"
              size="large"
              icon={<DownloadOutlined />}
              onClick={handleDownloadReport}
            >
              Download PDF Report
            </Button>
          )}
        </Space>
      </Card>

      <Card title="💾 Download Data Files">
        <Paragraph>
          Download processed data and analysis results in CSV format:
        </Paragraph>
        
        <Divider orientation="left">Processed Data</Divider>
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} md={8}>
            <Card hoverable onClick={() => handleDownloadCSV('clean_tweets.csv')}>
              <div style={{ textAlign: 'center' }}>
                <FileExcelOutlined style={{ fontSize: '48px', color: '#52c41a' }} />
                <Title level={5}>Cleaned Tweets</Title>
                <Paragraph>Preprocessed and cleaned text data</Paragraph>
                <Button type="link" icon={<DownloadOutlined />}>
                  Download CSV
                </Button>
              </div>
            </Card>
          </Col>
          
          <Col xs={24} sm={12} md={8}>
            <Card hoverable onClick={() => handleDownloadCSV('sentiment_results.csv')}>
              <div style={{ textAlign: 'center' }}>
                <FileExcelOutlined style={{ fontSize: '48px', color: '#1890ff' }} />
                <Title level={5}>Sentiment Results</Title>
                <Paragraph>Sentiment scores and classifications</Paragraph>
                <Button type="link" icon={<DownloadOutlined />}>
                  Download CSV
                </Button>
              </div>
            </Card>
          </Col>
          
          <Col xs={24} sm={12} md={8}>
            <Card hoverable onClick={() => handleDownloadCSV('topic_results.csv')}>
              <div style={{ textAlign: 'center' }}>
                <FileExcelOutlined style={{ fontSize: '48px', color: '#722ed1' }} />
                <Title level={5}>Topic Modeling Results</Title>
                <Paragraph>Topic assignments and probabilities</Paragraph>
                <Button type="link" icon={<DownloadOutlined />}>
                  Download CSV
                </Button>
              </div>
            </Card>
          </Col>
        </Row>

        <Divider orientation="left">Analysis Summaries</Divider>
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} md={8}>
            <Card hoverable onClick={() => handleDownloadCSV('topics.csv')}>
              <div style={{ textAlign: 'center' }}>
                <FileExcelOutlined style={{ fontSize: '48px', color: '#faad14' }} />
                <Title level={5}>Topics Summary</Title>
                <Paragraph>Topic labels and keywords</Paragraph>
                <Button type="link" icon={<DownloadOutlined />}>
                  Download CSV
                </Button>
              </div>
            </Card>
          </Col>
          
          <Col xs={24} sm={12} md={8}>
            <Card hoverable onClick={() => handleDownloadCSV('airline_statistics.csv')}>
              <div style={{ textAlign: 'center' }}>
                <FileExcelOutlined style={{ fontSize: '48px', color: '#ff4d4f' }} />
                <Title level={5}>Airline Statistics</Title>
                <Paragraph>Sentiment stats by airline</Paragraph>
                <Button type="link" icon={<DownloadOutlined />}>
                  Download CSV
                </Button>
              </div>
            </Card>
          </Col>
        </Row>
      </Card>

      <Card title="ℹ️ Report Information" style={{ marginTop: '24px' }}>
        <Paragraph>
          <strong>Note:</strong> Make sure to complete the full analysis pipeline before generating reports:
        </Paragraph>
        <ol style={{ marginLeft: '20px' }}>
          <li>Data Preprocessing</li>
          <li>Sentiment Analysis</li>
          <li>Topic Modeling</li>
          <li>Insights Generation</li>
          <li>Report Generation</li>
        </ol>
        <Paragraph style={{ marginTop: '16px' }}>
          You can run the complete pipeline from the Home page with a single click, 
          or execute each step individually from their respective pages.
        </Paragraph>
      </Card>
    </div>
  );
}

export default ReportsPage;
