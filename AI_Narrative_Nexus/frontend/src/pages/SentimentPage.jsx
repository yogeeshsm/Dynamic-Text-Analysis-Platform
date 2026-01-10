import React, { useState } from 'react';
import { Button, Card, Alert, Spin, Typography, Row, Col, Statistic, Switch, Table } from 'antd';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { SmileOutlined, MehOutlined, FrownOutlined } from '@ant-design/icons';
import { analyzeSentiment, getWordCloud } from '../api';

const { Title, Paragraph } = Typography;

const COLORS = {
  positive: '#52c41a',
  neutral: '#faad14',
  negative: '#ff4d4f',
};

function SentimentPage() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  // Use SVM classifier by default (fast & accurate)
  const [useSvm, setUseSvm] = useState(true);

  const handleAnalyze = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await analyzeSentiment({ use_svm: useSvm });
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Sentiment analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const preparePieData = () => {
    if (!results?.sentiment_distribution) return [];
    
    return Object.entries(results.sentiment_distribution).map(([name, value]) => ({
      name: name.charAt(0).toUpperCase() + name.slice(1),
      value,
    }));
  };

  const previewColumns = [
    {
      title: 'Text',
      dataIndex: 'cleaned_text',
      key: 'text',
      ellipsis: true,
      width: '60%',
    },
    {
      title: 'Sentiment',
      dataIndex: 'sentiment_label',
      key: 'sentiment',
      render: (sentiment) => (
        <span style={{ color: COLORS[sentiment], fontWeight: 'bold' }}>
          {sentiment === 'positive' && <SmileOutlined />}
          {sentiment === 'neutral' && <MehOutlined />}
          {sentiment === 'negative' && <FrownOutlined />}
          {' '}{sentiment.toUpperCase()}
        </span>
      ),
    },
    {
      title: 'Score',
      dataIndex: 'sentiment_score',
      key: 'score',
      render: (score) => score?.toFixed(4),
    },
  ];

  return (
    <div>
      <Title level={2}>💬 Sentiment Analysis</Title>
      <Paragraph>
        Analyze the emotional tone of tweets using VADER, TextBlob, and an SVM classifier for faster, accurate predictions.
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

      <Card style={{ marginBottom: '24px' }}>
        <Title level={4}>Configuration</Title>
        <Row gutter={[16, 16]} align="middle">
          <Col>
            <Paragraph style={{ margin: 0 }}>
              Use SVM classifier (fast and accurate, recommended):
            </Paragraph>
          </Col>
          <Col>
            <Switch 
              checked={useSvm} 
              onChange={setUseSvm}
              disabled={loading}
            />
          </Col>
        </Row>
        <Button
          type="primary"
          size="large"
          onClick={handleAnalyze}
          loading={loading}
          disabled={loading}
          style={{ marginTop: '16px' }}
        >
          {loading ? 'Analyzing...' : 'Analyze Sentiment'}
        </Button>
      </Card>

      {loading && (
        <div style={{ textAlign: 'center', padding: '50px' }}>
          <Spin size="large" />
          <p style={{ marginTop: '20px' }}>Analyzing sentiment... This may take a few minutes.</p>
        </div>
      )}

      {results && !loading && (
        <>
          <Card title="📊 Sentiment Statistics" style={{ marginBottom: '24px' }}>
            <Row gutter={[16, 16]}>
              <Col xs={24} sm={8}>
                <Card style={{ background: COLORS.positive, color: 'white' }}>
                  <Statistic
                    title={<span style={{ color: 'white' }}>Positive</span>}
                    value={results.sentiment_distribution?.positive || 0}
                    prefix={<SmileOutlined />}
                    valueStyle={{ color: 'white' }}
                  />
                </Card>
              </Col>
              <Col xs={24} sm={8}>
                <Card style={{ background: COLORS.neutral, color: 'white' }}>
                  <Statistic
                    title={<span style={{ color: 'white' }}>Neutral</span>}
                    value={results.sentiment_distribution?.neutral || 0}
                    prefix={<MehOutlined />}
                    valueStyle={{ color: 'white' }}
                  />
                </Card>
              </Col>
              <Col xs={24} sm={8}>
                <Card style={{ background: COLORS.negative, color: 'white' }}>
                  <Statistic
                    title={<span style={{ color: 'white' }}>Negative</span>}
                    value={results.sentiment_distribution?.negative || 0}
                    prefix={<FrownOutlined />}
                    valueStyle={{ color: 'white' }}
                  />
                </Card>
              </Col>
            </Row>
            <Row style={{ marginTop: '16px' }}>
              <Col span={24}>
                <Statistic
                  title="Average Sentiment Score"
                  value={results.average_sentiment?.toFixed(4) || 0}
                  precision={4}
                  valueStyle={{ 
                    color: results.average_sentiment > 0 ? '#3f8600' : '#cf1322' 
                  }}
                />
              </Col>
            </Row>
          </Card>

          <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
            <Col xs={24} lg={12}>
              <Card title="📈 Sentiment Distribution (Pie Chart)">
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={preparePieData()}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {preparePieData().map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[entry.name.toLowerCase()]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="📊 Sentiment Distribution (Bar Chart)">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={preparePieData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="value" fill="#8884d8">
                      {preparePieData().map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[entry.name.toLowerCase()]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>

          <Card title="📝 Sample Results" style={{ marginBottom: '24px' }}>
            <Table
              columns={previewColumns}
              dataSource={results.preview?.map((item, index) => ({ ...item, key: index }))}
              pagination={{ pageSize: 10 }}
              scroll={{ x: true }}
            />
          </Card>
        </>
      )}
    </div>
  );
}

export default SentimentPage;
