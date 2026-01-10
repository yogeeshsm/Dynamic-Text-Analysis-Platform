import React, { useState } from 'react';
import { 
  Button, Card, Alert, Spin, Typography, Row, Col, Tabs, Image, Table, 
  Statistic, Descriptions, Tag 
} from 'antd';
import { 
  TrophyOutlined, WarningOutlined, SmileOutlined, FileTextOutlined,
  BarChartOutlined, KeyOutlined, LineChartOutlined, CheckCircleOutlined,
  RocketOutlined, ThunderboltOutlined, FireOutlined, StarOutlined
} from '@ant-design/icons';
import { generateInsights, getWordCloud } from '../api';
import './InsightsPage.css';

const { Title, Paragraph, Text } = Typography;
const { TabPane } = Tabs;

function InsightsPage() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleGenerate = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await generateInsights();
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Insights generation failed');
    } finally {
      setLoading(false);
    }
  };

  const airlineColumns = [
    {
      title: 'Rank',
      dataIndex: 'rank',
      key: 'rank',
      render: (rank) => <strong>#{rank}</strong>,
      sorter: (a, b) => a.rank - b.rank,
    },
    {
      title: 'Airline',
      dataIndex: 'airline',
      key: 'airline',
      render: (airline, record) => (
        record.rank === 1 ? 
          <Text style={{ fontSize: '16px', fontWeight: 'bold', color: '#52c41a' }}>
            <TrophyOutlined /> {airline}
          </Text> :
          <strong>{airline}</strong>
      ),
    },
    {
      title: 'Total Mentions',
      dataIndex: 'total_mentions',
      key: 'mentions',
      sorter: (a, b) => a.total_mentions - b.total_mentions,
    },
    {
      title: 'Positive %',
      dataIndex: 'positive_pct',
      key: 'positive',
      render: (pct) => `${pct?.toFixed(1)}%`,
      sorter: (a, b) => a.positive_pct - b.positive_pct,
    },
    {
      title: 'Negative %',
      dataIndex: 'negative_pct',
      key: 'negative',
      render: (pct) => `${pct?.toFixed(1)}%`,
      sorter: (a, b) => a.negative_pct - b.negative_pct,
    },
    {
      title: 'Avg Sentiment',
      dataIndex: 'avg_sentiment',
      key: 'sentiment',
      render: (score) => (
        <span style={{ color: score > 0 ? '#52c41a' : '#ff4d4f', fontWeight: 'bold' }}>
          {score?.toFixed(4)}
        </span>
      ),
      sorter: (a, b) => a.avg_sentiment - b.avg_sentiment,
    },
  ];

  return (
    <div className="insights-page">
      <Title level={2}>
        <RocketOutlined className="gradient-text" /> Insights & Recommendations
      </Title>
      <Paragraph style={{ fontSize: '16px', color: '#666' }}>
        Generate actionable insights by combining sentiment analysis and topic modeling results. 
        Identify key issues, positive aspects, and recommendations for each airline.
      </Paragraph>

      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          showIcon
          closable
          style={{ marginBottom: '24px', borderRadius: '12px' }}
          onClose={() => setError(null)}
        />
      )}

      <Card style={{ marginBottom: '24px', borderRadius: '12px' }} className="insights-card-enter insights-card-1">
        <Title level={4}>
          <ThunderboltOutlined style={{ color: '#667eea' }} /> Generate Comprehensive Insights
        </Title>
        <Paragraph style={{ fontSize: '15px' }}>
          Click below to analyze sentiment and topic data to generate comprehensive insights, 
          word clouds, keyword analysis, and recommendations.
        </Paragraph>
        <Button
          type="primary"
          size="large"
          onClick={handleGenerate}
          loading={loading}
          disabled={loading}
          className="generate-insights-btn"
          icon={<FireOutlined />}
        >
          {loading ? 'Generating Insights...' : 'Generate Insights'}
        </Button>
      </Card>

      {loading && (
        <div className="insights-spinner loading-overlay" style={{ textAlign: 'center', padding: '80px', background: 'rgba(255,255,255,0.95)', borderRadius: '12px' }}>
          <Spin size="large" />
          <p className="pulse-element" style={{ marginTop: '24px', fontSize: '18px', color: '#667eea', fontWeight: 600 }}>
            🔍 Analyzing data and generating insights... This may take a moment.
          </p>
        </div>
      )}

      {results && !loading && (
        <>
          {results.airline_rankings && (
            <Card 
              title={
                <span>
                  <TrophyOutlined className="card-title-icon" style={{ color: '#faad14' }} />
                  Airline Performance Rankings
                </span>
              } 
              style={{ marginBottom: '24px', borderRadius: '12px' }}
              className="insights-card-enter insights-card-2"
            >
              <Table
                columns={airlineColumns}
                dataSource={results.airline_rankings?.map((item, index) => ({ 
                  ...item, 
                  key: index 
                }))}
                pagination={false}
                className="insights-table"
              />
            </Card>
          )}

          <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
            {results.airline_rankings && results.airline_rankings[0] && (
              <Col xs={24} md={12}>
                <Card 
                  hoverable
                  className="stats-card stats-card-best insights-card-enter insights-card-3"
                  style={{ 
                    color: 'white',
                    minHeight: '220px',
                    borderRadius: '16px',
                    border: 'none'
                  }}
                >
                  <div className="animated-icon">
                    <TrophyOutlined className="trophy-icon" style={{ fontSize: '56px', marginBottom: '16px' }} />
                  </div>
                  <Title level={3} style={{ color: 'white', marginBottom: '8px' }}>
                    🏆 Best Performing
                  </Title>
                  <Title level={2} style={{ color: 'white', marginBottom: '16px' }}>
                    {results.airline_rankings[0].airline}
                  </Title>
                  <Paragraph style={{ color: 'white', fontSize: '16px', marginBottom: '8px' }}>
                    ✅ Positive mentions: <strong>{results.airline_rankings[0].positive_pct?.toFixed(1)}%</strong><br />
                    📊 Avg sentiment: <strong>{results.airline_rankings[0].avg_sentiment?.toFixed(4)}</strong>
                  </Paragraph>
                </Card>
              </Col>
            )}

            {results.airline_rankings && results.airline_rankings[results.airline_rankings.length - 1] && (
              <Col xs={24} md={12}>
                <Card 
                  hoverable
                  className="stats-card stats-card-worst insights-card-enter insights-card-4"
                  style={{ 
                    color: 'white',
                    minHeight: '220px',
                    borderRadius: '16px',
                    border: 'none'
                  }}
                >
                  <div className="animated-icon">
                    <WarningOutlined className="warning-icon" style={{ fontSize: '56px', marginBottom: '16px' }} />
                  </div>
                  <Title level={3} style={{ color: 'white', marginBottom: '8px' }}>
                    ⚠️ Needs Improvement
                  </Title>
                  <Title level={2} style={{ color: 'white', marginBottom: '16px' }}>
                    {results.airline_rankings[results.airline_rankings.length - 1].airline}
                  </Title>
                  <Paragraph style={{ color: 'white', fontSize: '16px', marginBottom: '8px' }}>
                    ❌ Negative mentions: <strong>{results.airline_rankings[results.airline_rankings.length - 1].negative_pct?.toFixed(1)}%</strong><br />
                    📊 Avg sentiment: <strong>{results.airline_rankings[results.airline_rankings.length - 1].avg_sentiment?.toFixed(4)}</strong>
                  </Paragraph>
                </Card>
              </Col>
            )}
          </Row>

          <Card title="☁️ Word Clouds" style={{ marginBottom: '24px' }}>
            <Tabs defaultActiveKey="positive">
              <TabPane tab={<span><SmileOutlined /> Positive</span>} key="positive">
                <div style={{ textAlign: 'center', padding: '20px' }}>
                  <Image
                    src={getWordCloud('positive')}
                    alt="Positive Sentiment Word Cloud"
                    fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                  />
                </div>
              </TabPane>
              <TabPane tab="Neutral" key="neutral">
                <div style={{ textAlign: 'center', padding: '20px' }}>
                  <Image
                    src={getWordCloud('neutral')}
                    alt="Neutral Sentiment Word Cloud"
                    fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                  />
                </div>
              </TabPane>
              <TabPane tab={<span><WarningOutlined /> Negative</span>} key="negative">
                <div style={{ textAlign: 'center', padding: '20px' }}>
                  <Image
                    src={getWordCloud('negative')}
                    alt="Negative Sentiment Word Cloud"
                    fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                  />
                </div>
              </TabPane>
            </Tabs>
          </Card>

          {results.top_issues && (
            <Card title="⚠️ Top Issues by Airline" style={{ marginBottom: '24px' }}>
              <Row gutter={[16, 16]}>
                {Object.entries(results.top_issues).map(([airline, issues]) => (
                  <Col xs={24} md={12} lg={8} key={airline}>
                    <Card hoverable>
                      <Title level={5}>{airline}</Title>
                      <ul>
                        {Object.entries(issues).map(([topic, count]) => (
                          <li key={topic}>
                            <strong>{topic}</strong>: {count} mentions
                          </li>
                        ))}
                      </ul>
                    </Card>
                  </Col>
                ))}
              </Row>
            </Card>
          )}

          {results.top_positives && (
            <Card title="✅ Top Positive Aspects by Airline" style={{ marginBottom: '24px' }}>
              <Row gutter={[16, 16]}>
                {Object.entries(results.top_positives).map(([airline, positives]) => (
                  <Col xs={24} md={12} lg={8} key={airline}>
                    <Card hoverable style={{ borderColor: '#52c41a' }}>
                      <Title level={5}>{airline}</Title>
                      <ul>
                        {Object.entries(positives).map(([topic, count]) => (
                          <li key={topic}>
                            <strong>{topic}</strong>: {count} mentions
                          </li>
                        ))}
                      </ul>
                    </Card>
                  </Col>
                ))}
              </Row>
            </Card>
          )}

          {results.summary_text && (
            <Card title="📝 Summary Report">
              <pre style={{ 
                whiteSpace: 'pre-wrap', 
                fontFamily: 'monospace',
                background: '#f5f5f5',
                padding: '16px',
                borderRadius: '4px',
                fontSize: '12px'
              }}>
                {results.summary_text}
              </pre>
            </Card>
          )}
        </>
      )}
    </div>
  );
}

export default InsightsPage;
