import React, { useState } from 'react';
import { Button, Card, Alert, Spin, Typography, Row, Col, InputNumber, Select, Table, Tag } from 'antd';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { analyzeTopics } from '../api';

const { Title, Paragraph } = Typography;
const { Option } = Select;

function TopicsPage() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [method, setMethod] = useState('lda');

  const handleAnalyze = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await analyzeTopics({ 
        n_topics: 0,
        method: method 
      });
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Topic modeling failed');
    } finally {
      setLoading(false);
    }
  };

  const prepareTopicDistribution = () => {
    if (!results?.topic_distribution) return [];
    
    return Object.entries(results.topic_distribution)
      .map(([name, value]) => ({ topic: name, count: value }))
      .sort((a, b) => b.count - a.count);
  };

  const topicColumns = [
    {
      title: 'Topic ID',
      dataIndex: 'topic_id',
      key: 'topic_id',
      render: (id) => <Tag color="blue">Topic {id + 1}</Tag>,
    },
    {
      title: 'Topic Label',
      dataIndex: 'topic_label',
      key: 'label',
      render: (label) => <strong>{label}</strong>,
    },
    {
      title: 'Keywords',
      dataIndex: 'keywords',
      key: 'keywords',
      ellipsis: true,
    },
  ];

  const previewColumns = [
    {
      title: 'Text',
      dataIndex: 'cleaned_text',
      key: 'text',
      ellipsis: true,
      width: '60%',
    },
    {
      title: 'Topic',
      dataIndex: 'topic_label',
      key: 'topic',
      render: (topic) => <Tag color="purple">{topic}</Tag>,
    },
    {
      title: 'Topic ID',
      dataIndex: 'dominant_topic',
      key: 'id',
      render: (id) => <Tag>{id}</Tag>,
    },
  ];

  return (
    <div>
      <Title level={2}>🧭 Topic Modeling</Title>
      <Paragraph>
        Discover hidden themes and patterns in tweets using Latent Dirichlet Allocation (LDA) 
        or Non-negative Matrix Factorization (NMF) algorithms.
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
          <Col xs={24} sm={12} md={8}>
            <Paragraph>Number of Topics:</Paragraph>
            <InputNumber
              min={3}
              max={15}
              value={nTopics}
              onChange={setNTopics}
              disabled={loading}
              style={{ width: '100%' }}
            />
          </Col>
          <Col xs={24} sm={12} md={8}>
            <Paragraph>Method:</Paragraph>
            <Select
              value={method}
              onChange={setMethod}
              disabled={loading}
              style={{ width: '100%' }}
            >
              <Option value="lda">LDA (Latent Dirichlet Allocation)</Option>
              <Option value="nmf">NMF (Non-negative Matrix Factorization)</Option>
            </Select>
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
          {loading ? 'Extracting Topics...' : 'Extract Topics'}
        </Button>
      </Card>

      {loading && (
        <div style={{ textAlign: 'center', padding: '50px' }}>
          <Spin size="large" />
          <p style={{ marginTop: '20px' }}>Extracting topics... This may take a few minutes.</p>
        </div>
      )}

      {results && !loading && (
        <>
          <Card title="📚 Discovered Topics" style={{ marginBottom: '24px' }}>
            <Table
              columns={topicColumns}
              dataSource={results.topics?.map((item) => ({ ...item, key: item.topic_id }))}
              pagination={false}
            />
          </Card>

          <Card title="📊 Topic Distribution" style={{ marginBottom: '24px' }}>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart 
                data={prepareTopicDistribution()}
                margin={{ top: 20, right: 30, left: 20, bottom: 80 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="topic" 
                  angle={-45}
                  textAnchor="end"
                  height={100}
                  interval={0}
                />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="count" fill="#722ed1" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="📝 Sample Topic Assignments" style={{ marginBottom: '24px' }}>
            <Table
              columns={previewColumns}
              dataSource={results.preview?.map((item, index) => ({ ...item, key: index }))}
              pagination={{ pageSize: 10 }}
              scroll={{ x: true }}
            />
          </Card>

          <Card title="💡 Topic Insights">
            <Row gutter={[16, 16]}>
              {results.topics?.slice(0, 6).map((topic) => (
                <Col xs={24} md={12} lg={8} key={topic.topic_id}>
                  <Card 
                    hoverable
                    style={{ 
                      background: `linear-gradient(135deg, ${getTopicColor(topic.topic_id)} 0%, ${getTopicColor(topic.topic_id)}dd 100%)`,
                      color: 'white'
                    }}
                  >
                    <Title level={5} style={{ color: 'white' }}>
                      {topic.topic_label}
                    </Title>
                    <Paragraph style={{ color: 'white', fontSize: '12px' }}>
                      <strong>Top Keywords:</strong><br />
                      {topic.keywords?.split(', ').slice(0, 8).join(', ')}
                    </Paragraph>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        </>
      )}
    </div>
  );
}

// Helper function to get color for topic
const getTopicColor = (topicId) => {
  const colors = [
    '#1890ff', '#52c41a', '#faad14', '#f5222d', 
    '#722ed1', '#13c2c2', '#eb2f96', '#fa8c16'
  ];
  return colors[topicId % colors.length];
};

export default TopicsPage;
