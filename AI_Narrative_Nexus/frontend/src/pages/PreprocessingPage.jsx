import React, { useState } from 'react';
import { Button, Card, Table, Alert, Spin, Typography, Row, Col, Statistic } from 'antd';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { preprocessData } from '../api';

const { Title, Paragraph } = Typography;

function PreprocessingPage() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handlePreprocess = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await preprocessData({ text_column: 'text' });
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Preprocessing failed');
    } finally {
      setLoading(false);
    }
  };

  const previewColumns = [
    {
      title: 'Original Text',
      dataIndex: 'original_text',
      key: 'original_text',
      ellipsis: true,
      width: '45%',
    },
    {
      title: 'Cleaned Text',
      dataIndex: 'cleaned_text',
      key: 'cleaned_text',
      ellipsis: true,
      width: '45%',
    },
  ];

  return (
    <div>
      <Title level={2}>🧼 Data Preprocessing</Title>
      <Paragraph>
        Clean and prepare tweets for analysis by removing URLs, mentions, emojis, 
        and special characters, followed by tokenization, stopword removal, and lemmatization.
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
        <Title level={4}>Preprocessing Pipeline</Title>
        <ul style={{ marginLeft: '20px' }}>
          <li>✅ Remove URLs and web links</li>
          <li>✅ Remove @mentions and hashtag symbols</li>
          <li>✅ Remove emojis and special characters</li>
          <li>✅ Convert to lowercase</li>
          <li>✅ Tokenize text into words</li>
          <li>✅ Remove stopwords</li>
          <li>✅ Apply lemmatization</li>
        </ul>
        <Button
          type="primary"
          size="large"
          onClick={handlePreprocess}
          loading={loading}
          disabled={loading}
          style={{ marginTop: '16px' }}
        >
          {loading ? 'Processing...' : 'Start Preprocessing'}
        </Button>
      </Card>

      {loading && (
        <div style={{ textAlign: 'center', padding: '50px' }}>
          <Spin size="large" />
          <p style={{ marginTop: '20px' }}>Preprocessing data... This may take a few moments.</p>
        </div>
      )}

      {results && !loading && (
        <>
          <Card title="📊 Preprocessing Summary" style={{ marginBottom: '24px' }}>
            <Row gutter={[16, 16]}>
              <Col xs={24} sm={12} md={6}>
                <Statistic
                  title="Total Tweets"
                  value={results.total_records}
                  valueStyle={{ color: '#3f8600' }}
                />
              </Col>
              <Col xs={24} sm={12} md={6}>
                <Statistic
                  title="Avg Words (Original)"
                  value={results.summary?.avg_words_original?.toFixed(2) || 0}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
              <Col xs={24} sm={12} md={6}>
                <Statistic
                  title="Avg Words (Cleaned)"
                  value={results.summary?.avg_words_cleaned?.toFixed(2) || 0}
                  valueStyle={{ color: '#cf1322' }}
                />
              </Col>
              <Col xs={24} sm={12} md={6}>
                <Statistic
                  title="Word Reduction"
                  value={results.summary?.reduction_percentage?.toFixed(2) || 0}
                  suffix="%"
                  valueStyle={{ color: '#722ed1' }}
                />
              </Col>
            </Row>
          </Card>

          <Card title="📝 Text Comparison (Sample)" style={{ marginBottom: '24px' }}>
            <Table
              columns={previewColumns}
              dataSource={results.preview?.map((item, index) => ({ ...item, key: index }))}
              pagination={{ pageSize: 5 }}
              scroll={{ x: true }}
            />
          </Card>

          {results.summary && (
            <Card title="📈 Word Count Distribution">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={[
                    {
                      name: 'Average Words',
                      Original: results.summary.avg_words_original,
                      Cleaned: results.summary.avg_words_cleaned,
                    },
                  ]}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="Original" fill="#8884d8" />
                  <Bar dataKey="Cleaned" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}
        </>
      )}
    </div>
  );
}

export default PreprocessingPage;
