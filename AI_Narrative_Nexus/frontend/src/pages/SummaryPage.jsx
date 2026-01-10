import React, { useState } from 'react';
import { Card, Button, Radio, InputNumber, Alert, Spin, Typography, Divider, Row, Col, Tag } from 'antd';
import { FileTextOutlined, LoadingOutlined, RocketOutlined } from '@ant-design/icons';
import axios from 'axios';

const { Title, Paragraph, Text } = Typography;
const API_BASE_URL = 'http://localhost:5000';

function SummaryPage() {
  const [summaryType, setSummaryType] = useState('extractive');
  const [numSentences, setNumSentences] = useState(3);
  const [maxWords, setMaxWords] = useState(50);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleGenerateSummary = async () => {
    try {
      setLoading(true);
      setError(null);
      setResults(null);

      const endpoint = summaryType === 'extractive' 
        ? `${API_BASE_URL}/api/summary/extractive`
        : `${API_BASE_URL}/api/summary/abstractive`;

      const payload = summaryType === 'extractive'
        ? { num_sentences: numSentences }
        : { max_words: maxWords };

      const response = await axios.post(endpoint, payload);
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Failed to generate summary');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2} style={{
          background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text'
        }}>
          <FileTextOutlined /> Text Summarization
        </Title>
        <Paragraph style={{ color: '#666', fontSize: '15px' }}>
          Generate extractive or abstractive summaries of the analyzed text data.
          Extractive summaries select important sentences, while abstractive summaries
          create new condensed text.
        </Paragraph>
      </div>

      <Card 
        title="Summary Configuration" 
        style={{ 
          marginBottom: '24px',
          borderRadius: '12px',
          boxShadow: '0 2px 12px rgba(0,0,0,0.08)',
          border: '1px solid #e8e8e8'
        }}
        extra={
          <Tag color={summaryType === 'extractive' ? 'cyan' : 'blue'} style={{ fontSize: '13px', padding: '4px 12px' }}>
            {summaryType.toUpperCase()}
          </Tag>
        }
      >
        <Row gutter={24}>
          <Col span={12}>
            <div style={{ marginBottom: '20px' }}>
              <Text strong>Summary Type:</Text>
              <Radio.Group 
                value={summaryType} 
                onChange={(e) => setSummaryType(e.target.value)}
                style={{ display: 'block', marginTop: '10px' }}
              >
                <Radio.Button value="extractive">Extractive Summary</Radio.Button>
                <Radio.Button value="abstractive">Abstractive Summary</Radio.Button>
              </Radio.Group>
              <Paragraph style={{ marginTop: '10px', fontSize: '13px', color: '#666' }}>
                {summaryType === 'extractive' 
                  ? 'Selects the most important sentences from the original text'
                  : 'Generates new condensed text using key phrases and words'}
              </Paragraph>
            </div>
          </Col>

          <Col span={12}>
            {summaryType === 'extractive' ? (
              <div>
                <Text strong>Number of Sentences:</Text>
                <div style={{ marginTop: '10px' }}>
                  <InputNumber
                    min={1}
                    max={10}
                    value={numSentences}
                    onChange={(value) => setNumSentences(value)}
                    style={{ width: '100%' }}
                  />
                </div>
                <Paragraph style={{ marginTop: '10px', fontSize: '13px', color: '#666' }}>
                  Number of sentences to include in each summary
                </Paragraph>
              </div>
            ) : (
              <div>
                <Text strong>Maximum Words:</Text>
                <div style={{ marginTop: '10px' }}>
                  <InputNumber
                    min={20}
                    max={200}
                    value={maxWords}
                    onChange={(value) => setMaxWords(value)}
                    style={{ width: '100%' }}
                  />
                </div>
                <Paragraph style={{ marginTop: '10px', fontSize: '13px', color: '#666' }}>
                  Maximum number of words in the generated summary
                </Paragraph>
              </div>
            )}
          </Col>
        </Row>

        <Divider />

        <Button
          type="primary"
          size="large"
          icon={loading ? <LoadingOutlined /> : <RocketOutlined />}
          onClick={handleGenerateSummary}
          loading={loading}
          disabled={loading}
          block
          style={{
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
          onMouseEnter={(e) => !loading && (e.currentTarget.style.transform = 'translateY(-2px)')}
          onMouseLeave={(e) => !loading && (e.currentTarget.style.transform = 'translateY(0)')}
          onMouseDown={(e) => !loading && (e.currentTarget.style.transform = 'scale(0.98)')}
          onMouseUp={(e) => !loading && (e.currentTarget.style.transform = 'scale(1)')}
        >
          {loading ? 'Generating Summaries...' : 'Generate Summary'}
        </Button>
      </Card>

      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: '24px' }}
        />
      )}

      {loading && (
        <Card>
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <Spin size="large" />
            <p style={{ marginTop: '20px', fontSize: '16px' }}>
              Generating {summaryType} summaries...
            </p>
          </div>
        </Card>
      )}

      {results && !loading && (
        <div>
          <Card title="Overall Summaries by Sentiment" style={{ marginBottom: '24px' }}>
            {Object.entries(results.overall_summaries).map(([sentiment, summary]) => (
              <div key={sentiment} style={{ marginBottom: '20px' }}>
                <Title level={4}>
                  <Tag color={
                    sentiment === 'positive' ? 'green' :
                    sentiment === 'negative' ? 'red' :
                    sentiment === 'neutral' ? 'blue' : 'default'
                  }>
                    {sentiment.toUpperCase()}
                  </Tag>
                </Title>
                <Card size="small" style={{ backgroundColor: '#f9f9f9' }}>
                  <Text>{summary || 'No summary available'}</Text>
                </Card>
              </div>
            ))}
          </Card>

          <Card title="Sample Individual Summaries">
            <Paragraph>
              <Text strong>Total Summaries Generated:</Text> {results.total_summaries}
            </Paragraph>
            <Divider />
            {results.sample_summaries && results.sample_summaries.length > 0 ? (
              results.sample_summaries.map((item, index) => (
                <Card 
                  key={index} 
                  size="small" 
                  style={{ marginBottom: '12px', backgroundColor: '#fafafa' }}
                  title={`Sample ${index + 1}`}
                >
                  <Text>
                    {item[summaryType + '_summary'] || 'No summary available'}
                  </Text>
                </Card>
              ))
            ) : (
              <Text type="secondary">No sample summaries available</Text>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}

export default SummaryPage;
