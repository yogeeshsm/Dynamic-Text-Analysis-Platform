import React, { useState } from 'react';
import { 
  Card, 
  Upload, 
  Button, 
  message, 
  Row, 
  Col, 
  Input, 
  Tabs, 
  Typography,
  Divider,
  Space
} from 'antd';
import { 
  InboxOutlined, 
  FileTextOutlined, 
  EditOutlined, 
  DatabaseOutlined,
  FileWordOutlined,
  MessageOutlined,
  ReadOutlined,
  FileOutlined
} from '@ant-design/icons';
import { uploadDataset } from '../api';

const { Dragger } = Upload;
const { TextArea } = Input;
const { Title, Paragraph, Text } = Typography;
const { TabPane } = Tabs;

function UploadPage() {
  const [uploading, setUploading] = useState(false);
  const [manualText, setManualText] = useState('');
  const [activeTab, setActiveTab] = useState('upload');

  const uploadProps = {
    name: 'file',
    multiple: false,
    accept: '.csv,.txt,.docx',
    beforeUpload: (file) => {
      const isValidSize = file.size / 1024 / 1024 < 10; // 10MB limit
      if (!isValidSize) {
        message.error('File must be smaller than 10MB!');
        return false;
      }
      
      handleFileUpload(file);
      return false; // Prevent auto upload
    },
    showUploadList: false,
  };

  const handleFileUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    setUploading(true);
    try {
      const response = await uploadDataset(formData);
      message.success(`${file.name} uploaded successfully!`);
      console.log('Upload response:', response.data);
    } catch (error) {
      message.error(`Upload failed: ${error.response?.data?.error || error.message}`);
      console.error('Upload error:', error);
    } finally {
      setUploading(false);
    }
  };

  const handleManualSubmit = async () => {
    if (!manualText.trim()) {
      message.warning('Please enter some text to analyze');
      return;
    }

    setUploading(true);
    try {
      // Send manual text as a text file
      const blob = new Blob([manualText], { type: 'text/plain' });
      const file = new File([blob], 'manual_input.txt', { type: 'text/plain' });
      
      const formData = new FormData();
      formData.append('file', file);

      const response = await uploadDataset(formData);
      message.success('Text submitted successfully!');
      console.log('Manual text response:', response.data);
      setManualText('');
    } catch (error) {
      message.error(`Submission failed: ${error.response?.data?.error || error.message}`);
      console.error('Manual text error:', error);
    } finally {
      setUploading(false);
    }
  };

  const handleUseSample = async () => {
    message.info('Loading sample dataset...');
    // The backend automatically uses cleaned_dataset.csv.csv if no file is uploaded
    message.success('Sample dataset loaded! You can now proceed to Sentiment Analysis.');
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: '40px' }}>
        <Title level={2} style={{ 
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          marginBottom: '16px'
        }}>
          Data Input & Upload
        </Title>
        <Paragraph style={{ fontSize: '16px', color: '#666' }}>
          Upload your text data and uncover key themes and insights
        </Paragraph>
      </div>

      {/* Main Upload Card */}
      <Card style={{ borderRadius: '12px', marginBottom: '24px' }}>
        <Tabs 
          activeKey={activeTab} 
          onChange={setActiveTab}
          size="large"
          centered
        >
          {/* File Upload Tab */}
          <TabPane 
            tab={
              <span>
                <InboxOutlined style={{ marginRight: '8px' }} />
                Upload File
              </span>
            } 
            key="upload"
          >
            <div style={{ padding: '20px 0' }}>
              <Title level={4}>Upload Your Data</Title>
              <Dragger {...uploadProps} style={{ padding: '40px 20px' }}>
                <p className="ant-upload-drag-icon">
                  <InboxOutlined style={{ fontSize: '64px', color: '#667eea' }} />
                </p>
                <p className="ant-upload-text" style={{ fontSize: '18px', fontWeight: '500' }}>
                  Drag and drop your file here, or click to browse
                </p>
                <p className="ant-upload-hint" style={{ fontSize: '14px', color: '#999' }}>
                  Supported formats: TXT, CSV, DOCX (Max 10MB)
                </p>
              </Dragger>

              <Divider>OR</Divider>

              <div style={{ textAlign: 'center' }}>
                <Button 
                  type="primary" 
                  icon={<DatabaseOutlined />}
                  size="large"
                  onClick={handleUseSample}
                  style={{ 
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    border: 'none',
                    height: '48px',
                    fontSize: '16px'
                  }}
                >
                  Use Sample Dataset
                </Button>
                <div style={{ marginTop: '12px', color: '#666' }}>
                  <Text type="secondary">
                    Load the pre-configured Twitter US Airline Sentiment dataset
                  </Text>
                </div>
              </div>
            </div>
          </TabPane>

          {/* Manual Text Entry Tab */}
          <TabPane 
            tab={
              <span>
                <EditOutlined style={{ marginRight: '8px' }} />
                Manual Entry
              </span>
            } 
            key="manual"
          >
            <div style={{ padding: '20px 0' }}>
              <Title level={4}>Enter Text Manually</Title>
              <Paragraph type="secondary">
                Paste or type your text content below for analysis
              </Paragraph>
              
              <TextArea
                value={manualText}
                onChange={(e) => setManualText(e.target.value)}
                placeholder="Enter your text here... You can paste multiple lines, tweets, reviews, or any text content you want to analyze."
                rows={12}
                maxLength={50000}
                showCount
                style={{ 
                  fontSize: '14px',
                  marginBottom: '16px',
                  borderRadius: '8px'
                }}
              />

              <div style={{ textAlign: 'right' }}>
                <Space>
                  <Button 
                    onClick={() => setManualText('')}
                    disabled={!manualText.trim()}
                  >
                    Clear
                  </Button>
                  <Button 
                    type="primary" 
                    icon={<FileTextOutlined />}
                    size="large"
                    onClick={handleManualSubmit}
                    loading={uploading}
                    disabled={!manualText.trim()}
                    style={{ 
                      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      border: 'none',
                      height: '40px'
                    }}
                  >
                    Submit Text for Analysis
                  </Button>
                </Space>
              </div>
            </div>
          </TabPane>
        </Tabs>
      </Card>

      {/* Data Type Cards */}
      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} md={6}>
          <Card 
            hoverable
            style={{ textAlign: 'center', borderRadius: '12px' }}
          >
            <FileWordOutlined style={{ fontSize: '48px', color: '#4285f4' }} />
            <Title level={5} style={{ marginTop: '16px' }}>Documents</Title>
            <Text type="secondary">Supported</Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card 
            hoverable
            style={{ textAlign: 'center', borderRadius: '12px' }}
          >
            <MessageOutlined style={{ fontSize: '48px', color: '#9c27b0' }} />
            <Title level={5} style={{ marginTop: '16px' }}>Social Media</Title>
            <Text type="secondary">Supported</Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card 
            hoverable
            style={{ textAlign: 'center', borderRadius: '12px' }}
          >
            <ReadOutlined style={{ fontSize: '48px', color: '#0f9d58' }} />
            <Title level={5} style={{ marginTop: '16px' }}>Articles</Title>
            <Text type="secondary">Supported</Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card 
            hoverable
            style={{ textAlign: 'center', borderRadius: '12px' }}
          >
            <FileOutlined style={{ fontSize: '48px', color: '#f4b400' }} />
            <Title level={5} style={{ marginTop: '16px' }}>Reports</Title>
            <Text type="secondary">Supported</Text>
          </Card>
        </Col>
      </Row>
    </div>
  );
}

export default UploadPage;
