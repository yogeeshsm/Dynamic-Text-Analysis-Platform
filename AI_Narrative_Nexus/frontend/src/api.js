import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const healthCheck = () => api.get('/health');

export const getDatasetInfo = () => api.get('/dataset/info');

export const uploadDataset = (formData) => {
  return axios.post(`${API_BASE_URL}/dataset/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

export const preprocessData = (params = {}) => api.post('/preprocess', params);

export const analyzeSentiment = (params = {}) => api.post('/sentiment', params);

export const analyzeTopics = (params = {}) => api.post('/topics', params);

export const generateInsights = () => api.post('/insights');

export const generateReport = () => api.post('/report/generate');

export const downloadReport = () => {
  window.open(`${API_BASE_URL}/report/download`, '_blank');
};

export const downloadData = (filename) => {
  window.open(`${API_BASE_URL}/data/download/${filename}`, '_blank');
};

export const getWordCloud = (sentiment) => `${API_BASE_URL}/wordcloud/${sentiment}`;

export const getAnalysisStatus = () => api.get('/status');

export const runFullAnalysis = (params = {}) => api.post('/analysis/full', params);

export default api;
