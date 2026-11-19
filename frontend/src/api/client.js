/**
 * Axios API client for DermAI backend communication.
 * Manages JWT tokens in localStorage and intercepts authentication errors.
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

// Request interceptor: Attach JWT Bearer token if present
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('dermai_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor: Redirect on 401 Unauthorized
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      const isAuthPath =
        window.location.pathname === '/login' || window.location.pathname === '/register';
      if (!isAuthPath) {
        localStorage.removeItem('dermai_token');
        localStorage.removeItem('dermai_user');
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

export const authApi = {
  login: async (credentials) => {
    const response = await apiClient.post('/auth/login', credentials);
    return response.data;
  },
  register: async (userData) => {
    const response = await apiClient.post('/auth/register', userData);
    return response.data;
  },
  getMe: async () => {
    const response = await apiClient.get('/users/me');
    return response.data;
  },
};

export const casesApi = {
  getCases: async () => {
    const response = await apiClient.get('/cases');
    return response.data;
  },
  getCase: async (caseId) => {
    const response = await apiClient.get(`/cases/${caseId}`);
    return response.data;
  },
  createCase: async (caseData) => {
    const response = await apiClient.post('/cases', caseData);
    return response.data;
  },
  uploadImage: async (caseId, formData) => {
    const response = await apiClient.post(`/cases/${caseId}/images`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
  getCaseImages: async (caseId) => {
    const response = await apiClient.get(`/cases/${caseId}/images`);
    return response.data;
  },
  getImageBlobUrl: async (caseId, imageId) => {
    const response = await apiClient.get(`/cases/${caseId}/images/${imageId}/content`, {
      responseType: 'blob',
    });
    return URL.createObjectURL(response.data);
  },
  analyzeImage: async (imageId) => {
    const response = await apiClient.post(`/images/${imageId}/analyze`);
    return response.data;
  },
  getImagePredictions: async (imageId) => {
    const response = await apiClient.get(`/images/${imageId}/predictions`);
    return response.data;
  },
  getGradcamBlobUrl: async (imageId, type = 'overlay') => {
    const response = await apiClient.get(`/images/${imageId}/gradcam`, {
      params: { type },
      responseType: 'blob',
    });
    return URL.createObjectURL(response.data);
  },
  getPatientHistory: async (patientId = 'me', page = 1, limit = 10) => {
    const response = await apiClient.get(`/patients/${patientId}/history`, {
      params: { page, limit },
    });
    return response.data;
  },
};

export const historyApi = {
  getHistory: async (patientId = 'me', page = 1, limit = 10) => {
    const response = await apiClient.get(`/patients/${patientId}/history`, {
      params: { page, limit },
    });
    return response.data;
  },
};

export const dashboardApi = {
  getStats: async () => {
    const response = await apiClient.get('/dashboard/stats');
    return response.data;
  },
  getClassDistribution: async () => {
    const response = await apiClient.get('/dashboard/class-distribution');
    return response.data;
  },
  getConfidenceDistribution: async () => {
    const response = await apiClient.get('/dashboard/confidence-distribution');
    return response.data;
  },
  getRecentCases: async () => {
    const response = await apiClient.get('/dashboard/recent-cases');
    return response.data;
  },
  getCasesOverTime: async () => {
    const response = await apiClient.get('/dashboard/cases-over-time');
    return response.data;
  },
};

export const imagesApi = {
  analyzeImage: async (imageId) => {
    const response = await apiClient.post(`/images/${imageId}/analyze`);
    return response.data;
  },
  getImagePredictions: async (imageId) => {
    const response = await apiClient.get(`/images/${imageId}/predictions`);
    return response.data;
  },
  getGradcamBlobUrl: async (imageId, type = 'overlay') => {
    const response = await apiClient.get(`/images/${imageId}/gradcam`, {
      params: { type },
      responseType: 'blob',
    });
    return URL.createObjectURL(response.data);
  },
};

export default apiClient;
