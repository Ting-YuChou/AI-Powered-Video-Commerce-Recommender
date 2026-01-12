import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add timestamp to prevent caching
    config.params = {
      ...config.params,
      _t: Date.now(),
    };
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error);
    
    if (error.response?.status === 500) {
      console.error('Server error - check backend health');
    } else if (error.response?.status === 404) {
      console.error('API endpoint not found');
    } else if (error.code === 'ECONNREFUSED') {
      console.error('Cannot connect to backend server');
    }
    
    return Promise.reject(error);
  }
);

// API Functions
export const videoApi = {
  // Upload and process video
  uploadVideo: async (file, userId = 'demo_user') => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', userId);
    
    const response = await api.post('/api/content/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000, // 60 seconds for video upload
    });
    
    return response.data;
  },
  
  // Get content processing status
  getContentStatus: async (contentId) => {
    const response = await api.get(`/api/content/${contentId}/status`);
    return response.data;
  },
  
  // Get recommendations
  getRecommendations: async (userId, contentId = null, context = {}, k = 10) => {
    const requestData = {
      user_id: userId,
      k: k,
      context: {
        device: 'web',
        timestamp: Date.now(),
        ...context,
      },
    };
    
    if (contentId) {
      requestData.content_id = contentId;
    }
    
    const response = await api.post('/api/recommendations', requestData);
    return response.data;
  },
  
  // Log user interaction
  logInteraction: async (userId, productId, action, context = {}) => {
    const requestData = {
      user_id: userId,
      product_id: productId,
      action: action,
      context: {
        timestamp: Date.now(),
        device: 'web',
        ...context,
      },
    };
    
    const response = await api.post('/api/interactions', requestData);
    return response.data;
  },
};

export const systemApi = {
  // Get system health
  getHealth: async () => {
    const response = await api.get('/health');
    return response.data;
  },
  
  // Get system metrics
  getMetrics: async () => {
    const response = await api.get('/metrics');
    return response.data;
  },
  
  // Get analytics
  getAnalytics: async () => {
    const response = await api.get('/api/analytics');
    return response.data;
  },
};

// Utility functions
export const utils = {
  // Poll for content processing completion
  pollContentStatus: async (contentId, maxAttempts = 30, intervalMs = 2000) => {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const status = await videoApi.getContentStatus(contentId);
        
        if (status.status === 'completed') {
          return { success: true, status };
        } else if (status.status === 'failed') {
          return { success: false, status };
        }
        
        // Wait before next poll
        await new Promise(resolve => setTimeout(resolve, intervalMs));
        
      } catch (error) {
        console.error(`Polling attempt ${attempt + 1} failed:`, error);
        
        if (attempt === maxAttempts - 1) {
          return { success: false, error };
        }
        
        // Wait before retry
        await new Promise(resolve => setTimeout(resolve, intervalMs));
      }
    }
    
    return { success: false, error: 'Timeout' };
  },
  
  // Format file size
  formatFileSize: (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  },
  
  // Validate video file
  validateVideoFile: (file) => {
    const maxSize = 500 * 1024 * 1024; // 500MB
    const allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv'];
    
    if (file.size > maxSize) {
      return { valid: false, error: 'File size exceeds 500MB limit' };
    }
    
    if (!allowedTypes.includes(file.type)) {
      return { valid: false, error: 'File type not supported. Please use MP4, AVI, MOV, or MKV.' };
    }
    
    return { valid: true };
  },
  
  // Generate unique user ID for demo
  generateUserId: () => {
    return `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  },
};

export default api;