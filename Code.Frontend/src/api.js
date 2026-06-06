import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: '',
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
    return Promise.reject(error);
  }
);

export const interactionActions = {
  VIEW: 'view',
  CLICK: 'click',
  PURCHASE: 'purchase',
  ADD_TO_CART: 'add_to_cart',
  REMOVE_FROM_CART: 'remove_from_cart',
  FAVORITE: 'favorite',
  SHARE: 'share',
};

export const buildUploadParams = (userId = 'demo_user', priority = 'normal') => ({
  user_id: userId,
  priority,
});

export const buildRecommendationPayload = (
  userId,
  contentId = null,
  context = {},
  k = 10
) => {
  const requestData = {
    user_id: userId,
    k,
    context: {
      device: 'web',
      timestamp: Date.now(),
      ...context,
    },
  };

  if (contentId) {
    requestData.content_id = contentId;
  }

  return requestData;
};

export const buildInteractionPayload = (
  userId,
  productId,
  action,
  context = {}
) => ({
  user_id: userId,
  product_id: productId,
  action,
  context: {
    timestamp: Date.now(),
    device: 'web',
    ...context,
  },
});

// API Functions
export const videoApi = {
  // Upload and process video
  uploadVideo: async (file, userId = 'demo_user', priority = 'normal') => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/api/content/upload', formData, {
      params: buildUploadParams(userId, priority),
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
    const requestData = buildRecommendationPayload(userId, contentId, context, k);
    const response = await api.post('/api/recommendations', requestData);
    return response.data;
  },
  
  // Log user interaction
  logInteraction: async (userId, productId, action, context = {}) => {
    const requestData = buildInteractionPayload(userId, productId, action, context);
    const response = await api.post('/api/interactions', requestData);
    return response.data;
  },
};

export const systemApi = {
  // Get system health
  getHealth: async () => {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      if (error.response?.data) {
        return error.response.data;
      }
      throw error;
    }
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
  pollContentStatus: async (
    contentId,
    {
      maxAttempts = 120,
      intervalMs = 3000,
      onStatus = null,
    } = {}
  ) => {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const status = await videoApi.getContentStatus(contentId);
        if (onStatus) {
          onStatus(status, attempt + 1);
        }
        
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
    const maxSize = 100 * 1024 * 1024; // 100MB
    const allowedTypes = new Set([
      'video/mp4',
      'video/quicktime',
      'video/x-msvideo',
      'video/x-matroska',
      'video/webm',
    ]);
    const allowedExtensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm'];
    const fileName = file.name?.toLowerCase() ?? '';
    const hasAllowedExtension = allowedExtensions.some((extension) => fileName.endsWith(extension));
    
    if (file.size > maxSize) {
      return { valid: false, error: 'File size exceeds 100MB limit' };
    }
    
    if (file.type && !allowedTypes.has(file.type) && !hasAllowedExtension) {
      return { valid: false, error: 'File type not supported. Please use MP4, MOV, AVI, MKV, or WEBM.' };
    }
    
    return { valid: true };
  },

  formatDuration: (seconds) => {
    if (!Number.isFinite(seconds) || seconds <= 0) {
      return '—';
    }

    const totalMinutes = Math.floor(seconds / 60);
    const hours = Math.floor(totalMinutes / 60);
    const minutes = totalMinutes % 60;

    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }

    return `${minutes}m`;
  },

  formatPrice: (value, currency = 'USD') => {
    if (!Number.isFinite(Number(value))) {
      return '—';
    }

    try {
      return new Intl.NumberFormat(undefined, {
        style: 'currency',
        currency,
      }).format(Number(value));
    } catch {
      return `$${Number(value).toFixed(2)}`;
    }
  },

  getErrorMessage: (error, fallbackMessage) => {
    const detail = error?.response?.data?.detail;
    const envelopeMessage = error?.response?.data?.error?.message;

    if (typeof detail === 'string' && detail.trim()) {
      return detail;
    }

    if (typeof envelopeMessage === 'string' && envelopeMessage.trim()) {
      return envelopeMessage;
    }

    if (typeof error?.message === 'string' && error.message.trim()) {
      return error.message;
    }

    return fallbackMessage;
  },
  
  // Generate unique user ID for demo
  generateUserId: () => {
    return `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  },
};

export default api;
