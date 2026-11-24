import axios from 'axios';

// Create axios instance with base configuration. Do not set a default
// Content-Type: axios already sends JSON for plain-object bodies, and a JSON
// default makes axios serialize FormData uploads to JSON instead of multipart.
export const api = axios.create({
  baseURL: '',
  timeout: 30000,
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

// crypto.randomUUID exists only in secure contexts (HTTPS or localhost); plain
// HTTP hosts such as an ingress without TLS fall back to getRandomValues.
export const randomUuid = () => {
  const cryptoApi = globalThis.crypto;
  if (typeof cryptoApi?.randomUUID === 'function') {
    return cryptoApi.randomUUID();
  }
  const bytes = new Uint8Array(16);
  if (typeof cryptoApi?.getRandomValues === 'function') {
    cryptoApi.getRandomValues(bytes);
  } else {
    for (let index = 0; index < bytes.length; index += 1) {
      bytes[index] = Math.floor(Math.random() * 256);
    }
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
};

// Used until GET /api/content/upload-limits answers (or when it is unavailable).
// Keep in sync with the gateway's DataConfig defaults.
export const DEFAULT_UPLOAD_LIMITS = Object.freeze({
  max_file_size: 100 * 1024 * 1024,
  allowed_extensions: Object.freeze(['.avi', '.mkv', '.mov', '.mp4', '.webm']),
  allowed_mime_types: Object.freeze([
    'video/mp4',
    'video/quicktime',
    'video/webm',
    'video/x-matroska',
    'video/x-msvideo',
  ]),
});

// Browsers leave File.type empty for some containers; the gateway rejects an
// empty (application/octet-stream) part, so uploads declare the type implied
// by the extension instead.
const MIME_TYPE_BY_EXTENSION = Object.freeze({
  '.avi': 'video/x-msvideo',
  '.mkv': 'video/x-matroska',
  '.mov': 'video/quicktime',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
});

const fileExtension = (fileName) => {
  const match = /\.[^./\\]+$/.exec(String(fileName || '').toLowerCase());
  return match ? match[0] : '';
};

export const uploadMimeType = (file) => file?.type || MIME_TYPE_BY_EXTENSION[fileExtension(file?.name)] || '';

export const normalizeUploadLimits = (limits) => {
  const maxFileSize = Number(limits?.max_file_size);
  const extensions = Array.isArray(limits?.allowed_extensions)
    ? limits.allowed_extensions.map((value) => String(value).toLowerCase())
    : [];
  const mimeTypes = Array.isArray(limits?.allowed_mime_types)
    ? limits.allowed_mime_types.map((value) => String(value).toLowerCase())
    : [];
  return {
    max_file_size: Number.isFinite(maxFileSize) && maxFileSize > 0
      ? maxFileSize
      : DEFAULT_UPLOAD_LIMITS.max_file_size,
    allowed_extensions: extensions.length ? extensions : [...DEFAULT_UPLOAD_LIMITS.allowed_extensions],
    allowed_mime_types: mimeTypes.length ? mimeTypes : [...DEFAULT_UPLOAD_LIMITS.allowed_mime_types],
  };
};

// Statuses after which a content job will never complete; polling stops on
// these. Matches the gateway's TERMINAL_FAILED_CONTENT_STATUSES.
const TERMINAL_FAILED_CONTENT_STATUSES = new Set(['failed', 'failed_missing_object']);

export const isTerminalFailedStatus = (status) => TERMINAL_FAILED_CONTENT_STATUSES.has(
  String(status || '').toLowerCase()
);

// Reusing one key for retries of the same upload lets the gateway replay the
// original content job instead of creating a duplicate.
export const createUploadIdempotencyKey = () => randomUuid();

// 409 codes after which the key can never succeed for this request: the job
// behind it failed, or the key was used for a different request.
const STALE_IDEMPOTENCY_KEY_CODES = new Set(['IDEMPOTENCY_KEY_REUSED', 'IDEMPOTENCY_KEY_JOB_FAILED']);

export const isStaleIdempotencyKeyError = (error) => (
  error?.response?.status === 409
  && STALE_IDEMPOTENCY_KEY_CODES.has(error?.response?.data?.error?.code)
);

// The gateway answers 503 CONTENT_ENQUEUE_PENDING when the upload is durable
// but Kafka has not acknowledged it yet. The job will still be processed.
export const pendingUploadFromError = (error) => {
  const response = error?.response;
  const data = response?.data;
  if (response?.status !== 503 || data?.error?.code !== 'CONTENT_ENQUEUE_PENDING' || !data?.content_id) {
    return null;
  }
  return {
    content_id: data.content_id,
    status: data.status || 'pending_publish',
    message: 'Upload stored; waiting for the processing queue to accept it.',
  };
};

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
  context = {},
  eventId = randomUuid()
) => ({
  event_id: eventId,
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
  uploadVideo: async (file, userId = 'demo_user', priority = 'normal', { idempotencyKey } = {}) => {
    const formData = new FormData();
    const mimeType = uploadMimeType(file);
    formData.append(
      'file',
      mimeType && mimeType !== file.type ? new File([file], file.name, { type: mimeType }) : file
    );

    try {
      const response = await api.post('/api/content/upload', formData, {
        params: buildUploadParams(userId, priority),
        headers: idempotencyKey ? { 'Idempotency-Key': idempotencyKey } : undefined,
        timeout: 60000, // 60 seconds for video upload
      });
      return response.data;
    } catch (error) {
      const pending = pendingUploadFromError(error);
      if (pending) {
        return pending;
      }
      throw error;
    }
  },

  // Limits the gateway enforces for uploads
  getUploadLimits: async () => {
    const response = await api.get('/api/content/upload-limits');
    return normalizeUploadLimits(response.data);
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
  logInteraction: async (userId, productId, action, context = {}, eventId) => {
    const requestData = buildInteractionPayload(userId, productId, action, context, eventId);
    const response = await api.post('/api/interactions', requestData);
    return response.data;
  },

  logViewedImpression: async (impressionId, items, context = {}, viewedAt = Date.now() / 1000) => {
    const response = await api.post('/api/impressions/viewed', {
      impression_id: impressionId,
      viewed_at: viewedAt,
      items,
      context,
    });
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
        } else if (isTerminalFailedStatus(status.status)) {
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
  // Mirrors the gateway checks: size, extension, and declared MIME type.
  // checkSize=false skips the size check while the gateway limits are unknown.
  validateVideoFile: (file, limits = DEFAULT_UPLOAD_LIMITS, { checkSize = true } = {}) => {
    const { max_file_size: maxSize, allowed_extensions: extensions, allowed_mime_types: mimeTypes } = (
      normalizeUploadLimits(limits)
    );
    const fileName = file.name?.toLowerCase() ?? '';
    const extensionList = extensions
      .map((extension) => extension.replace(/^\./, '').toUpperCase())
      .join(', ');

    if (checkSize && file.size > maxSize) {
      return {
        valid: false,
        error: `File size exceeds the ${utils.formatFileSize(maxSize)} upload limit`,
      };
    }

    if (!extensions.some((extension) => fileName.endsWith(extension))) {
      return { valid: false, error: `File type not supported. Please use ${extensionList}.` };
    }

    // An empty type is sent as the type implied by the extension (uploadMimeType).
    const mimeType = uploadMimeType(file).toLowerCase();
    if (!mimeTypes.includes(mimeType)) {
      return {
        valid: false,
        error: `This browser reports the file as "${file.type || 'unknown type'}", which the upload service does not accept.`,
      };
    }

    return { valid: true };
  },

  describeUploadLimits: (limits = DEFAULT_UPLOAD_LIMITS) => {
    const normalized = normalizeUploadLimits(limits);
    const extensions = normalized.allowed_extensions
      .map((extension) => extension.replace(/^\./, '').toUpperCase())
      .join(', ');
    return `${extensions} up to ${utils.formatFileSize(normalized.max_file_size)}.`;
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
