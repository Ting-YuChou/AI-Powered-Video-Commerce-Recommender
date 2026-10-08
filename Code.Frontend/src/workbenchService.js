import { systemApi, videoApi } from './api';
import {
  createMockInteractionResponse,
  createMockRecommendationResponse,
  createMockStatusResponse,
  createMockUploadResponse,
  mockAnalytics,
  mockHealth,
} from './mockData';

export const WORKBENCH_MODES = {
  LIVE: 'live',
  MOCK: 'mock',
};

const delay = (ms) => new Promise((resolve) => {
  window.setTimeout(resolve, ms);
});

const createMockService = () => ({
  uploadVideo: async (file) => {
    await delay(80);
    return createMockUploadResponse(file);
  },
  getContentStatus: async (contentId) => {
    await delay(80);
    return createMockStatusResponse(contentId);
  },
  getRecommendations: async (userId, contentId, context, k) => {
    await delay(80);
    return createMockRecommendationResponse({ userId, contentId, context, k });
  },
  logInteraction: async (_userId, _productId, action) => {
    await delay(60);
    return createMockInteractionResponse(action);
  },
  logViewedImpression: async (impressionId, items) => {
    await delay(20);
    return { status: 'accepted', impression_id: impressionId, accepted_items: items.length };
  },
  getAnalytics: async () => {
    await delay(40);
    return mockAnalytics;
  },
  getHealth: async () => {
    await delay(40);
    return mockHealth;
  },
});

const liveService = {
  uploadVideo: videoApi.uploadVideo,
  getContentStatus: videoApi.getContentStatus,
  getRecommendations: videoApi.getRecommendations,
  logInteraction: videoApi.logInteraction,
  logViewedImpression: videoApi.logViewedImpression,
  getAnalytics: systemApi.getAnalytics,
  getHealth: systemApi.getHealth,
};

export const createWorkbenchService = (mode = WORKBENCH_MODES.LIVE) => (
  mode === WORKBENCH_MODES.MOCK ? createMockService() : liveService
);
