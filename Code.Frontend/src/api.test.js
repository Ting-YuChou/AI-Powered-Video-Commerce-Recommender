import { describe, expect, it, vi } from 'vitest';
import {
  buildInteractionPayload,
  buildRecommendationPayload,
  buildUploadParams,
  api,
  interactionActions,
  videoApi,
} from './api';

describe('api payload builders', () => {
  it('builds upload params with user id and priority', () => {
    expect(buildUploadParams('user-123', 'high')).toEqual({
      user_id: 'user-123',
      priority: 'high',
    });
  });

  it('builds recommendation payload with k and context', () => {
    vi.spyOn(Date, 'now').mockReturnValue(1234);

    expect(buildRecommendationPayload('user-123', 'content-1', {
      device: 'mobile',
      session_id: 'session-1',
      location: 'home',
    }, 7)).toEqual({
      user_id: 'user-123',
      content_id: 'content-1',
      k: 7,
      context: {
        timestamp: 1234,
        device: 'mobile',
        session_id: 'session-1',
        location: 'home',
      },
    });
  });

  it('builds interaction payload with action enum and context', () => {
    vi.spyOn(Date, 'now').mockReturnValue(5678);

    expect(buildInteractionPayload('user-123', 'product-1', interactionActions.ADD_TO_CART, {
      session_id: 'session-1',
    }, 'event-1')).toEqual({
      event_id: 'event-1',
      user_id: 'user-123',
      product_id: 'product-1',
      action: 'add_to_cart',
      context: {
        timestamp: 5678,
        device: 'web',
        session_id: 'session-1',
      },
    });
  });

  it('preserves recommendation impression attribution context', () => {
    const payload = buildInteractionPayload('user-1', 'product-1', 'click', {
      impression_id: 'imp-1',
      recommendation_position: 2,
      recommendation_ranking_score: 0.81,
      recommendation_source: 'two_tower+content',
      content_id: 'content-1',
      session_id: 'session-1',
      surface: 'recommendations',
    });

    expect(payload.context).toMatchObject({
      impression_id: 'imp-1',
      recommendation_position: 2,
      recommendation_ranking_score: 0.81,
      recommendation_source: 'two_tower+content',
      content_id: 'content-1',
      session_id: 'session-1',
      surface: 'recommendations',
    });
  });
});

describe('api transport', () => {
  const captureRequest = () => {
    const seen = [];
    const original = api.defaults.adapter;
    api.defaults.adapter = async (config) => {
      seen.push(config);
      return { data: {}, status: 200, statusText: 'OK', headers: {}, config };
    };
    return { seen, restore: () => { api.defaults.adapter = original; } };
  };

  it('sends video uploads as multipart FormData, not JSON', async () => {
    const { seen, restore } = captureRequest();
    try {
      const file = new File(['video-bytes'], 'clip.mp4', { type: 'video/mp4' });
      await videoApi.uploadVideo(file, 'user-1', 'high');
    } finally {
      restore();
    }

    expect(seen).toHaveLength(1);
    expect(seen[0].data).toBeInstanceOf(FormData);
    expect(seen[0].data.get('file')).toBeInstanceOf(File);
    expect(String(seen[0].headers.get('Content-Type') || '')).not.toContain('application/json');
  });

  it('still sends object bodies as JSON', async () => {
    const { seen, restore } = captureRequest();
    try {
      await videoApi.getRecommendations('user-1');
    } finally {
      restore();
    }

    expect(String(seen[0].headers.get('Content-Type'))).toContain('application/json');
    expect(JSON.parse(seen[0].data).user_id).toBe('user-1');
  });
});
