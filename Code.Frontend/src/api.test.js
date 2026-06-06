import { describe, expect, it, vi } from 'vitest';
import {
  buildInteractionPayload,
  buildRecommendationPayload,
  buildUploadParams,
  interactionActions,
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
    })).toEqual({
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
});
