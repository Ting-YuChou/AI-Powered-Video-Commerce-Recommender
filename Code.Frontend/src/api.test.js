import { describe, expect, it, vi } from 'vitest';
import {
  DEFAULT_UPLOAD_LIMITS,
  buildInteractionPayload,
  buildRecommendationPayload,
  buildUploadParams,
  api,
  interactionActions,
  isStaleIdempotencyKeyError,
  isTerminalFailedStatus,
  normalizeUploadLimits,
  randomUuid,
  uploadMimeType,
  utils,
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

describe('upload helpers', () => {
  const respondWith = (status, data) => {
    const original = api.defaults.adapter;
    const seen = [];
    api.defaults.adapter = async (config) => {
      seen.push(config);
      const response = { data, status, statusText: String(status), headers: {}, config };
      if (status >= 400) {
        const error = new Error(`Request failed with status code ${status}`);
        error.response = response;
        error.config = config;
        throw error;
      }
      return response;
    };
    return { seen, restore: () => { api.defaults.adapter = original; } };
  };

  it('sends the Idempotency-Key header when given', async () => {
    const { seen, restore } = respondWith(200, { content_id: 'c1', status: 'queued' });
    try {
      const file = new File(['video-bytes'], 'clip.mp4', { type: 'video/mp4' });
      await videoApi.uploadVideo(file, 'user-1', 'normal', { idempotencyKey: 'key-1' });
      await videoApi.uploadVideo(file, 'user-1', 'normal');
    } finally {
      restore();
    }

    expect(seen[0].headers.get('Idempotency-Key')).toBe('key-1');
    expect(seen[1].headers.get('Idempotency-Key')).toBeFalsy();
  });

  it('treats 503 CONTENT_ENQUEUE_PENDING as an accepted, pending upload', async () => {
    const { restore } = respondWith(503, {
      error: { code: 'CONTENT_ENQUEUE_PENDING', retryable: true },
      content_id: 'content-7',
      status: 'pending_publish',
    });
    let result;
    try {
      result = await videoApi.uploadVideo(new File(['x'], 'clip.mp4', { type: 'video/mp4' }));
    } finally {
      restore();
    }

    expect(result).toMatchObject({ content_id: 'content-7', status: 'pending_publish' });
  });

  it('still throws other 503 responses', async () => {
    const { restore } = respondWith(503, { detail: 'Content upload Kafka producer is unavailable' });
    try {
      await expect(
        videoApi.uploadVideo(new File(['x'], 'clip.mp4', { type: 'video/mp4' }))
      ).rejects.toThrow('503');
    } finally {
      restore();
    }
  });

  it('normalizes gateway upload limits and falls back to defaults', async () => {
    const { restore } = respondWith(200, {
      max_file_size: 2048,
      allowed_extensions: ['.MP4'],
      allowed_mime_types: ['VIDEO/MP4'],
    });
    try {
      await expect(videoApi.getUploadLimits()).resolves.toEqual({
        max_file_size: 2048,
        allowed_extensions: ['.mp4'],
        allowed_mime_types: ['video/mp4'],
      });
    } finally {
      restore();
    }

    expect(normalizeUploadLimits({ max_file_size: 'nope' })).toEqual({
      ...DEFAULT_UPLOAD_LIMITS,
      allowed_extensions: [...DEFAULT_UPLOAD_LIMITS.allowed_extensions],
      allowed_mime_types: [...DEFAULT_UPLOAD_LIMITS.allowed_mime_types],
    });
  });

  it('validates files against the given limits like the gateway does', () => {
    const limits = { max_file_size: 10, allowed_extensions: ['.mp4'], allowed_mime_types: ['video/mp4'] };
    const file = (name, size, type) => ({ name, size, type });

    expect(utils.validateVideoFile(file('a.mp4', 10, 'video/mp4'), limits)).toEqual({ valid: true });
    expect(utils.validateVideoFile(file('a.mp4', 11, 'video/mp4'), limits)).toEqual({
      valid: false,
      error: 'File size exceeds the 10 Bytes upload limit',
    });
    expect(utils.validateVideoFile(file('a.mp4', 11, 'video/mp4'), limits, { checkSize: false }))
      .toEqual({ valid: true });
    expect(utils.validateVideoFile(file('a.webm', 5, 'video/webm'), limits)).toEqual({
      valid: false,
      error: 'File type not supported. Please use MP4.',
    });
    // An empty type is sent as the extension's type, so it passes.
    expect(utils.validateVideoFile(file('a.mp4', 5, ''), limits)).toEqual({ valid: true });
    expect(utils.describeUploadLimits(limits)).toBe('MP4 up to 10 Bytes.');
  });

  it('names the reported MIME type when only the MIME type is rejected', () => {
    const result = utils.validateVideoFile({ name: 'clip.avi', size: 5, type: 'video/avi' });

    expect(result.valid).toBe(false);
    expect(result.error).toContain('"video/avi"');
    expect(result.error).not.toContain('Please use');
  });

  it('sends files without a browser type with the type implied by the extension', async () => {
    const original = api.defaults.adapter;
    const seen = [];
    api.defaults.adapter = async (config) => {
      seen.push(config);
      return { data: {}, status: 200, statusText: 'OK', headers: {}, config };
    };
    try {
      await videoApi.uploadVideo(new File(['x'], 'clip.MKV', { type: '' }));
      await videoApi.uploadVideo(new File(['x'], 'clip.mp4', { type: 'video/mp4' }));
    } finally {
      api.defaults.adapter = original;
    }

    expect(seen[0].data.get('file').type).toBe('video/x-matroska');
    expect(seen[0].data.get('file').name).toBe('clip.MKV');
    expect(seen[1].data.get('file').type).toBe('video/mp4');
    expect(uploadMimeType({ name: 'clip.unknown', type: '' })).toBe('');
  });

  it('knows every terminal failed status', () => {
    for (const status of ['failed', 'failed_missing_object', 'upload_expired', 'FAILED']) {
      expect(isTerminalFailedStatus(status)).toBe(true);
    }
    for (const status of ['queued', 'pending_publish', 'processing', 'completed', 'awaiting_upload', undefined]) {
      expect(isTerminalFailedStatus(status)).toBe(false);
    }
  });

  it('recognizes 409 responses that make an Idempotency-Key unusable', () => {
    const error = (status, code) => ({ response: { status, data: { error: { code } } } });

    expect(isStaleIdempotencyKeyError(error(409, 'IDEMPOTENCY_KEY_REUSED'))).toBe(true);
    expect(isStaleIdempotencyKeyError(error(409, 'IDEMPOTENCY_KEY_JOB_FAILED'))).toBe(true);
    expect(isStaleIdempotencyKeyError({ response: { status: 409, data: { detail: 'in use' } } })).toBe(false);
    expect(isStaleIdempotencyKeyError(error(400, 'IDEMPOTENCY_KEY_REUSED'))).toBe(false);
    expect(isStaleIdempotencyKeyError(new Error('network'))).toBe(false);
  });

  it('keeps the default limits immutable', () => {
    expect(Object.isFrozen(DEFAULT_UPLOAD_LIMITS.allowed_extensions)).toBe(true);
    expect(Object.isFrozen(DEFAULT_UPLOAD_LIMITS.allowed_mime_types)).toBe(true);
  });

  it('builds UUIDs without crypto.randomUUID (non-secure origins)', () => {
    const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
    const original = globalThis.crypto;
    Object.defineProperty(globalThis, 'crypto', {
      configurable: true,
      value: { getRandomValues: (bytes) => original.getRandomValues(bytes) },
    });
    try {
      const first = randomUuid();
      expect(first).toMatch(uuidPattern);
      expect(randomUuid()).not.toBe(first);
      expect(buildInteractionPayload('u', 'p', 'view').event_id).toMatch(uuidPattern);
    } finally {
      Object.defineProperty(globalThis, 'crypto', { configurable: true, value: original });
    }
  });
});

