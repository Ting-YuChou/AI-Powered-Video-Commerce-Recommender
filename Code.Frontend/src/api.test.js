import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  DEFAULT_UPLOAD_LIMITS,
  buildInteractionPayload,
  buildRecommendationPayload,
  buildUploadParams,
  api,
  interactionActions,
  isDirectUploadUnavailableError,
  isRetryableCompleteError,
  isStaleIdempotencyKeyError,
  isTerminalFailedStatus,
  normalizeUploadLimits,
  randomUuid,
  storageUploadTimeoutMs,
  uploadMimeType,
  uploadToStorage,
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
        direct_upload_enabled: false,
      });
    } finally {
      restore();
    }

    expect(normalizeUploadLimits({ max_file_size: 'nope' })).toEqual({
      ...DEFAULT_UPLOAD_LIMITS,
      allowed_extensions: [...DEFAULT_UPLOAD_LIMITS.allowed_extensions],
      allowed_mime_types: [...DEFAULT_UPLOAD_LIMITS.allowed_mime_types],
    });
    expect(normalizeUploadLimits({ direct_upload_enabled: true }).direct_upload_enabled).toBe(true);
    expect(normalizeUploadLimits({ direct_upload_enabled: 'yes' }).direct_upload_enabled).toBe(false);
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
    expect(isStaleIdempotencyKeyError(error(409, 'DIRECT_UPLOAD_URL_EXPIRED'))).toBe(true);
    expect(isStaleIdempotencyKeyError(error(409, 'CONTENT_JOB_FAILED'))).toBe(true);
    expect(isStaleIdempotencyKeyError(error(410, 'DIRECT_UPLOAD_EXPIRED'))).toBe(true);
    // Retryable /complete answers keep the key.
    expect(isStaleIdempotencyKeyError(error(409, 'DIRECT_UPLOAD_OBJECT_MISSING'))).toBe(false);
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


// Fake XMLHttpRequest for presigned POSTs: records the request and answers
// with the status the test chooses, after reporting progress.
const installFakeXhr = ({
  status = 204,
  progress = [],
  fail = false,
  timeout = false,
  responseText = '<Error><Code>EntityTooLarge</Code></Error>',
} = {}) => {
  const requests = [];
  class FakeXhr {
    constructor() {
      this.upload = {};
      requests.push(this);
    }

    open(method, url) {
      this.method = method;
      this.url = url;
    }

    send(body) {
      this.body = body;
      setTimeout(() => {
        for (const [loaded, total] of progress) {
          this.upload.onprogress?.({ loaded, total, lengthComputable: true });
        }
        if (fail) {
          this.onerror?.();
          return;
        }
        if (timeout) {
          this.ontimeout?.();
          return;
        }
        this.status = status;
        this.responseText = status < 300 ? '' : responseText;
        this.onload?.();
      }, 0);
    }
  }
  vi.stubGlobal('XMLHttpRequest', FakeXhr);
  return requests;
};

describe('direct uploads', () => {
  const ticket = {
    content_id: 'content-1',
    status: 'awaiting_upload',
    upload: { url: 'http://127.0.0.1:9000/bucket', fields: { key: 'uploads/incoming/content-1.mp4', policy: 'p' } },
    complete_path: '/api/content/uploads/content-1/complete',
  };

  // Answers gateway calls in order with [status, data] pairs.
  const gatewayResponses = (...responses) => {
    const original = api.defaults.adapter;
    const seen = [];
    api.defaults.adapter = async (config) => {
      seen.push(config);
      const [status, data] = responses.shift();
      const response = { data, status, statusText: String(status), headers: {}, config };
      if (status >= 400) {
        throw Object.assign(new Error(`Request failed with status code ${status}`), { response, config });
      }
      return response;
    };
    return { seen, restore: () => { api.defaults.adapter = original; } };
  };

  const file = () => new File(['video-bytes'], 'clip.mp4', { type: 'video/mp4' });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('reserves, posts the file to storage, then completes', async () => {
    const requests = installFakeXhr({ progress: [[5, 11], [11, 11]] });
    const { seen, restore } = gatewayResponses(
      [200, ticket],
      [200, { content_id: 'content-1', status: 'queued' }],
    );
    const progress = [];
    let result;
    try {
      result = await videoApi.uploadVideoDirect(file(), 'user-1', 'high', {
        idempotencyKey: 'key-1',
        onProgress: (fraction) => progress.push(fraction),
      });
    } finally {
      restore();
    }

    expect(result).toEqual({ content_id: 'content-1', status: 'queued' });
    expect(seen[0].url).toBe('/api/content/uploads');
    expect(JSON.parse(seen[0].data)).toEqual({
      filename: 'clip.mp4',
      content_type: 'video/mp4',
      size_bytes: 11,
      priority: 'high',
      user_id: 'user-1',
    });
    expect(seen[0].headers.get('Idempotency-Key')).toBe('key-1');
    // Straight to storage, with the policy fields first and the file last.
    expect(requests).toHaveLength(1);
    expect(requests[0].method).toBe('POST');
    expect(requests[0].url).toBe('http://127.0.0.1:9000/bucket');
    expect([...requests[0].body.keys()]).toEqual(['key', 'policy', 'file']);
    expect(requests[0].body.get('file').name).toBe('clip.mp4');
    expect(progress).toEqual([5 / 11, 1]);
    expect(seen[1].url).toBe('/api/content/uploads/content-1/complete');
    expect(seen[1].params.user_id).toBe('user-1');
  });

  it('marks only a 404 reservation as unavailable', async () => {
    const reserve404 = gatewayResponses([404, { error: { code: 'DIRECT_UPLOAD_DISABLED' } }]);
    let error;
    try {
      await videoApi.uploadVideoDirect(file());
    } catch (caught) {
      error = caught;
    } finally {
      reserve404.restore();
    }
    expect(isDirectUploadUnavailableError(error)).toBe(true);

    installFakeXhr();
    const complete404 = gatewayResponses([200, ticket], [404, { detail: 'Direct upload not found' }]);
    try {
      await videoApi.uploadVideoDirect(file());
    } catch (caught) {
      error = caught;
    } finally {
      complete404.restore();
    }
    expect(error.response.status).toBe(404);
    expect(isDirectUploadUnavailableError(error)).toBe(false);
  });

  it('skips the upload when a retried reservation was already completed', async () => {
    const requests = installFakeXhr();
    const { seen, restore } = gatewayResponses([200, {
      content_id: 'content-1', status: 'queued', upload: null, idempotent_replay: true,
    }]);
    let result;
    try {
      result = await videoApi.uploadVideoDirect(file(), 'user-1', 'normal', { idempotencyKey: 'key-1' });
    } finally {
      restore();
    }

    expect(result).toMatchObject({ content_id: 'content-1', status: 'queued' });
    expect(requests).toHaveLength(0);
    expect(seen).toHaveLength(1);
  });

  it('retries complete while the object is not visible yet', async () => {
    installFakeXhr();
    const { seen, restore } = gatewayResponses(
      [200, ticket],
      [409, { error: { code: 'DIRECT_UPLOAD_OBJECT_MISSING', retryable: true } }],
      [200, { content_id: 'content-1', status: 'queued' }],
    );
    let result;
    try {
      result = await videoApi.uploadVideoDirect(file(), 'user-1', 'normal', { completeRetryDelayMs: 0 });
    } finally {
      restore();
    }

    expect(result.status).toBe('queued');
    expect(seen.map((config) => config.url)).toEqual([
      '/api/content/uploads',
      '/api/content/uploads/content-1/complete',
      '/api/content/uploads/content-1/complete',
    ]);
  });

  it('treats a pending complete as accepted and other complete errors as failures', async () => {
    installFakeXhr();
    const pending = gatewayResponses([200, ticket], [503, {
      error: { code: 'CONTENT_ENQUEUE_PENDING', retryable: true },
      content_id: 'content-1',
      status: 'pending_publish',
    }]);
    try {
      await expect(videoApi.uploadVideoDirect(file())).resolves.toMatchObject({
        content_id: 'content-1',
        status: 'pending_publish',
      });
    } finally {
      pending.restore();
    }

    const mismatch = gatewayResponses([200, ticket], [422, { error: { code: 'DIRECT_UPLOAD_MISMATCH' } }]);
    try {
      await expect(videoApi.uploadVideoDirect(file())).rejects.toThrow('422');
    } finally {
      mismatch.restore();
    }
  });

  it('reports storage rejections with the S3 error, network errors and stalls', async () => {
    installFakeXhr({
      status: 403,
      responseText: '<?xml version="1.0"?><Error><Code>AccessDenied</Code>'
        + '<Message>Invalid according to Policy: Policy expired.</Message></Error>',
    });
    await expect(uploadToStorage(ticket.upload, file())).rejects.toMatchObject({
      message: 'Storage rejected the upload (HTTP 403, AccessDenied: Invalid according to Policy: Policy expired.)',
      response: {
        status: 403,
        data: { error: { code: 'AccessDenied' } },
      },
    });
    installFakeXhr({ status: 501, responseText: '' });
    await expect(uploadToStorage(ticket.upload, file())).rejects.toMatchObject({
      message: 'Storage rejected the upload (HTTP 501)',
      response: { data: { error: { code: 'STORAGE_REJECTED' } } },
    });

    installFakeXhr({ fail: true });
    await expect(uploadToStorage(ticket.upload, file())).rejects.toThrow(/network error or blocked by CORS/);

    const requests = installFakeXhr({ timeout: true });
    await expect(uploadToStorage(ticket.upload, file())).rejects.toThrow(/timed out/);
    // A stall guard that grows with the file size.
    expect(requests[0].timeout).toBe(storageUploadTimeoutMs(11));
    expect(storageUploadTimeoutMs(128 * 1000)).toBe(60000 + 1000);
  });

  it('tries complete before sending a replayed reservation again', async () => {
    const requests = installFakeXhr();
    const replay = { ...ticket, idempotent_replay: true };
    const completed = gatewayResponses([200, replay], [200, { content_id: 'content-1', status: 'queued' }]);
    try {
      await expect(videoApi.uploadVideoDirect(file(), 'user-1', 'normal', { idempotencyKey: 'key-1' }))
        .resolves.toMatchObject({ status: 'queued' });
    } finally {
      completed.restore();
    }
    // The file was already in storage: nothing was sent again.
    expect(requests).toHaveLength(0);

    const missing = gatewayResponses(
      [200, replay],
      [409, { error: { code: 'DIRECT_UPLOAD_OBJECT_MISSING', retryable: true } }],
      [200, { content_id: 'content-1', status: 'queued' }],
    );
    try {
      await expect(videoApi.uploadVideoDirect(file(), 'user-1', 'normal', {
        idempotencyKey: 'key-1',
        completeRetryDelayMs: 0,
      })).resolves.toMatchObject({ status: 'queued' });
    } finally {
      missing.restore();
    }
    // One complete before the upload (no retries on OBJECT_MISSING), one after.
    expect(missing.seen.map((config) => config.url)).toEqual([
      '/api/content/uploads',
      '/api/content/uploads/content-1/complete',
      '/api/content/uploads/content-1/complete',
    ]);
    expect(requests).toHaveLength(1);
  });

  it('retries complete after a network error and gives it a long timeout', async () => {
    installFakeXhr();
    const original = api.defaults.adapter;
    const seen = [];
    api.defaults.adapter = async (config) => {
      seen.push(config);
      if (seen.length === 1) {
        return { data: ticket, status: 200, statusText: 'OK', headers: {}, config };
      }
      if (seen.length === 2) {
        throw Object.assign(new Error('Network Error'), { config });
      }
      return { data: { content_id: 'content-1', status: 'queued' }, status: 200, statusText: 'OK', headers: {}, config };
    };
    try {
      await expect(videoApi.uploadVideoDirect(file(), 'user-1', 'normal', { completeRetryDelayMs: 0 }))
        .resolves.toMatchObject({ status: 'queued' });
    } finally {
      api.defaults.adapter = original;
    }

    expect(seen).toHaveLength(3);
    expect(seen[2].timeout).toBe(120000);
  });

  it('knows which complete answers keep the reservation usable', () => {
    const error = (code) => ({ response: { status: 409, data: { error: { code } } } });

    expect(isRetryableCompleteError(error('DIRECT_UPLOAD_OBJECT_MISSING'))).toBe(true);
    expect(isRetryableCompleteError(error('DIRECT_UPLOAD_CHANGED'))).toBe(true);
    expect(isRetryableCompleteError(error('IDEMPOTENCY_KEY_REUSED'))).toBe(false);
    expect(isRetryableCompleteError(new Error('network'))).toBe(false);
  });

  it('reports multipart upload progress as a fraction', async () => {
    const original = api.defaults.adapter;
    api.defaults.adapter = async (config) => {
      config.onUploadProgress({ loaded: 3, total: 4 });
      return { data: { content_id: 'c1' }, status: 200, statusText: 'OK', headers: {}, config };
    };
    const progress = [];
    try {
      await videoApi.uploadVideo(file(), 'user-1', 'normal', { onProgress: (fraction) => progress.push(fraction) });
    } finally {
      api.defaults.adapter = original;
    }

    expect(progress).toEqual([0.75]);
  });
});
