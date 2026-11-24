import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import { createUploadIdempotencyKey, videoApi } from './api';

vi.mock('./api', async () => {
  const actual = await vi.importActual('./api');
  const interactionActions = {
    VIEW: 'view',
    CLICK: 'click',
    PURCHASE: 'purchase',
    ADD_TO_CART: 'add_to_cart',
    REMOVE_FROM_CART: 'remove_from_cart',
    FAVORITE: 'favorite',
    SHARE: 'share',
  };

  return {
    DEFAULT_UPLOAD_LIMITS: actual.DEFAULT_UPLOAD_LIMITS,
    createUploadIdempotencyKey: vi.fn(),
    isStaleIdempotencyKeyError: actual.isStaleIdempotencyKeyError,
    isTerminalFailedStatus: actual.isTerminalFailedStatus,
    randomUuid: actual.randomUuid,
    interactionActions,
    videoApi: {
      uploadVideo: vi.fn(),
      getUploadLimits: vi.fn(),
      getContentStatus: vi.fn(),
      getRecommendations: vi.fn().mockRejectedValue(new Error('live recommendations unavailable')),
      logInteraction: vi.fn().mockRejectedValue(new Error('live interaction unavailable')),
    },
    systemApi: {
      getHealth: vi.fn().mockRejectedValue(new Error('gateway down')),
      getAnalytics: vi.fn().mockRejectedValue(new Error('analytics down')),
    },
    utils: {
      generateUserId: () => 'user-test',
      validateVideoFile: actual.utils.validateVideoFile,
      formatFileSize: (bytes) => `${bytes} Bytes`,
      formatPrice: (value, currency = 'USD') => `$${Number(value || 0).toFixed(2)} ${currency}`,
      getErrorMessage: (error, fallback) => error?.message || fallback,
      describeUploadLimits: actual.utils.describeUploadLimits,
    },
  };
});

// vi.restoreAllMocks (test/setup.js) does not reset vi.fn implementations in
// vitest 4, so every test starts from these defaults.
beforeEach(() => {
  let keyCounter = 0;
  createUploadIdempotencyKey.mockReset();
  createUploadIdempotencyKey.mockImplementation(() => {
    keyCounter += 1;
    return `upload-key-${keyCounter}`;
  });
  videoApi.uploadVideo.mockReset();
  videoApi.uploadVideo.mockRejectedValue(new Error('live upload unavailable'));
  videoApi.getUploadLimits.mockReset();
  videoApi.getUploadLimits.mockRejectedValue(new Error('live limits unavailable'));
  videoApi.getContentStatus.mockReset();
  videoApi.getContentStatus.mockRejectedValue(new Error('live status unavailable'));
});

const selectLiveFile = async (user, file = new File(['video-bytes'], 'clip.mp4', { type: 'video/mp4' })) => {
  await user.upload(screen.getByTestId('video-file-input'), file);
  return file;
};

const runAnalysisButton = () => screen.getByRole('button', { name: /Run analysis/i });

const uploadKeys = () => videoApi.uploadVideo.mock.calls.map((call) => call[3]?.idempotencyKey);

const renderWorkbench = async () => {
  const user = userEvent.setup();
  render(<App />);
  return user;
};

const runMockPipeline = async (user) => {
  await user.click(screen.getByRole('button', { name: /^Mock$/i }));
  await user.click(screen.getByRole('button', { name: /run mock sample/i }));
  await screen.findByText('AeroStride Knit Runner');
};

describe('VideoCommerce workbench', () => {
  it('shows offline CTA when live backend health fails', async () => {
    const user = await renderWorkbench();

    await user.click(screen.getByRole('button', { name: /System/i }));

    expect(await screen.findByText('Gateway offline')).toBeInTheDocument();
    expect(screen.getByText('gateway down')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Switch to Mock/i })).toBeInTheDocument();
  });

  it('runs the mock pipeline and renders recommendations', async () => {
    const user = await renderWorkbench();

    await runMockPipeline(user);

    expect(screen.getByText('Ranked products')).toBeInTheDocument();
    expect(screen.getByText('TrailShell Utility Jacket')).toBeInTheDocument();
    expect(screen.getByText('mock-workbench-v1')).toBeInTheDocument();
  });

  it('filters and sorts mock recommendations', async () => {
    const user = await renderWorkbench();
    await runMockPipeline(user);

    await user.type(screen.getByPlaceholderText(/Search by title/i), 'watch');
    expect(screen.getByText('PulseTrack Sport Watch')).toBeInTheDocument();
    expect(screen.queryByText('AeroStride Knit Runner')).not.toBeInTheDocument();

    await user.clear(screen.getByPlaceholderText(/Search by title/i));
    await user.selectOptions(screen.getByRole('combobox', { name: /Filter category/i }), 'Accessories');
    expect(screen.getByText('PulseTrack Sport Watch')).toBeInTheDocument();
    expect(screen.queryByText('TrailShell Utility Jacket')).not.toBeInTheDocument();

    await user.selectOptions(screen.getByRole('combobox', { name: /Sort recommendations/i }), 'price_desc');
    expect(screen.getByText('PulseTrack Sport Watch')).toBeInTheDocument();
  });

  it('opens product detail drawer and records action feedback', async () => {
    const user = await renderWorkbench();
    await runMockPipeline(user);

    await user.click(screen.getAllByRole('button', { name: /Details/i })[0]);
    const dialog = await screen.findByRole('dialog', { name: /Recommendation detail/i });

    expect(within(dialog).getByText('AeroStride Knit Runner')).toBeInTheDocument();
    expect(within(dialog).getByText('Ranking score')).toBeInTheDocument();

    await user.click(within(dialog).getByRole('button', { name: /Favorite/i }));

    await waitFor(() => {
      expect(within(dialog).getByText(/Accepted by mock event log/i)).toBeInTheDocument();
    });
    expect(within(dialog).getByText('Session actions for this product')).toBeInTheDocument();
  });

  it('renders mock analytics and system tabs', async () => {
    const user = await renderWorkbench();

    await user.click(screen.getByRole('button', { name: /^Mock$/i }));
    await user.click(screen.getByRole('button', { name: /Analytics/i }));

    expect(await screen.findByText('Interaction analytics')).toBeInTheDocument();
    expect(screen.getByText('Action count distribution')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /System/i }));
    expect(await screen.findByText('Gateway and component health')).toBeInTheDocument();
    expect(await screen.findByText('gateway_api')).toBeInTheDocument();
  });

  it('shows the upload limit reported by the gateway and validates against it', async () => {
    videoApi.getUploadLimits.mockResolvedValue({
      max_file_size: 10,
      allowed_extensions: ['.mp4'],
      allowed_mime_types: ['video/mp4'],
    });
    const user = await renderWorkbench();

    await waitFor(() => {
      expect(screen.getByTestId('upload-limits')).toHaveTextContent('MP4 up to 10 Bytes.');
    });
    await selectLiveFile(user, new File(['01234567890'], 'clip.mp4', { type: 'video/mp4' }));

    expect(await screen.findByText('File size exceeds the 10 Bytes upload limit')).toBeInTheDocument();
  });

  it('skips the size check until the gateway limits are known', async () => {
    const user = await renderWorkbench();
    const file = new File(['video-bytes'], 'big.mp4', { type: 'video/mp4' });
    Object.defineProperty(file, 'size', { value: 200 * 1024 * 1024 });

    await selectLiveFile(user, file);

    expect(await screen.findByText('big.mp4')).toBeInTheDocument();
    expect(screen.queryByText(/upload limit/)).not.toBeInTheDocument();
  });

  it('polls a pending upload, stops on failed_missing_object, and uses a new key next time', async () => {
    videoApi.uploadVideo.mockResolvedValueOnce({
      content_id: 'content-9',
      status: 'pending_publish',
      message: 'Upload stored; waiting for the processing queue to accept it.',
    });
    videoApi.getContentStatus.mockResolvedValueOnce({
      content_id: 'content-9',
      status: 'failed_missing_object',
      message: 'Uploaded object is missing',
    });
    const user = await renderWorkbench();

    await selectLiveFile(user);
    await user.click(runAnalysisButton());

    expect(await screen.findAllByText('Uploaded object is missing')).not.toHaveLength(0);
    expect(videoApi.uploadVideo).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'clip.mp4' }),
      'user-test',
      'normal',
      { idempotencyKey: 'upload-key-1' }
    );
    expect(videoApi.getContentStatus).toHaveBeenCalledTimes(1);

    await waitFor(() => expect(runAnalysisButton()).toBeEnabled());
    await user.click(runAnalysisButton());
    await waitFor(() => expect(videoApi.uploadVideo).toHaveBeenCalledTimes(2));
    expect(uploadKeys()).toEqual(['upload-key-1', 'upload-key-2']);
  });

  it('reuses the key on retries and switches keys when the priority changes', async () => {
    const user = await renderWorkbench();
    await selectLiveFile(user);

    for (let attempt = 1; attempt <= 2; attempt += 1) {
      await user.click(runAnalysisButton());
      await waitFor(() => expect(videoApi.uploadVideo).toHaveBeenCalledTimes(attempt));
      await waitFor(() => expect(runAnalysisButton()).toBeEnabled());
    }
    await user.selectOptions(screen.getByLabelText('Priority'), 'high');
    await user.click(runAnalysisButton());
    await waitFor(() => expect(videoApi.uploadVideo).toHaveBeenCalledTimes(3));

    expect(uploadKeys()).toEqual(['upload-key-1', 'upload-key-1', 'upload-key-2']);
    expect(videoApi.uploadVideo.mock.calls[2][2]).toBe('high');
  });

  it('retries once with a new key when the gateway rejects a stale key', async () => {
    const staleKeyError = Object.assign(new Error('Request failed with status code 409'), {
      response: { status: 409, data: { error: { code: 'IDEMPOTENCY_KEY_JOB_FAILED' } } },
    });
    videoApi.uploadVideo
      .mockRejectedValueOnce(staleKeyError)
      .mockResolvedValueOnce({ content_id: 'content-5', status: 'queued' });
    videoApi.getContentStatus.mockResolvedValueOnce({
      content_id: 'content-5',
      status: 'failed',
      message: 'Processing failed',
    });
    const user = await renderWorkbench();

    await selectLiveFile(user);
    await user.click(runAnalysisButton());

    expect(await screen.findAllByText('Processing failed')).not.toHaveLength(0);
    expect(uploadKeys()).toEqual(['upload-key-1', 'upload-key-2']);
  });

  it('reports rejected uploads as upload failures, not as an offline gateway', async () => {
    videoApi.uploadVideo.mockRejectedValueOnce(Object.assign(new Error('File MIME type not supported'), {
      response: { status: 400, data: { detail: 'File MIME type not supported' } },
    }));
    const user = await renderWorkbench();

    await selectLiveFile(user);
    await user.click(runAnalysisButton());

    expect(await screen.findByText('Upload failed')).toBeInTheDocument();
    expect(screen.queryByText('Gateway offline')).not.toBeInTheDocument();
  });
});
