import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import App from './App';

vi.mock('./api', () => {
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
    interactionActions,
    videoApi: {
      uploadVideo: vi.fn().mockRejectedValue(new Error('live upload unavailable')),
      getContentStatus: vi.fn().mockRejectedValue(new Error('live status unavailable')),
      getRecommendations: vi.fn().mockRejectedValue(new Error('live recommendations unavailable')),
      logInteraction: vi.fn().mockRejectedValue(new Error('live interaction unavailable')),
    },
    systemApi: {
      getHealth: vi.fn().mockRejectedValue(new Error('gateway down')),
      getAnalytics: vi.fn().mockRejectedValue(new Error('analytics down')),
    },
    utils: {
      generateUserId: () => 'user-test',
      validateVideoFile: (file) => (
        file?.type?.startsWith('video/')
          ? { valid: true }
          : { valid: false, error: 'File type not supported.' }
      ),
      formatFileSize: (bytes) => `${bytes} Bytes`,
      formatPrice: (value, currency = 'USD') => `$${Number(value || 0).toFixed(2)} ${currency}`,
      getErrorMessage: (error, fallback) => error?.message || fallback,
    },
  };
});

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
});
