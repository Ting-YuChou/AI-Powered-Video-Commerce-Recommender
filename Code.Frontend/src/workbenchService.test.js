import { describe, expect, it } from 'vitest';
import { WORKBENCH_MODES, createWorkbenchService } from './workbenchService';

describe('mock workbench service', () => {
  it('returns upload, status, recommendations, analytics, and health without backend calls', async () => {
    const service = createWorkbenchService(WORKBENCH_MODES.MOCK);
    const file = new File(['demo'], 'demo.mp4', { type: 'video/mp4' });

    const upload = await service.uploadVideo(file, 'user-1', 'high');
    expect(upload.content_id).toContain('mock-demo.mp4');
    expect(upload.status).toBe('queued');

    const status = await service.getContentStatus(upload.content_id);
    expect(status).toMatchObject({
      content_id: upload.content_id,
      status: 'completed',
    });

    const recommendations = await service.getRecommendations('user-1', upload.content_id, {
      device: 'mobile',
      session_id: 'session-1',
      location: 'work',
    }, 3);
    expect(recommendations.recommendations).toHaveLength(3);
    expect(recommendations.metadata).toMatchObject({
      mode: 'mock',
      fallback: true,
    });

    const interaction = await service.logInteraction('user-1', 'product-1', 'favorite');
    expect(interaction.status).toBe('accepted');

    await expect(service.getAnalytics()).resolves.toHaveProperty('action_counts');
    await expect(service.getHealth()).resolves.toHaveProperty('status', 'healthy');
  });
});
