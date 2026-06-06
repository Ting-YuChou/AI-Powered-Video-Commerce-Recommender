import { expect, test } from '@playwright/test';

const runMockPipeline = async (page) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Mock' }).click();
  await page.getByRole('button', { name: /Run mock sample/i }).click();
  await expect(page.getByText('AeroStride Knit Runner')).toBeVisible();
};

test('mock pipeline renders recommendations', async ({ page }) => {
  await runMockPipeline(page);

  await expect(page.getByText('Ranked products')).toBeVisible();
  await expect(page.getByText('mock-workbench-v1')).toBeVisible();
});

test('results search, filter, sort, drawer, and actions work', async ({ page }) => {
  await runMockPipeline(page);

  await page.getByPlaceholder(/Search by title/i).fill('watch');
  await expect(page.getByText('PulseTrack Sport Watch')).toBeVisible();
  await expect(page.getByText('AeroStride Knit Runner')).toHaveCount(0);

  await page.getByPlaceholder(/Search by title/i).fill('');
  await page.getByLabel('Filter category').selectOption('Accessories');
  await expect(page.getByText('PulseTrack Sport Watch')).toBeVisible();

  await page.getByLabel('Filter category').selectOption('all');
  await page.getByLabel('Sort recommendations').selectOption('price_desc');
  await expect(page.getByText('PulseTrack Sport Watch')).toBeVisible();

  await page.getByRole('button', { name: /Details/i }).first().click();
  const drawer = page.getByRole('dialog', { name: /Recommendation detail/i });
  await expect(drawer).toBeVisible();
  await expect(drawer.getByText('Ranking score')).toBeVisible();

  await drawer.getByRole('button', { name: /Add to cart/i }).click();
  await expect(drawer.getByText(/Accepted by mock event log/i)).toBeVisible();
});

test('analytics and system tabs render mock data', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Mock' }).click();

  await page.getByRole('button', { name: /Analytics/i }).click();
  await expect(page.getByText('Interaction analytics')).toBeVisible();
  await expect(page.getByText('Action count distribution')).toBeVisible();

  await page.getByRole('button', { name: /System/i }).click();
  await expect(page.getByText('Gateway and component health')).toBeVisible();
  await expect(page.getByText('gateway_api')).toBeVisible();
});

test('mobile viewport has no horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await runMockPipeline(page);

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(1);
});
