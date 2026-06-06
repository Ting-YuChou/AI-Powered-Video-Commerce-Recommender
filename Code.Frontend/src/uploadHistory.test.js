import { describe, expect, it } from 'vitest';
import {
  UPLOAD_HISTORY_STORAGE_KEY,
  readUploadHistory,
  upsertUploadHistoryEntry,
  writeUploadHistory,
} from './uploadHistory';

const createEntry = (index, overrides = {}) => ({
  content_id: `content-${index}`,
  filename: `video-${index}.mp4`,
  status: 'completed',
  created_at: `2026-06-0${index}T00:00:00.000Z`,
  mode: index % 2 === 0 ? 'mock' : 'live',
  ...overrides,
});

describe('upload history helper', () => {
  it('safely falls back when stored data is malformed', () => {
    window.localStorage.setItem(UPLOAD_HISTORY_STORAGE_KEY, '{bad json');

    expect(readUploadHistory()).toEqual([]);
  });

  it('filters invalid records without crashing', () => {
    window.localStorage.setItem(UPLOAD_HISTORY_STORAGE_KEY, JSON.stringify([
      createEntry(1),
      { content_id: 'missing-fields' },
      createEntry(2, { mode: 'invalid' }),
    ]));

    expect(readUploadHistory()).toEqual([createEntry(1)]);
  });

  it('saves only the newest 10 entries', () => {
    const entries = Array.from({ length: 12 }, (_, index) => createEntry(index + 1));

    const saved = writeUploadHistory(entries);

    expect(saved).toHaveLength(10);
    expect(saved[0].content_id).toBe('content-1');
    expect(saved[9].content_id).toBe('content-10');
  });

  it('upserts by content id and keeps latest first', () => {
    writeUploadHistory([createEntry(1), createEntry(2)]);

    const nextHistory = upsertUploadHistoryEntry(createEntry(2, {
      filename: 'updated-video.mp4',
      status: 'processing',
    }));

    expect(nextHistory).toHaveLength(2);
    expect(nextHistory[0]).toMatchObject({
      content_id: 'content-2',
      filename: 'updated-video.mp4',
      status: 'processing',
    });
  });
});
