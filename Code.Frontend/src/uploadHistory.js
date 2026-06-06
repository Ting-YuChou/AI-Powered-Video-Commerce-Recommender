export const UPLOAD_HISTORY_STORAGE_KEY = 'videoCommerceWorkbench.history.v1';
const MAX_HISTORY_ITEMS = 10;

const isHistoryEntry = (entry) => (
  entry
  && typeof entry === 'object'
  && typeof entry.content_id === 'string'
  && entry.content_id.trim().length > 0
  && typeof entry.filename === 'string'
  && entry.filename.trim().length > 0
  && typeof entry.status === 'string'
  && typeof entry.created_at === 'string'
  && (entry.mode === 'live' || entry.mode === 'mock')
);

const resolveStorage = (storage) => {
  if (storage) {
    return storage;
  }

  if (typeof window !== 'undefined' && window.localStorage) {
    return window.localStorage;
  }

  return null;
};

export const readUploadHistory = (storage) => {
  const targetStorage = resolveStorage(storage);
  if (!targetStorage) {
    return [];
  }

  try {
    const rawValue = targetStorage.getItem(UPLOAD_HISTORY_STORAGE_KEY);
    if (!rawValue) {
      return [];
    }

    const parsed = JSON.parse(rawValue);
    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed.filter(isHistoryEntry).slice(0, MAX_HISTORY_ITEMS);
  } catch {
    return [];
  }
};

export const writeUploadHistory = (history, storage) => {
  const targetStorage = resolveStorage(storage);
  if (!targetStorage) {
    return [];
  }

  const safeHistory = Array.isArray(history)
    ? history.filter(isHistoryEntry).slice(0, MAX_HISTORY_ITEMS)
    : [];

  try {
    targetStorage.setItem(UPLOAD_HISTORY_STORAGE_KEY, JSON.stringify(safeHistory));
  } catch {
    return [];
  }

  return safeHistory;
};

export const upsertUploadHistoryEntry = (entry, storage) => {
  if (!isHistoryEntry(entry)) {
    return readUploadHistory(storage);
  }

  const existingHistory = readUploadHistory(storage);
  const nextHistory = [
    entry,
    ...existingHistory.filter((historyEntry) => historyEntry.content_id !== entry.content_id),
  ].slice(0, MAX_HISTORY_ITEMS);

  return writeUploadHistory(nextHistory, storage);
};

export const clearUploadHistory = (storage) => {
  const targetStorage = resolveStorage(storage);
  if (!targetStorage) {
    return [];
  }

  try {
    targetStorage.removeItem(UPLOAD_HISTORY_STORAGE_KEY);
  } catch {
    return [];
  }

  return [];
};
