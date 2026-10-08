import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, vi } from 'vitest';

const localStorageValues = new Map();
Object.defineProperty(window, 'localStorage', {
  configurable: true,
  value: {
    clear: () => localStorageValues.clear(),
    getItem: (key) => localStorageValues.get(String(key)) ?? null,
    removeItem: (key) => localStorageValues.delete(String(key)),
    setItem: (key, value) => localStorageValues.set(String(key), String(value)),
  },
});

beforeEach(() => {
  vi.spyOn(Date, 'now').mockReturnValue(1_700_000_000_000);

  if (!URL.createObjectURL) {
    URL.createObjectURL = vi.fn();
  }
  if (!URL.revokeObjectURL) {
    URL.revokeObjectURL = vi.fn();
  }

  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock-video');
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
  window.localStorage.clear();
});
