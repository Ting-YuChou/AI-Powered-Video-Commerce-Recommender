import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  CreditCard,
  Eye,
  Heart,
  History,
  Loader2,
  MousePointerClick,
  PackageSearch,
  Play,
  RefreshCw,
  Search,
  Server,
  Share2,
  ShoppingCart,
  UploadCloud,
  Wifi,
  WifiOff,
  X,
} from 'lucide-react';
import { interactionActions, utils } from './api';
import { clearUploadHistory, readUploadHistory, upsertUploadHistoryEntry } from './uploadHistory';
import { WORKBENCH_MODES, createWorkbenchService } from './workbenchService';

const tabs = [
  { id: 'demo', label: 'Demo Flow', icon: Play },
  { id: 'results', label: 'Results', icon: PackageSearch },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'system', label: 'System', icon: Server },
];

const priorityOptions = ['low', 'normal', 'high'];
const deviceOptions = ['web', 'mobile'];
const timeOfDayOptions = ['morning', 'afternoon', 'evening', 'night'];
const locationOptions = ['home', 'work', 'travel'];
const sortOptions = [
  { value: 'confidence_desc', label: 'Confidence' },
  { value: 'ranking_desc', label: 'Ranking score' },
  { value: 'price_asc', label: 'Price low to high' },
  { value: 'price_desc', label: 'Price high to low' },
];

const actionLabels = {
  [interactionActions.CLICK]: 'Click',
  [interactionActions.FAVORITE]: 'Favorite',
  [interactionActions.SHARE]: 'Share',
  [interactionActions.ADD_TO_CART]: 'Add to cart',
  [interactionActions.PURCHASE]: 'Purchase',
  [interactionActions.VIEW]: 'View',
};

const fallbackProductImage = `data:image/svg+xml;utf8,${encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 360"><rect width="480" height="360" fill="#eef2f7"/><rect x="88" y="84" width="304" height="192" rx="24" fill="#fff"/><text x="240" y="174" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="28" font-weight="700" fill="#1f2937">Product</text><text x="240" y="214" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="15" fill="#6b7280">No image URL</text></svg>'
)}`;

const classNames = (...classes) => classes.filter(Boolean).join(' ');

const safeNumber = (value, fallback = 0) => {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : fallback;
};

const formatPercent = (value) => `${Math.round(safeNumber(value) * 100)}%`;

const formatScore = (value) => safeNumber(value).toFixed(2);

const formatPrice = (value, currency = 'USD') => utils.formatPrice(value, currency);

const formatTimestamp = (value) => {
  if (!value) {
    return '-';
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return '-';
  }

  return date.toLocaleString();
};

const getErrorMessage = (error, fallback) => utils.getErrorMessage(error, fallback);

const createMockFile = () => ({
  name: 'mock-lookbook.mp4',
  size: 18_400_000,
  type: 'video/mp4',
});

const normalizeRecommendation = (item, index) => {
  const product = item?.product || item || {};
  const productId = (
    product.product_id
    || product.id
    || item?.product_id
    || item?.id
    || `product-${index + 1}`
  );

  return {
    id: productId,
    product_id: productId,
    title: product.title || product.name || item?.title || item?.name || `Product ${index + 1}`,
    brand: product.brand || item?.brand || '',
    category: product.category || item?.category || 'Uncategorized',
    rating: safeNumber(product.rating ?? item?.rating, null),
    price: safeNumber(product.price ?? item?.price, null),
    currency: product.currency || item?.currency || 'USD',
    image_url: product.image_url || item?.image_url || fallbackProductImage,
    confidence_score: safeNumber(
      item?.confidence_score ?? item?.confidence ?? product.confidence_score ?? product.confidence,
      0
    ),
    ranking_score: safeNumber(
      item?.ranking_score ?? item?.score ?? product.ranking_score ?? product.score,
      0
    ),
    reason: item?.reason || item?.explanation || product.reason || product.description || '',
    rank: safeNumber(item?.rank ?? item?.position, index + 1),
    raw: item,
  };
};

const normalizeRecommendations = (response) => {
  const items = Array.isArray(response?.recommendations)
    ? response.recommendations
    : Array.isArray(response?.items)
      ? response.items
      : [];

  return items.map(normalizeRecommendation);
};

const normalizeMetadata = (response) => response?.metadata || response?.meta || {};

const normalizeAnalytics = (analytics) => {
  const source = analytics?.summary || analytics || {};
  return {
    interactions: safeNumber(source.interactions ?? source.total_interactions, 0),
    uniqueUsers: safeNumber(source.unique_users ?? source.uniqueUsers, 0),
    uniqueProducts: safeNumber(source.unique_products ?? source.uniqueProducts, 0),
    ctr: safeNumber(source.ctr ?? source.click_through_rate, 0),
    conversionRate: safeNumber(source.conversion_rate ?? source.conversionRate, 0),
    actionCounts: source.action_counts || source.actionCounts || {},
  };
};

const getHealthStatus = (health) => {
  if (!health) {
    return 'offline';
  }

  return health.status || health.overall_status || 'unknown';
};

const getHealthComponents = (health) => {
  if (!health?.components || typeof health.components !== 'object') {
    return [];
  }

  return Object.entries(health.components).map(([name, value]) => ({
    name,
    status: typeof value === 'string' ? value : value?.status || 'unknown',
    message: typeof value === 'string' ? '' : value?.message || value?.detail || '',
  }));
};

const getStatusStyles = (status) => {
  const normalized = String(status || '').toLowerCase();

  if (['healthy', 'completed', 'accepted', 'ready', 'mock'].includes(normalized)) {
    return 'border-emerald-200 bg-emerald-50 text-emerald-700';
  }

  if (['queued', 'processing', 'pending', 'uploading', 'recommending'].includes(normalized)) {
    return 'border-sky-200 bg-sky-50 text-sky-700';
  }

  if (['warning', 'degraded', 'fallback', 'timeout'].includes(normalized)) {
    return 'border-amber-200 bg-amber-50 text-amber-700';
  }

  if (['failed', 'offline', 'unhealthy', 'error'].includes(normalized)) {
    return 'border-red-200 bg-red-50 text-red-700';
  }

  return 'border-slate-200 bg-slate-50 text-slate-600';
};

function StatusPill({ status, label = status }) {
  return (
    <span className={classNames(
      'inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold capitalize',
      getStatusStyles(status)
    )}
    >
      {label || 'unknown'}
    </span>
  );
}

function SectionHeader({ eyebrow, title, description, action }) {
  return (
    <div className="flex flex-col gap-3 border-b border-slate-200 pb-4 sm:flex-row sm:items-start sm:justify-between">
      <div>
        {eyebrow && <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{eyebrow}</p>}
        <h2 className="mt-1 text-xl font-semibold text-slate-950">{title}</h2>
        {description && <p className="mt-1 max-w-3xl text-sm text-slate-600">{description}</p>}
      </div>
      {action}
    </div>
  );
}

function ModeToggle({ mode, onChange }) {
  return (
    <div className="inline-flex rounded-md border border-slate-200 bg-white p-1 shadow-sm" aria-label="Workbench mode">
      {[
        { value: WORKBENCH_MODES.LIVE, label: 'Live', icon: Wifi },
        { value: WORKBENCH_MODES.MOCK, label: 'Mock', icon: WifiOff },
      ].map((option) => {
        const Icon = option.icon;
        const selected = mode === option.value;
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            className={classNames(
              'inline-flex items-center gap-2 rounded px-3 py-2 text-sm font-semibold transition',
              selected ? 'bg-slate-950 text-white' : 'text-slate-600 hover:bg-slate-100'
            )}
          >
            <Icon className="h-4 w-4" aria-hidden="true" />
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

function Tabs({ activeTab, onChange }) {
  return (
    <nav className="flex gap-1 overflow-x-auto border-b border-slate-200" aria-label="Workbench tabs">
      {tabs.map((tab) => {
        const Icon = tab.icon;
        const selected = activeTab === tab.id;
        return (
          <button
            key={tab.id}
            type="button"
            onClick={() => onChange(tab.id)}
            className={classNames(
              'inline-flex min-h-[44px] items-center gap-2 border-b-2 px-4 py-3 text-sm font-semibold transition',
              selected
                ? 'border-slate-950 text-slate-950'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            )}
          >
            <Icon className="h-4 w-4" aria-hidden="true" />
            {tab.label}
          </button>
        );
      })}
    </nav>
  );
}

function ErrorCallout({
  type,
  title,
  message,
  mode,
  onRetry,
  onRefreshHealth,
  onSwitchMock,
}) {
  if (!message && !title) {
    return null;
  }

  const resolvedTitle = title || {
    gateway_offline: 'Gateway offline',
    upload_failed: 'Upload failed',
    processing_timeout: 'Processing timeout',
    recommendation_fallback: 'Recommendation fallback',
    analytics_unavailable: 'Analytics unavailable',
  }[type] || 'Something needs attention';

  return (
    <div className="rounded-md border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900" role="alert">
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
        <div className="min-w-0 flex-1">
          <p className="font-semibold">{resolvedTitle}</p>
          {message && <p className="mt-1 text-amber-800">{message}</p>}
          <div className="mt-3 flex flex-wrap gap-2">
            {onRetry && (
              <button type="button" onClick={onRetry} className="rounded bg-amber-900 px-3 py-1.5 text-xs font-semibold text-white">
                Retry
              </button>
            )}
            {onRefreshHealth && (
              <button type="button" onClick={onRefreshHealth} className="rounded border border-amber-300 px-3 py-1.5 text-xs font-semibold text-amber-900">
                Refresh health
              </button>
            )}
            {mode === WORKBENCH_MODES.LIVE && onSwitchMock && (
              <button type="button" onClick={onSwitchMock} className="rounded border border-amber-300 px-3 py-1.5 text-xs font-semibold text-amber-900">
                Switch to Mock
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label className="block text-sm font-medium text-slate-700">
      <span>{label}</span>
      <div className="mt-1">{children}</div>
    </label>
  );
}

function ProcessingTimeline({ phase, contentStatus }) {
  const status = contentStatus?.status || 'idle';
  const steps = [
    {
      label: 'Upload accepted',
      complete: ['processing', 'recommending', 'ready'].includes(phase),
      active: phase === 'uploading',
    },
    {
      label: 'Content processing',
      complete: phase === 'recommending' || phase === 'ready',
      active: phase === 'processing',
    },
    {
      label: 'Recommendations ready',
      complete: phase === 'ready',
      active: phase === 'recommending',
    },
  ];

  return (
    <div className="space-y-3" data-testid="processing-timeline">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-semibold text-slate-900">Processing status</p>
        <StatusPill status={status} label={status} />
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        {steps.map((step) => (
          <div key={step.label} className="flex items-center gap-3 rounded-md border border-slate-200 bg-white p-3">
            <span className={classNames(
              'flex h-8 w-8 shrink-0 items-center justify-center rounded-full border',
              step.complete ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : '',
              step.active ? 'border-sky-200 bg-sky-50 text-sky-700' : '',
              !step.complete && !step.active ? 'border-slate-200 bg-slate-50 text-slate-400' : ''
            )}
            >
              {step.active ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> : <CheckCircle2 className="h-4 w-4" aria-hidden="true" />}
            </span>
            <div>
              <p className="text-sm font-medium text-slate-900">{step.label}</p>
              <p className="text-xs text-slate-500">{step.complete ? 'Complete' : step.active ? 'Running' : 'Waiting'}</p>
            </div>
          </div>
        ))}
      </div>
      {contentStatus?.message && <p className="text-sm text-slate-600">{contentStatus.message}</p>}
    </div>
  );
}

function UploadHistoryList({ uploadHistory, onRestore, onClear }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <History className="h-4 w-4 text-slate-500" aria-hidden="true" />
          <p className="text-sm font-semibold text-slate-900">Recent uploads</p>
        </div>
        {uploadHistory.length > 0 && (
          <button type="button" onClick={onClear} className="text-xs font-semibold text-slate-500 hover:text-slate-900">
            Clear
          </button>
        )}
      </div>
      {uploadHistory.length === 0 ? (
        <p className="rounded-md border border-dashed border-slate-300 p-3 text-sm text-slate-500">
          No uploads saved in this browser yet.
        </p>
      ) : (
        <div className="space-y-2">
          {uploadHistory.map((entry) => (
            <button
              key={entry.content_id}
              type="button"
              onClick={() => onRestore(entry)}
              className="w-full rounded-md border border-slate-200 bg-white p-3 text-left transition hover:border-slate-300 hover:bg-slate-50"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="max-w-[15rem] truncate text-sm font-semibold text-slate-900">{entry.filename}</p>
                <div className="flex items-center gap-2">
                  <StatusPill status={entry.status} label={entry.status} />
                  <StatusPill status={entry.mode} label={entry.mode} />
                </div>
              </div>
              <p className="mt-1 truncate text-xs text-slate-500">{entry.content_id}</p>
              <p className="mt-1 text-xs text-slate-500">{formatTimestamp(entry.created_at)}</p>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function UploadAnalysisPanel({
  mode,
  userId,
  setUserId,
  priority,
  setPriority,
  recommendationCount,
  setRecommendationCount,
  device,
  setDevice,
  timeOfDay,
  setTimeOfDay,
  location,
  setLocation,
  sessionId,
  setSessionId,
  selectedFile,
  selectedVideo,
  dragActive,
  setDragActive,
  onFileSelect,
  onRunAnalysis,
  onRunMockSample,
  isBusy,
  phase,
  contentStatus,
  currentContentId,
  error,
  onRetry,
  onRefreshHealth,
  onSwitchMock,
  uploadHistory,
  onRestoreHistory,
  onClearHistory,
}) {
  const canUseMockSample = mode === WORKBENCH_MODES.MOCK;

  return (
    <section className="space-y-5 rounded-md border border-slate-200 bg-white p-5 shadow-sm">
      <SectionHeader
        eyebrow="Upload & Analysis"
        title="Run the recommendation flow"
        description="Live mode uses the gateway. Mock mode is a local demo path and does not send backend requests."
        action={<StatusPill status={mode} label={mode === WORKBENCH_MODES.MOCK ? 'Mock mode' : 'Live mode'} />}
      />

      {mode === WORKBENCH_MODES.MOCK && (
        <div className="rounded-md border border-sky-200 bg-sky-50 p-3 text-sm text-sky-800">
          Mock mode is enabled. Upload, processing, recommendations, health, analytics, and interactions are served from fixed frontend data.
        </div>
      )}

      {error && (
        <ErrorCallout
          type={error.type}
          message={error.message}
          mode={mode}
          onRetry={onRetry}
          onRefreshHealth={onRefreshHealth}
          onSwitchMock={onSwitchMock}
        />
      )}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Field label="User id">
          <input
            type="text"
            value={userId}
            onChange={(event) => setUserId(event.target.value)}
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-200"
          />
        </Field>
        <Field label="Priority">
          <select
            value={priority}
            onChange={(event) => setPriority(event.target.value)}
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-200"
          >
            {priorityOptions.map((option) => <option key={option} value={option}>{option}</option>)}
          </select>
        </Field>
        <Field label="Result count">
          <input
            type="number"
            min="1"
            max="20"
            value={recommendationCount}
            onChange={(event) => setRecommendationCount(Math.max(1, Math.min(20, Number(event.target.value) || 1)))}
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-200"
          />
        </Field>
        <Field label="Device">
          <select
            value={device}
            onChange={(event) => setDevice(event.target.value)}
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-200"
          >
            {deviceOptions.map((option) => <option key={option} value={option}>{option}</option>)}
          </select>
        </Field>
        <Field label="Time of day">
          <select
            value={timeOfDay}
            onChange={(event) => setTimeOfDay(event.target.value)}
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-200"
          >
            {timeOfDayOptions.map((option) => <option key={option} value={option}>{option}</option>)}
          </select>
        </Field>
        <Field label="Location">
          <select
            value={location}
            onChange={(event) => setLocation(event.target.value)}
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-200"
          >
            {locationOptions.map((option) => <option key={option} value={option}>{option}</option>)}
          </select>
        </Field>
        <Field label="Session id">
          <input
            type="text"
            value={sessionId}
            onChange={(event) => setSessionId(event.target.value)}
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-200"
          />
        </Field>
      </div>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.4fr)_minmax(18rem,0.6fr)]">
        <div className="space-y-4">
          <div
            className={classNames(
              'rounded-md border-2 border-dashed p-5 transition',
              dragActive ? 'border-sky-400 bg-sky-50' : 'border-slate-300 bg-slate-50'
            )}
            onDragOver={(event) => {
              event.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragActive(false);
              onFileSelect(event.dataTransfer.files);
            }}
          >
            <div className="flex flex-col items-center justify-center gap-3 text-center">
              <UploadCloud className="h-10 w-10 text-slate-500" aria-hidden="true" />
              <div>
                <p className="font-semibold text-slate-900">Drop a video or choose a file</p>
                <p className="text-sm text-slate-500">MP4, MOV, AVI, MKV, or WEBM up to 100MB.</p>
              </div>
              <label className="inline-flex cursor-pointer items-center gap-2 rounded-md bg-slate-950 px-4 py-2 text-sm font-semibold text-white">
                Choose video
                <input
                  data-testid="video-file-input"
                  type="file"
                  accept="video/*"
                  className="sr-only"
                  onChange={(event) => onFileSelect(event.target.files)}
                />
              </label>
            </div>
          </div>

          {selectedFile && (
            <div className="rounded-md border border-slate-200 bg-white p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-slate-900">{selectedFile.name}</p>
                  <p className="text-xs text-slate-500">{utils.formatFileSize(selectedFile.size || 0)}</p>
                </div>
                <StatusPill status="ready" label="Ready to upload" />
              </div>
              {selectedVideo && (
                <video
                  className="mt-4 aspect-video w-full rounded-md bg-slate-950 object-contain"
                  src={selectedVideo}
                  controls
                />
              )}
            </div>
          )}

          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => onRunAnalysis(false)}
              disabled={isBusy}
              className="inline-flex min-h-[44px] items-center gap-2 rounded-md bg-slate-950 px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-slate-400"
            >
              {isBusy ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> : <Play className="h-4 w-4" aria-hidden="true" />}
              Run analysis
            </button>
            {canUseMockSample && (
              <button
                type="button"
                onClick={onRunMockSample}
                disabled={isBusy}
                className="inline-flex min-h-[44px] items-center gap-2 rounded-md border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-800 disabled:cursor-not-allowed disabled:text-slate-400"
              >
                <PackageSearch className="h-4 w-4" aria-hidden="true" />
                Run mock sample
              </button>
            )}
          </div>

          {currentContentId && (
            <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Content id</p>
              <p className="mt-1 break-all font-mono text-sm text-slate-900">{currentContentId}</p>
            </div>
          )}

          <ProcessingTimeline phase={phase} contentStatus={contentStatus} />
        </div>

        <UploadHistoryList
          uploadHistory={uploadHistory}
          onRestore={onRestoreHistory}
          onClear={onClearHistory}
        />
      </div>
    </section>
  );
}

function RecommendationMetadata({ metadata, onRefresh }) {
  const fields = [
    ['total_candidates', 'Candidates'],
    ['response_time_ms', 'Response ms'],
    ['model_version', 'Model'],
    ['cache_hit', 'Cache hit'],
    ['fallback', 'Fallback'],
  ];

  return (
    <div className="space-y-3">
      {metadata?.fallback && (
        <ErrorCallout
          type="recommendation_fallback"
          message="The recommendation response indicates fallback behavior. Refresh or adjust context to compare results."
          onRetry={onRefresh}
        />
      )}
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        {fields.map(([key, label]) => {
          const value = metadata?.[key];
          const displayValue = typeof value === 'boolean' ? String(value) : value ?? '-';
          return (
            <div key={key} className="rounded-md border border-slate-200 bg-white p-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
              <p className="mt-1 truncate text-sm font-semibold text-slate-900">{displayValue}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ProductActionButton({ action, icon: Icon, active, onClick }) {
  return (
    <button
      type="button"
      aria-label={actionLabels[action]}
      title={actionLabels[action]}
      onClick={onClick}
      className={classNames(
        'inline-flex h-9 w-9 items-center justify-center rounded-md border text-slate-600 transition hover:border-slate-400 hover:text-slate-950',
        active ? 'border-slate-950 bg-slate-950 text-white hover:text-white' : 'border-slate-200 bg-white'
      )}
    >
      <Icon className="h-4 w-4" aria-hidden="true" />
    </button>
  );
}

function ProductCard({
  product,
  isFavorite,
  isInCart,
  isPurchased,
  feedback,
  onOpen,
  onAction,
}) {
  return (
    <article
      className="group flex h-full flex-col overflow-hidden rounded-md border border-slate-200 bg-white shadow-sm transition hover:border-slate-300 hover:shadow-md"
    >
      <button type="button" onClick={() => onOpen(product)} className="text-left">
        <img
          src={product.image_url || fallbackProductImage}
          alt={product.title}
          className="aspect-[4/3] w-full bg-slate-100 object-cover"
          onError={(event) => {
            event.currentTarget.src = fallbackProductImage;
          }}
        />
      </button>
      <div className="flex flex-1 flex-col gap-3 p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <button type="button" onClick={() => onOpen(product)} className="block text-left">
              <h3 className="line-clamp-2 text-base font-semibold text-slate-950 group-hover:text-slate-700">
                {product.title}
              </h3>
            </button>
            <p className="mt-1 text-sm text-slate-500">
              {product.brand || 'Brand unavailable'} · {product.category || 'Uncategorized'}
            </p>
          </div>
          <p className="shrink-0 text-sm font-semibold text-slate-950">{formatPrice(product.price, product.currency)}</p>
        </div>

        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="rounded bg-slate-50 p-2">
            <p className="text-slate-500">Confidence</p>
            <p className="font-semibold text-slate-900">{formatPercent(product.confidence_score)}</p>
          </div>
          <div className="rounded bg-slate-50 p-2">
            <p className="text-slate-500">Ranking</p>
            <p className="font-semibold text-slate-900">{formatScore(product.ranking_score)}</p>
          </div>
          <div className="rounded bg-slate-50 p-2">
            <p className="text-slate-500">Rating</p>
            <p className="font-semibold text-slate-900">{product.rating ? product.rating.toFixed(1) : '-'}</p>
          </div>
        </div>

        <p className="line-clamp-2 text-sm text-slate-600">{product.reason || 'Recommendation reason unavailable.'}</p>

        {feedback && (
          <div className={classNames(
            'rounded-md border px-3 py-2 text-xs font-medium',
            feedback.status === 'failed'
              ? 'border-red-200 bg-red-50 text-red-700'
              : feedback.status === 'pending'
                ? 'border-sky-200 bg-sky-50 text-sky-700'
                : 'border-emerald-200 bg-emerald-50 text-emerald-700'
          )}
          >
            {feedback.message}
          </div>
        )}

        <div className="mt-auto flex flex-wrap items-center gap-2">
          <ProductActionButton
            action={interactionActions.FAVORITE}
            icon={Heart}
            active={isFavorite}
            onClick={(event) => onAction(product, interactionActions.FAVORITE, event)}
          />
          <ProductActionButton
            action={interactionActions.SHARE}
            icon={Share2}
            onClick={(event) => onAction(product, interactionActions.SHARE, event)}
          />
          <ProductActionButton
            action={interactionActions.ADD_TO_CART}
            icon={ShoppingCart}
            active={isInCart}
            onClick={(event) => onAction(product, interactionActions.ADD_TO_CART, event)}
          />
          <ProductActionButton
            action={interactionActions.PURCHASE}
            icon={CreditCard}
            active={isPurchased}
            onClick={(event) => onAction(product, interactionActions.PURCHASE, event)}
          />
          <button
            type="button"
            onClick={() => onOpen(product)}
            className="ml-auto inline-flex items-center gap-1 rounded-md border border-slate-200 px-3 py-2 text-xs font-semibold text-slate-700 hover:border-slate-400"
          >
            <Eye className="h-3.5 w-3.5" aria-hidden="true" />
            Details
          </button>
        </div>
      </div>
    </article>
  );
}

function RecommendationsPanel({
  recommendations,
  metadata,
  searchQuery,
  setSearchQuery,
  filterCategory,
  setFilterCategory,
  sortBy,
  setSortBy,
  categories,
  onRefresh,
  onRunMockSample,
  mode,
  isBusy,
  favorites,
  cartItems,
  purchasedItems,
  actionFeedback,
  onOpenProduct,
  onProductAction,
}) {
  return (
    <section className="space-y-5 rounded-md border border-slate-200 bg-white p-5 shadow-sm">
      <SectionHeader
        eyebrow="Recommendations"
        title="Ranked products"
        description="Search, filter, and sort the recommendation payload returned by the active frontend service."
        action={(
          <button
            type="button"
            onClick={onRefresh}
            disabled={isBusy}
            className="inline-flex min-h-[40px] items-center gap-2 rounded-md border border-slate-300 px-3 py-2 text-sm font-semibold text-slate-700 disabled:cursor-not-allowed disabled:text-slate-400"
          >
            <RefreshCw className={classNames('h-4 w-4', isBusy && 'animate-spin')} aria-hidden="true" />
            Refresh
          </button>
        )}
      />

      <RecommendationMetadata metadata={metadata} onRefresh={onRefresh} />

      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_12rem_13rem]">
        <label className="relative block">
          <span className="sr-only">Search products</span>
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" aria-hidden="true" />
          <input
            type="search"
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="Search by title, brand, category, reason"
            className="w-full rounded-md border border-slate-300 py-2 pl-9 pr-3 text-sm focus:border-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-200"
          />
        </label>
        <label className="block">
          <span className="sr-only">Filter category</span>
          <select
            value={filterCategory}
            onChange={(event) => setFilterCategory(event.target.value)}
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-200"
          >
            <option value="all">All categories</option>
            {categories.map((category) => <option key={category} value={category}>{category}</option>)}
          </select>
        </label>
        <label className="block">
          <span className="sr-only">Sort recommendations</span>
          <select
            value={sortBy}
            onChange={(event) => setSortBy(event.target.value)}
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-200"
          >
            {sortOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
        </label>
      </div>

      {recommendations.length === 0 ? (
        <div className="rounded-md border border-dashed border-slate-300 p-8 text-center">
          <PackageSearch className="mx-auto h-10 w-10 text-slate-400" aria-hidden="true" />
          <p className="mt-3 font-semibold text-slate-900">No recommendations to display</p>
          <p className="mt-1 text-sm text-slate-500">Run an upload flow, refresh recommendations, or switch to Mock for a complete offline demo.</p>
          <div className="mt-4 flex justify-center gap-3">
            <button type="button" onClick={onRefresh} className="rounded-md border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700">
              Refresh
            </button>
            {mode === WORKBENCH_MODES.MOCK && (
              <button type="button" onClick={onRunMockSample} className="rounded-md bg-slate-950 px-4 py-2 text-sm font-semibold text-white">
                Run mock sample
              </button>
            )}
          </div>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {recommendations.map((product) => (
            <ProductCard
              key={product.product_id}
              product={product}
              isFavorite={favorites.has(product.product_id)}
              isInCart={cartItems.has(product.product_id)}
              isPurchased={purchasedItems.has(product.product_id)}
              feedback={actionFeedback[product.product_id]}
              onOpen={onOpenProduct}
              onAction={onProductAction}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function UserActionsPanel({ interactionLog }) {
  const recentLog = interactionLog.slice(0, 8);

  return (
    <section className="space-y-4 rounded-md border border-slate-200 bg-white p-5 shadow-sm">
      <SectionHeader
        eyebrow="User Actions"
        title="Interaction event feedback"
        description="Events are logged through the active service. Mock mode records a frontend-only accepted response."
      />
      {recentLog.length === 0 ? (
        <p className="rounded-md border border-dashed border-slate-300 p-4 text-sm text-slate-500">
          No interaction events in this session yet.
        </p>
      ) : (
        <div className="space-y-2">
          {recentLog.map((entry) => (
            <div key={entry.id} className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-slate-200 p-3">
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold text-slate-900">
                  {actionLabels[entry.action] || entry.action} · {entry.productTitle}
                </p>
                <p className="text-xs text-slate-500">{formatTimestamp(entry.timestamp)} · {entry.mode}</p>
              </div>
              <StatusPill status={entry.status} label={entry.status} />
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function ProductDetailDrawer({
  product,
  actionHistory,
  isFavorite,
  isInCart,
  isPurchased,
  feedback,
  onClose,
  onAction,
}) {
  if (!product) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-slate-950/30" role="presentation">
      <aside
        className="h-full w-full max-w-xl overflow-y-auto bg-white shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-label="Recommendation detail"
      >
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-200 bg-white px-5 py-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Recommendation detail</p>
            <h2 className="text-lg font-semibold text-slate-950">{product.title}</h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close detail drawer"
            className="inline-flex h-10 w-10 items-center justify-center rounded-md border border-slate-200 text-slate-600 hover:border-slate-400 hover:text-slate-950"
          >
            <X className="h-5 w-5" aria-hidden="true" />
          </button>
        </div>

        <div className="space-y-5 p-5">
          <img
            src={product.image_url || fallbackProductImage}
            alt={product.title}
            className="aspect-[4/3] w-full rounded-md bg-slate-100 object-cover"
            onError={(event) => {
              event.currentTarget.src = fallbackProductImage;
            }}
          />

          <div className="grid grid-cols-2 gap-3">
            {[
              ['Brand', product.brand || 'Unavailable'],
              ['Category', product.category || 'Uncategorized'],
              ['Price', formatPrice(product.price, product.currency)],
              ['Rating', product.rating ? product.rating.toFixed(1) : '-'],
              ['Confidence', formatPercent(product.confidence_score)],
              ['Ranking score', formatScore(product.ranking_score)],
            ].map(([label, value]) => (
              <div key={label} className="rounded-md border border-slate-200 p-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
                <p className="mt-1 text-sm font-semibold text-slate-900">{value}</p>
              </div>
            ))}
          </div>

          <div className="rounded-md border border-slate-200 p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Reason</p>
            <p className="mt-2 text-sm text-slate-700">{product.reason || 'Recommendation reason unavailable.'}</p>
          </div>

          {feedback && (
            <div className={classNames(
              'rounded-md border px-3 py-2 text-sm font-medium',
              feedback.status === 'failed'
                ? 'border-red-200 bg-red-50 text-red-700'
                : feedback.status === 'pending'
                  ? 'border-sky-200 bg-sky-50 text-sky-700'
                  : 'border-emerald-200 bg-emerald-50 text-emerald-700'
            )}
            >
              {feedback.message}
            </div>
          )}

          <div className="flex flex-wrap gap-2">
            <ProductActionButton
              action={interactionActions.FAVORITE}
              icon={Heart}
              active={isFavorite}
              onClick={(event) => onAction(product, interactionActions.FAVORITE, event)}
            />
            <ProductActionButton
              action={interactionActions.SHARE}
              icon={Share2}
              onClick={(event) => onAction(product, interactionActions.SHARE, event)}
            />
            <ProductActionButton
              action={interactionActions.ADD_TO_CART}
              icon={ShoppingCart}
              active={isInCart}
              onClick={(event) => onAction(product, interactionActions.ADD_TO_CART, event)}
            />
            <ProductActionButton
              action={interactionActions.PURCHASE}
              icon={CreditCard}
              active={isPurchased}
              onClick={(event) => onAction(product, interactionActions.PURCHASE, event)}
            />
          </div>

          <div className="rounded-md border border-slate-200 p-4">
            <p className="text-sm font-semibold text-slate-900">Session actions for this product</p>
            {actionHistory.length === 0 ? (
              <p className="mt-2 text-sm text-slate-500">No actions recorded for this product yet.</p>
            ) : (
              <div className="mt-3 space-y-2">
                {actionHistory.map((entry) => (
                  <div key={entry.id} className="flex items-center justify-between gap-3 rounded border border-slate-200 p-2 text-sm">
                    <span>{actionLabels[entry.action] || entry.action}</span>
                    <StatusPill status={entry.status} label={entry.status} />
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </aside>
    </div>
  );
}

function Metric({ label, value, icon: Icon }) {
  return (
    <div className="rounded-md border border-slate-200 bg-white p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-medium text-slate-500">{label}</p>
        <Icon className="h-4 w-4 text-slate-400" aria-hidden="true" />
      </div>
      <p className="mt-2 text-2xl font-semibold text-slate-950">{value}</p>
    </div>
  );
}

function HorizontalBar({ label, value, max, suffix = '' }) {
  const width = max > 0 ? Math.max(4, Math.min(100, (safeNumber(value) / max) * 100)) : 0;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between gap-3 text-sm">
        <span className="font-medium text-slate-700">{label}</span>
        <span className="text-slate-500">{value}{suffix}</span>
      </div>
      <div className="h-2 rounded-full bg-slate-100">
        <div className="h-2 rounded-full bg-slate-950" style={{ width: `${width}%` }} />
      </div>
    </div>
  );
}

function AnalyticsPanel({ analytics, analyticsError, mode, onRefresh, onSwitchMock, isLoading }) {
  const normalized = normalizeAnalytics(analytics);
  const actionEntries = Object.entries(normalized.actionCounts);
  const maxActionCount = Math.max(...actionEntries.map(([, value]) => safeNumber(value)), 0);

  return (
    <section className="space-y-5 rounded-md border border-slate-200 bg-white p-5 shadow-sm">
      <SectionHeader
        eyebrow="System & Analytics"
        title="Interaction analytics"
        description="CSS-only visuals summarize action counts, CTR, and conversion without adding a chart dependency."
        action={(
          <button
            type="button"
            onClick={onRefresh}
            disabled={isLoading}
            className="inline-flex min-h-[40px] items-center gap-2 rounded-md border border-slate-300 px-3 py-2 text-sm font-semibold text-slate-700 disabled:cursor-not-allowed disabled:text-slate-400"
          >
            <RefreshCw className={classNames('h-4 w-4', isLoading && 'animate-spin')} aria-hidden="true" />
            Refresh
          </button>
        )}
      />

      {analyticsError && (
        <ErrorCallout
          type="analytics_unavailable"
          message={analyticsError}
          mode={mode}
          onRetry={onRefresh}
          onSwitchMock={onSwitchMock}
        />
      )}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Metric label="Interactions" value={normalized.interactions.toLocaleString()} icon={MousePointerClick} />
        <Metric label="Unique users" value={normalized.uniqueUsers.toLocaleString()} icon={Activity} />
        <Metric label="Unique products" value={normalized.uniqueProducts.toLocaleString()} icon={PackageSearch} />
        <Metric label="CTR" value={formatPercent(normalized.ctr)} icon={Eye} />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <div className="rounded-md border border-slate-200 p-4">
          <p className="text-sm font-semibold text-slate-900">Action count distribution</p>
          <div className="mt-4 space-y-3">
            {actionEntries.length === 0 ? (
              <p className="text-sm text-slate-500">No action counts available.</p>
            ) : actionEntries.map(([action, value]) => (
              <HorizontalBar
                key={action}
                label={actionLabels[action] || action}
                value={safeNumber(value)}
                max={maxActionCount}
              />
            ))}
          </div>
        </div>

        <div className="rounded-md border border-slate-200 p-4">
          <p className="text-sm font-semibold text-slate-900">Funnel rates</p>
          <div className="mt-4 space-y-4">
            <HorizontalBar label="Click-through rate" value={Math.round(normalized.ctr * 100)} max={100} suffix="%" />
            <HorizontalBar label="Conversion rate" value={Math.round(normalized.conversionRate * 100)} max={100} suffix="%" />
          </div>
        </div>
      </div>
    </section>
  );
}

function SystemPanel({
  health,
  healthError,
  mode,
  dashboardUpdatedAt,
  onRefresh,
  onSwitchMock,
  isLoading,
}) {
  const status = getHealthStatus(health);
  const components = getHealthComponents(health);

  return (
    <section className="space-y-5 rounded-md border border-slate-200 bg-white p-5 shadow-sm">
      <SectionHeader
        eyebrow="System"
        title="Gateway and component health"
        description="Live mode reads /health through the public edge. Mock mode uses frontend demo health data."
        action={(
          <button
            type="button"
            onClick={onRefresh}
            disabled={isLoading}
            className="inline-flex min-h-[40px] items-center gap-2 rounded-md border border-slate-300 px-3 py-2 text-sm font-semibold text-slate-700 disabled:cursor-not-allowed disabled:text-slate-400"
          >
            <RefreshCw className={classNames('h-4 w-4', isLoading && 'animate-spin')} aria-hidden="true" />
            Refresh
          </button>
        )}
      />

      {healthError && (
        <ErrorCallout
          type="gateway_offline"
          message={healthError}
          mode={mode}
          onRetry={onRefresh}
          onSwitchMock={onSwitchMock}
        />
      )}

      <div className="flex flex-wrap items-center gap-3 rounded-md border border-slate-200 bg-slate-50 p-4">
        <StatusPill status={status} label={status} />
        <p className="text-sm text-slate-600">Last updated: {formatTimestamp(dashboardUpdatedAt)}</p>
      </div>

      {components.length === 0 ? (
        <p className="rounded-md border border-dashed border-slate-300 p-4 text-sm text-slate-500">
          No component details available from this health response.
        </p>
      ) : (
        <div className="grid gap-3 md:grid-cols-2">
          {components.map((component) => (
            <div key={component.name} className="rounded-md border border-slate-200 p-4">
              <div className="flex items-center justify-between gap-3">
                <p className="font-semibold text-slate-900">{component.name}</p>
                <StatusPill status={component.status} label={component.status} />
              </div>
              {component.message && <p className="mt-2 text-sm text-slate-500">{component.message}</p>}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function VideoCommerceApp() {
  const [activeTab, setActiveTab] = useState('demo');
  const [mode, setMode] = useState(WORKBENCH_MODES.LIVE);
  const service = useMemo(() => createWorkbenchService(mode), [mode]);

  const [userId, setUserId] = useState(() => utils.generateUserId());
  const [priority, setPriority] = useState('normal');
  const [recommendationCount, setRecommendationCount] = useState(6);
  const [device, setDevice] = useState('web');
  const [timeOfDay, setTimeOfDay] = useState('evening');
  const [location, setLocation] = useState('home');
  const [sessionId, setSessionId] = useState(() => `session-${Math.random().toString(36).slice(2, 10)}`);

  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedVideo, setSelectedVideo] = useState('');
  const selectedVideoRef = useRef('');
  const dashboardRequestRef = useRef(0);
  const [dragActive, setDragActive] = useState(false);
  const [phase, setPhase] = useState('idle');
  const [contentStatus, setContentStatus] = useState({ status: 'idle', message: 'Waiting for upload.' });
  const [currentContentId, setCurrentContentId] = useState('');
  const [error, setError] = useState(null);

  const [recommendations, setRecommendations] = useState([]);
  const [metadata, setMetadata] = useState({});
  const [searchQuery, setSearchQuery] = useState('');
  const [filterCategory, setFilterCategory] = useState('all');
  const [sortBy, setSortBy] = useState('confidence_desc');

  const [favorites, setFavorites] = useState(() => new Set());
  const [cartItems, setCartItems] = useState(() => new Set());
  const [purchasedItems, setPurchasedItems] = useState(() => new Set());
  const [interactionLog, setInteractionLog] = useState([]);
  const [actionFeedback, setActionFeedback] = useState({});
  const [selectedProduct, setSelectedProduct] = useState(null);

  const [uploadHistory, setUploadHistory] = useState(() => readUploadHistory());
  const [analytics, setAnalytics] = useState(null);
  const [analyticsError, setAnalyticsError] = useState('');
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState('');
  const [dashboardUpdatedAt, setDashboardUpdatedAt] = useState('');
  const [dashboardLoading, setDashboardLoading] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => () => {
    if (selectedVideoRef.current) {
      URL.revokeObjectURL(selectedVideoRef.current);
    }
  }, []);

  const switchMode = useCallback((nextMode) => {
    setMode(nextMode);
    setError(null);
    setAnalyticsError('');
    setHealthError('');
    setActionFeedback({});
  }, []);

  const buildRequestContext = useCallback((modeOverride = mode) => ({
    session_id: sessionId.trim() || 'session-unknown',
    surface: 'frontend_workbench',
    device,
    time_of_day: timeOfDay,
    location,
    priority,
    demo_mode: modeOverride,
  }), [device, location, mode, priority, sessionId, timeOfDay]);

  const persistHistoryEntry = useCallback((entry) => {
    const nextHistory = upsertUploadHistoryEntry(entry);
    setUploadHistory(nextHistory);
    return nextHistory;
  }, []);

  const loadDashboardData = useCallback(async () => {
    const requestId = dashboardRequestRef.current + 1;
    dashboardRequestRef.current = requestId;
    setDashboardLoading(true);
    const updatedAt = new Date().toISOString();

    try {
      const nextHealth = await service.getHealth();
      if (dashboardRequestRef.current !== requestId) {
        return;
      }
      setHealth(nextHealth);
      setHealthError('');
    } catch (healthRequestError) {
      if (dashboardRequestRef.current !== requestId) {
        return;
      }
      setHealth({ status: 'offline', components: {} });
      setHealthError(getErrorMessage(healthRequestError, 'Gateway health is unavailable.'));
    }

    try {
      const nextAnalytics = await service.getAnalytics();
      if (dashboardRequestRef.current !== requestId) {
        return;
      }
      setAnalytics(nextAnalytics);
      setAnalyticsError('');
    } catch (analyticsRequestError) {
      if (dashboardRequestRef.current !== requestId) {
        return;
      }
      setAnalytics(null);
      setAnalyticsError(getErrorMessage(analyticsRequestError, 'Analytics are unavailable.'));
    }

    if (dashboardRequestRef.current !== requestId) {
      return;
    }
    setDashboardUpdatedAt(updatedAt);
    setDashboardLoading(false);
  }, [service]);

  useEffect(() => {
    loadDashboardData();
  }, [loadDashboardData]);

  const handleFileSelect = useCallback((fileList) => {
    const [file] = Array.from(fileList || []);
    if (!file) {
      return;
    }

    const validation = utils.validateVideoFile(file);
    if (!validation.valid) {
      setError({ type: 'upload_failed', message: validation.error });
      return;
    }

    if (selectedVideoRef.current) {
      URL.revokeObjectURL(selectedVideoRef.current);
      selectedVideoRef.current = '';
    }

    const objectUrl = URL.createObjectURL(file);
    selectedVideoRef.current = objectUrl;
    setSelectedVideo(objectUrl);
    setSelectedFile(file);
    setError(null);
    setPhase('idle');
    setContentStatus({ status: 'ready', message: 'File selected and validated.' });
  }, []);

  const pollContentStatus = useCallback(async (contentId, activeService) => {
    const maxAttempts = mode === WORKBENCH_MODES.MOCK ? 1 : 40;
    const intervalMs = mode === WORKBENCH_MODES.MOCK ? 50 : 2500;

    for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
      const status = await activeService.getContentStatus(contentId);
      setContentStatus(status);

      if (status.status === 'completed') {
        return { success: true, status };
      }

      if (status.status === 'failed') {
        return { success: false, status };
      }

      await new Promise((resolve) => {
        window.setTimeout(resolve, intervalMs);
      });
    }

    return {
      success: false,
      timeout: true,
      status: {
        content_id: contentId,
        status: 'timeout',
        message: 'Processing did not complete before the frontend timeout.',
      },
    };
  }, [mode]);

  const fetchRecommendations = useCallback(async ({
    contentId = currentContentId,
    activeService = service,
    modeOverride = mode,
    moveToResults = true,
  } = {}) => {
    setBusy(true);
    setError(null);
    setPhase((previousPhase) => (previousPhase === 'ready' ? previousPhase : 'recommending'));

    try {
      const response = await activeService.getRecommendations(
        userId.trim() || 'demo_user',
        contentId || null,
        buildRequestContext(modeOverride),
        recommendationCount
      );
      const nextRecommendations = normalizeRecommendations(response);
      setRecommendations(nextRecommendations);
      setMetadata(normalizeMetadata(response));
      setPhase('ready');
      if (moveToResults) {
        setActiveTab('results');
      }
    } catch (recommendationError) {
      setError({
        type: 'gateway_offline',
        message: getErrorMessage(recommendationError, 'Recommendations are unavailable from the active service.'),
      });
      setPhase('failed');
    } finally {
      setBusy(false);
    }
  }, [buildRequestContext, currentContentId, mode, recommendationCount, service, userId]);

  const runAnalysis = useCallback(async (useMockSample = false) => {
    const activeFile = selectedFile || (useMockSample && mode === WORKBENCH_MODES.MOCK ? createMockFile() : null);

    if (!activeFile) {
      setError({
        type: 'upload_failed',
        message: mode === WORKBENCH_MODES.MOCK
          ? 'Choose a file or run the mock sample to start the demo flow.'
          : 'Choose a video file before starting the live upload.',
      });
      return;
    }

    const validation = utils.validateVideoFile(activeFile);
    if (!validation.valid) {
      setError({ type: 'upload_failed', message: validation.error });
      return;
    }

    setBusy(true);
    setError(null);
    setPhase('uploading');
    setContentStatus({ status: 'uploading', message: 'Uploading video through the active service.' });

    try {
      const uploadResponse = await service.uploadVideo(activeFile, userId.trim() || 'demo_user', priority);
      const contentId = uploadResponse.content_id || uploadResponse.contentId;
      if (!contentId) {
        throw new Error('Upload response did not include content_id.');
      }

      setCurrentContentId(contentId);
      const queuedStatus = {
        content_id: contentId,
        status: uploadResponse.status || 'queued',
        message: uploadResponse.message || 'Upload accepted.',
      };
      setContentStatus(queuedStatus);
      const createdAt = uploadResponse.upload_timestamp || new Date().toISOString();
      persistHistoryEntry({
        content_id: contentId,
        filename: activeFile.name || 'video.mp4',
        status: queuedStatus.status,
        created_at: createdAt,
        mode,
      });

      setPhase('processing');
      const pollResult = await pollContentStatus(contentId, service);
      setContentStatus(pollResult.status);
      persistHistoryEntry({
        content_id: contentId,
        filename: activeFile.name || 'video.mp4',
        status: pollResult.status?.status || 'unknown',
        created_at: createdAt,
        mode,
      });

      if (!pollResult.success) {
        const status = pollResult.status?.status || 'failed';
        setPhase(status === 'timeout' ? 'timeout' : 'failed');
        setError({
          type: status === 'timeout' ? 'processing_timeout' : 'upload_failed',
          message: pollResult.status?.message || 'Content processing did not complete successfully.',
        });
        return;
      }

      await fetchRecommendations({ contentId, activeService: service, modeOverride: mode });
    } catch (uploadError) {
      setPhase('failed');
      setContentStatus({ status: 'failed', message: getErrorMessage(uploadError, 'Upload failed.') });
      setError({
        type: mode === WORKBENCH_MODES.LIVE ? 'gateway_offline' : 'upload_failed',
        message: getErrorMessage(uploadError, 'Upload failed.'),
      });
    } finally {
      setBusy(false);
    }
  }, [fetchRecommendations, mode, persistHistoryEntry, pollContentStatus, priority, selectedFile, service, userId]);

  const restoreHistoryEntry = useCallback(async (entry) => {
    const entryMode = entry.mode === WORKBENCH_MODES.MOCK ? WORKBENCH_MODES.MOCK : WORKBENCH_MODES.LIVE;
    const restoreService = createWorkbenchService(entryMode);
    switchMode(entryMode);
    setCurrentContentId(entry.content_id);
    setContentStatus({
      content_id: entry.content_id,
      status: entry.status,
      message: `Restored ${entry.filename} from local upload history.`,
    });
    setPhase(entry.status === 'completed' ? 'ready' : entry.status === 'failed' ? 'failed' : 'processing');
    setError(null);

    if (entry.status === 'completed' || entryMode === WORKBENCH_MODES.MOCK) {
      await fetchRecommendations({
        contentId: entry.content_id,
        activeService: restoreService,
        modeOverride: entryMode,
      });
    } else {
      setActiveTab('demo');
    }
  }, [fetchRecommendations, switchMode]);

  const clearHistory = useCallback(() => {
    setUploadHistory(clearUploadHistory());
  }, []);

  const handleProductAction = useCallback(async (product, action, event) => {
    if (event) {
      event.stopPropagation();
    }

    const productId = product.product_id;
    const actionLabel = actionLabels[action] || action;
    setActionFeedback((previous) => ({
      ...previous,
      [productId]: { status: 'pending', action, message: `${actionLabel} event sending...` },
    }));

    try {
      const response = await service.logInteraction(
        userId.trim() || 'demo_user',
        productId,
        action,
        {
          ...buildRequestContext(),
          content_id: currentContentId || null,
          product_title: product.title,
          impression_id: metadata?.impression_id || null,
          recommendation_position: product.rank ?? null,
          recommendation_ranking_score: product.ranking_score ?? null,
          recommendation_source: (
            product?.source
            || product?.raw?.source
            || product?.raw?.candidate_source
            || null
          ),
        }
      );

      if (action === interactionActions.FAVORITE) {
        setFavorites((previous) => new Set(previous).add(productId));
      }
      if (action === interactionActions.ADD_TO_CART) {
        setCartItems((previous) => new Set(previous).add(productId));
      }
      if (action === interactionActions.PURCHASE) {
        setPurchasedItems((previous) => new Set(previous).add(productId));
      }

      const acceptedMessage = mode === WORKBENCH_MODES.MOCK
        ? 'Accepted by mock event log.'
        : 'Accepted by backend async path.';
      setActionFeedback((previous) => ({
        ...previous,
        [productId]: {
          status: 'accepted',
          action,
          message: response?.message || acceptedMessage,
        },
      }));
      setInteractionLog((previous) => [{
        id: `${Date.now()}-${productId}-${action}`,
        productId,
        productTitle: product.title,
        action,
        status: 'accepted',
        timestamp: new Date().toISOString(),
        mode,
      }, ...previous]);
    } catch (interactionError) {
      const message = getErrorMessage(interactionError, `${actionLabel} event failed.`);
      setActionFeedback((previous) => ({
        ...previous,
        [productId]: { status: 'failed', action, message },
      }));
      setInteractionLog((previous) => [{
        id: `${Date.now()}-${productId}-${action}`,
        productId,
        productTitle: product.title,
        action,
        status: 'failed',
        timestamp: new Date().toISOString(),
        mode,
      }, ...previous]);
    }
  }, [buildRequestContext, currentContentId, metadata, mode, service, userId]);

  const handleOpenProduct = useCallback(async (product) => {
    setSelectedProduct(product);
    await handleProductAction(product, interactionActions.CLICK);
  }, [handleProductAction]);

  const categories = useMemo(() => {
    const values = recommendations.map((product) => product.category).filter(Boolean);
    return Array.from(new Set(values)).sort();
  }, [recommendations]);

  const visibleRecommendations = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    const filtered = recommendations.filter((product) => {
      const matchesCategory = filterCategory === 'all' || product.category === filterCategory;
      const matchesQuery = !query || [
        product.title,
        product.brand,
        product.category,
        product.reason,
      ].some((value) => String(value || '').toLowerCase().includes(query));
      return matchesCategory && matchesQuery;
    });

    return [...filtered].sort((left, right) => {
      if (sortBy === 'price_asc') {
        return safeNumber(left.price) - safeNumber(right.price);
      }
      if (sortBy === 'price_desc') {
        return safeNumber(right.price) - safeNumber(left.price);
      }
      if (sortBy === 'ranking_desc') {
        return safeNumber(right.ranking_score) - safeNumber(left.ranking_score);
      }
      return safeNumber(right.confidence_score) - safeNumber(left.confidence_score);
    });
  }, [filterCategory, recommendations, searchQuery, sortBy]);

  const selectedProductActionHistory = useMemo(() => (
    selectedProduct
      ? interactionLog.filter((entry) => entry.productId === selectedProduct.product_id)
      : []
  ), [interactionLog, selectedProduct]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-950">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-5 sm:px-6 lg:px-8">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-sm font-semibold uppercase tracking-wide text-slate-500">VideoCommerce frontend workbench</p>
              <h1 className="mt-1 text-2xl font-semibold text-slate-950">Recommendation demo console</h1>
            </div>
            <ModeToggle mode={mode} onChange={switchMode} />
          </div>
          <Tabs activeTab={activeTab} onChange={setActiveTab} />
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-5 px-4 py-5 sm:px-6 lg:px-8">
        {activeTab === 'demo' && (
          <UploadAnalysisPanel
            mode={mode}
            userId={userId}
            setUserId={setUserId}
            priority={priority}
            setPriority={setPriority}
            recommendationCount={recommendationCount}
            setRecommendationCount={setRecommendationCount}
            device={device}
            setDevice={setDevice}
            timeOfDay={timeOfDay}
            setTimeOfDay={setTimeOfDay}
            location={location}
            setLocation={setLocation}
            sessionId={sessionId}
            setSessionId={setSessionId}
            selectedFile={selectedFile}
            selectedVideo={selectedVideo}
            dragActive={dragActive}
            setDragActive={setDragActive}
            onFileSelect={handleFileSelect}
            onRunAnalysis={runAnalysis}
            onRunMockSample={() => runAnalysis(true)}
            isBusy={busy}
            phase={phase}
            contentStatus={contentStatus}
            currentContentId={currentContentId}
            error={error}
            onRetry={() => runAnalysis(false)}
            onRefreshHealth={loadDashboardData}
            onSwitchMock={() => switchMode(WORKBENCH_MODES.MOCK)}
            uploadHistory={uploadHistory}
            onRestoreHistory={restoreHistoryEntry}
            onClearHistory={clearHistory}
          />
        )}

        {activeTab === 'results' && (
          <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_24rem]">
            <RecommendationsPanel
              recommendations={visibleRecommendations}
              metadata={metadata}
              searchQuery={searchQuery}
              setSearchQuery={setSearchQuery}
              filterCategory={filterCategory}
              setFilterCategory={setFilterCategory}
              sortBy={sortBy}
              setSortBy={setSortBy}
              categories={categories}
              onRefresh={() => fetchRecommendations()}
              onRunMockSample={() => runAnalysis(true)}
              mode={mode}
              isBusy={busy}
              favorites={favorites}
              cartItems={cartItems}
              purchasedItems={purchasedItems}
              actionFeedback={actionFeedback}
              onOpenProduct={handleOpenProduct}
              onProductAction={handleProductAction}
            />
            <UserActionsPanel interactionLog={interactionLog} />
          </div>
        )}

        {activeTab === 'analytics' && (
          <AnalyticsPanel
            analytics={analytics}
            analyticsError={analyticsError}
            mode={mode}
            onRefresh={loadDashboardData}
            onSwitchMock={() => switchMode(WORKBENCH_MODES.MOCK)}
            isLoading={dashboardLoading}
          />
        )}

        {activeTab === 'system' && (
          <SystemPanel
            health={health}
            healthError={healthError}
            mode={mode}
            dashboardUpdatedAt={dashboardUpdatedAt}
            onRefresh={loadDashboardData}
            onSwitchMock={() => switchMode(WORKBENCH_MODES.MOCK)}
            isLoading={dashboardLoading}
          />
        )}
      </main>

      <ProductDetailDrawer
        product={selectedProduct}
        actionHistory={selectedProductActionHistory}
        isFavorite={selectedProduct ? favorites.has(selectedProduct.product_id) : false}
        isInCart={selectedProduct ? cartItems.has(selectedProduct.product_id) : false}
        isPurchased={selectedProduct ? purchasedItems.has(selectedProduct.product_id) : false}
        feedback={selectedProduct ? actionFeedback[selectedProduct.product_id] : null}
        onClose={() => setSelectedProduct(null)}
        onAction={handleProductAction}
      />
    </div>
  );
}

export default VideoCommerceApp;
