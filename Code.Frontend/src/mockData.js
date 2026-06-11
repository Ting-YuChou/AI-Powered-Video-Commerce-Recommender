const svgImage = (label, background, accent) => {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 360"><rect width="480" height="360" fill="${background}"/><circle cx="370" cy="95" r="76" fill="${accent}" opacity="0.22"/><rect x="72" y="92" width="336" height="176" rx="24" fill="#ffffff" opacity="0.92"/><text x="240" y="180" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="34" font-weight="700" fill="#111827">${label}</text><text x="240" y="220" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#4b5563">Mock product image</text></svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
};

export const mockRecommendations = [
  {
    product_id: 'mock-sneaker-001',
    title: 'AeroStride Knit Runner',
    brand: 'Northline',
    category: 'Footwear',
    rating: 4.7,
    price: 128,
    currency: 'USD',
    image_url: svgImage('Runner', '#dbeafe', '#2563eb'),
    confidence_score: 0.94,
    ranking_score: 0.89,
    source: 'mock_cf+content',
    reason: 'Matches the sporty outfit, neutral color palette, and motion-heavy video scene.',
  },
  {
    product_id: 'mock-jacket-002',
    title: 'TrailShell Utility Jacket',
    brand: 'Vanta Field',
    category: 'Outerwear',
    rating: 4.5,
    price: 164,
    currency: 'USD',
    image_url: svgImage('Jacket', '#dcfce7', '#16a34a'),
    confidence_score: 0.9,
    ranking_score: 0.84,
    source: 'mock_content',
    reason: 'OCR and visual features suggest outdoor performance styling and layered apparel.',
  },
  {
    product_id: 'mock-watch-003',
    title: 'PulseTrack Sport Watch',
    brand: 'TempoLab',
    category: 'Accessories',
    rating: 4.3,
    price: 219,
    currency: 'USD',
    image_url: svgImage('Watch', '#fef3c7', '#d97706'),
    confidence_score: 0.86,
    ranking_score: 0.81,
    source: 'mock_cf',
    reason: 'Recommended from active lifestyle cues and prior session interest in wearable tech.',
  },
  {
    product_id: 'mock-bag-004',
    title: 'Transit Sling Pack',
    brand: 'Carrywell',
    category: 'Bags',
    rating: 4.4,
    price: 78,
    currency: 'USD',
    image_url: svgImage('Sling', '#fce7f3', '#db2777'),
    confidence_score: 0.82,
    ranking_score: 0.77,
    source: 'mock_popularity',
    reason: 'Complements the travel context and has strong click-through history in similar sessions.',
  },
  {
    product_id: 'mock-lamp-005',
    title: 'Halo Desk Lamp',
    brand: 'Luma Works',
    category: 'Home',
    rating: 4.6,
    price: 92,
    currency: 'USD',
    image_url: svgImage('Lamp', '#e0f2fe', '#0284c7'),
    confidence_score: 0.76,
    ranking_score: 0.68,
    source: 'mock_fallback',
    reason: 'Fallback candidate from broad visual similarity when lifestyle and room cues are present.',
  },
  {
    product_id: 'mock-bottle-006',
    title: 'Insulated Flow Bottle',
    brand: 'HydraPeak',
    category: 'Fitness',
    rating: 4.2,
    price: 36,
    currency: 'USD',
    image_url: svgImage('Bottle', '#ede9fe', '#7c3aed'),
    confidence_score: 0.73,
    ranking_score: 0.64,
    source: 'mock_trending',
    reason: 'Low-cost accessory candidate for active scenes and high add-to-cart affinity.',
  },
];

export const createMockUploadResponse = (file) => {
  const normalizedName = file?.name?.replace(/\s+/g, '-').toLowerCase() || 'demo-video.mp4';
  const contentId = `mock-${normalizedName.replace(/[^a-z0-9.-]/g, '')}-${Date.now()}`;

  return {
    content_id: contentId,
    filename: file?.name || 'demo-video.mp4',
    size_bytes: file?.size || 18_400_000,
    status: 'queued',
    message: 'Mock upload accepted. No backend request was sent.',
    upload_timestamp: new Date().toISOString(),
  };
};

export const createMockStatusResponse = (contentId) => ({
  content_id: contentId,
  status: 'completed',
  progress: 100,
  stage: 'recommendation_ready',
  message: 'Mock processing complete.',
  updated_at: new Date().toISOString(),
});

export const createMockRecommendationResponse = ({
  userId,
  contentId,
  context = {},
  k = 10,
} = {}) => {
  const deviceWeight = context.device === 'mobile' ? 0.02 : 0;
  const locationBoost = context.location === 'work' ? -0.01 : 0.01;
  const recommendations = mockRecommendations.slice(0, k).map((item, index) => ({
    ...item,
    confidence_score: Math.max(
      0.5,
      Math.min(0.99, Number((item.confidence_score + deviceWeight + locationBoost - index * 0.005).toFixed(3)))
    ),
    rank: index + 1,
  }));

  return {
    user_id: userId,
    content_id: contentId,
    recommendations,
    metadata: {
      impression_id: `mock-impression-${contentId || 'no-content'}-${k}`,
      total_candidates: 128,
      response_time_ms: 24,
      model_version: 'mock-workbench-v1',
      cache_hit: false,
      fallback: context.location === 'work',
      context,
      mode: 'mock',
    },
  };
};

export const createMockInteractionResponse = (action) => ({
  status: 'accepted',
  action,
  transport: 'mock-event-log',
  message: 'Accepted by mock event log.',
  accepted_at: new Date().toISOString(),
});

export const mockAnalytics = {
  interactions: 1840,
  unique_users: 312,
  unique_products: 78,
  ctr: 0.186,
  conversion_rate: 0.041,
  action_counts: {
    view: 980,
    click: 182,
    favorite: 64,
    share: 39,
    add_to_cart: 48,
    purchase: 18,
  },
};

export const mockHealth = {
  status: 'healthy',
  mode: 'mock',
  components: {
    gateway_api: { status: 'healthy', message: 'Mock gateway available' },
    recommendation_service: { status: 'healthy', message: 'Mock recommendations ready' },
    interaction_ingest_service: { status: 'healthy', message: 'Mock events accepted' },
    content_worker: { status: 'healthy', message: 'Mock processing complete' },
    kafka: { status: 'healthy', message: 'Mock Kafka path simulated' },
  },
};
