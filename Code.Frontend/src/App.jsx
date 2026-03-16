import { useCallback, useEffect, useRef, useState } from 'react';
import { Upload, ShoppingBag, Star, TrendingUp, Filter, Search, Heart, Share2, Eye } from 'lucide-react';
import { videoApi, systemApi, utils } from './api';

const DEMO_USER_ID = 'demo_user';
const FALLBACK_REASON = 'Recommended based on similarity to the uploaded video.';
const FALLBACK_IMAGE =
  'https://images.unsplash.com/photo-1523275335684-37898b6baf30?auto=format&fit=crop&w=800&q=80';

const normalizeRecommendation = (item) => {
  const productId = item.product_id ?? item.id ?? '';
  const category = item.category?.trim() || 'Uncategorized';
  const confidenceScore = Number(item.confidence_score ?? item.confidence ?? 0);

  return {
    ...item,
    productId,
    category,
    categoryKey: category.toLowerCase(),
    imageUrl: item.image_url ?? item.image ?? FALLBACK_IMAGE,
    confidenceScore: Number.isFinite(confidenceScore) ? confidenceScore : 0,
    rating: Number.isFinite(Number(item.rating)) ? Number(item.rating) : null,
    reason: item.reason ?? item.description ?? FALLBACK_REASON,
    currency: item.currency ?? 'USD',
  };
};

const VideoCommerceApp = () => {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [processingVideo, setProcessingVideo] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterCategory, setFilterCategory] = useState('all');
  const [currentContentId, setCurrentContentId] = useState(null);
  const [systemHealth, setSystemHealth] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const objectUrlRef = useRef(null);

  useEffect(() => {
    const loadDashboardData = async () => {
      const [healthResult, analyticsResult] = await Promise.allSettled([
        systemApi.getHealth(),
        systemApi.getAnalytics(),
      ]);

      setSystemHealth(healthResult.status === 'fulfilled' ? healthResult.value : null);
      setAnalytics(analyticsResult.status === 'fulfilled' ? analyticsResult.value : null);
    };

    loadDashboardData();
  }, []);

  useEffect(() => (
    () => {
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
    }
  ), []);

  const updateSelectedVideo = useCallback((file) => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }

    if (!file) {
      setSelectedVideo(null);
      return;
    }

    const nextObjectUrl = URL.createObjectURL(file);
    objectUrlRef.current = nextObjectUrl;
    setSelectedVideo(nextObjectUrl);
  }, []);

  const fetchRecommendations = useCallback(async (contentId) => {
    try {
      const data = await videoApi.getRecommendations(DEMO_USER_ID, contentId);
      setRecommendations((data.recommendations ?? []).map(normalizeRecommendation));
    } catch (err) {
      console.error('Failed to fetch recommendations:', err);
      setError(utils.getErrorMessage(err, 'Failed to load recommendations.'));
    }
  }, []);

  const handleVideoUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    event.target.value = '';

    const validation = utils.validateVideoFile(file);
    if (!validation.valid) {
      setError(validation.error);
      return;
    }

    updateSelectedVideo(file);
    setProcessingVideo(true);
    setRecommendations([]);
    setError(null);
    setSearchQuery('');
    setFilterCategory('all');

    try {
      const uploadResult = await videoApi.uploadVideo(file, DEMO_USER_ID);
      const contentId = uploadResult.content_id;
      setCurrentContentId(contentId);

      const pollResult = await utils.pollContentStatus(contentId);
      if (pollResult.success) {
        await fetchRecommendations(contentId);
      } else {
        setError('Video processing is taking longer than expected. Please check again shortly.');
      }
    } catch (err) {
      console.error('Error processing video:', err);
      setError(utils.getErrorMessage(err, 'Failed to upload video. Is the backend running?'));
    } finally {
      setProcessingVideo(false);
    }
  };

  const handleProductClick = async (productId, position) => {
    if (!productId) {
      return;
    }

    try {
      await videoApi.logInteraction(
        DEMO_USER_ID,
        String(productId),
        'click',
        {
          position,
          content_id: currentContentId,
        }
      );
    } catch (err) {
      console.error('Error logging interaction:', err);
    }
  };

  const filteredRecommendations = recommendations.filter(item => {
    const matchesSearch = item.title.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = filterCategory === 'all' || item.categoryKey === filterCategory;
    return matchesSearch && matchesCategory;
  });

  const categories = ['all', ...new Set(recommendations.map((item) => item.categoryKey))];
  const categoryLabels = recommendations.reduce((labels, item) => {
    labels[item.categoryKey] = item.category;
    return labels;
  }, {});
  const systemStatus = systemHealth?.status ?? 'offline';
  const healthyComponentCount = Object.values(systemHealth?.components ?? {}).filter(
    (component) => component?.status === 'healthy'
  ).length;
  const totalComponentCount = Object.keys(systemHealth?.components ?? {}).length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <ShoppingBag className="h-8 w-8 text-indigo-600" />
              <h1 className="text-2xl font-bold text-gray-900">VideoCommerce AI</h1>
            </div>
            <nav className="flex space-x-8">
              <a href="#" className="text-gray-700 hover:text-indigo-600 transition-colors">Dashboard</a>
              <a href="#" className="text-gray-700 hover:text-indigo-600 transition-colors">Analytics</a>
              <a href="#" className="text-gray-700 hover:text-indigo-600 transition-colors">Settings</a>
            </nav>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Video Upload and Processing Section */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-semibold text-gray-900 mb-6">Video Content Analysis</h2>
              
              {!selectedVideo ? (
                <div 
                  className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center cursor-pointer hover:border-indigo-400 hover:bg-indigo-50 transition-colors"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-lg text-gray-600 mb-2">Upload a video to get AI-powered product recommendations</p>
                  <p className="text-sm text-gray-500">Supports MP4, MOV, AVI, MKV, and WEBM files up to 500MB</p>
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleVideoUpload}
                    accept="video/*"
                    className="hidden"
                  />
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative rounded-lg overflow-hidden bg-black">
                    <video 
                      src={selectedVideo} 
                      controls 
                      className="w-full h-64 object-contain"
                    />
                    {processingVideo && (
                      <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                        <div className="text-center text-white">
                          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
                          <p className="text-lg font-medium">Analyzing video content...</p>
                          <p className="text-sm opacity-80">Extracting features and generating recommendations</p>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <button
                      onClick={() => {
                        updateSelectedVideo(null);
                        setRecommendations([]);
                        setCurrentContentId(null);
                        setSearchQuery('');
                        setFilterCategory('all');
                        setError(null);
                      }}
                      className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
                    >
                      Upload New Video
                    </button>
                    
                    {recommendations.length > 0 && (
                      <div className="flex items-center space-x-2 text-green-600">
                        <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse"></div>
                        <span className="text-sm font-medium">Analysis Complete</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Processing Status */}
            {processingVideo && (
              <div className="mt-6 bg-blue-50 border border-blue-200 rounded-xl p-4">
                <div className="flex items-center space-x-3">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                  <div>
                    <h3 className="font-medium text-blue-900">AI Processing Pipeline</h3>
                    <p className="text-sm text-blue-700">Extracting visual features, detecting objects, and analyzing content...</p>
                  </div>
                </div>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="mt-6 bg-red-50 border border-red-200 rounded-xl p-4">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-red-700">{error}</p>
                  <button
                    onClick={() => setError(null)}
                    className="text-red-400 hover:text-red-600 text-sm font-medium"
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Sidebar - System Stats */}
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">System Performance</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Service Version</span>
                  <span className="text-sm font-medium text-gray-900">
                    {systemHealth?.version ?? 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Uptime</span>
                  <span className="text-sm font-medium text-blue-600">
                    {utils.formatDuration(systemHealth?.uptime_seconds)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Healthy Components</span>
                  <span className="text-sm font-medium text-blue-600">
                    {totalComponentCount > 0 ? `${healthyComponentCount}/${totalComponentCount}` : '—'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Status</span>
                  {systemStatus !== 'offline' ? (
                    <div className="flex items-center space-x-1">
                      <div
                        className={`h-2 w-2 rounded-full ${
                          systemStatus === 'healthy'
                            ? 'bg-green-500'
                            : systemStatus === 'degraded'
                              ? 'bg-yellow-500'
                              : 'bg-red-500'
                        }`}
                      ></div>
                      <span
                        className={`text-sm font-medium ${
                          systemStatus === 'healthy'
                            ? 'text-green-600'
                            : systemStatus === 'degraded'
                              ? 'text-yellow-600'
                              : 'text-red-600'
                        }`}
                      >
                        {systemStatus}
                      </span>
                    </div>
                  ) : (
                    <div className="flex items-center space-x-1">
                      <div className="h-2 w-2 bg-red-500 rounded-full"></div>
                      <span className="text-sm font-medium text-red-600">Offline</span>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Stats</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Eye className="h-4 w-4 text-blue-500" />
                    <span className="text-sm text-gray-600">Interactions</span>
                  </div>
                  <span className="font-semibold text-gray-900">
                    {analytics?.total_interactions?.toLocaleString() ?? '—'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <ShoppingBag className="h-4 w-4 text-green-500" />
                    <span className="text-sm text-gray-600">Unique Products</span>
                  </div>
                  <span className="font-semibold text-gray-900">
                    {analytics?.unique_products?.toLocaleString() ?? '—'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <TrendingUp className="h-4 w-4 text-purple-500" />
                    <span className="text-sm text-gray-600">Active Users</span>
                  </div>
                  <span className="font-semibold text-gray-900">
                    {analytics?.unique_users?.toLocaleString() ?? '—'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Filter className="h-4 w-4 text-indigo-500" />
                    <span className="text-sm text-gray-600">CTR</span>
                  </div>
                  <span className="font-semibold text-gray-900">
                    {analytics?.ctr != null ? `${(analytics.ctr * 100).toFixed(1)}%` : '—'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Recommendations Section */}
        {recommendations.length > 0 && (
          <div className="mt-8 bg-white rounded-xl shadow-lg p-6">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6">
              <h2 className="text-2xl font-semibold text-gray-900 mb-4 sm:mb-0">
                AI-Generated Recommendations
              </h2>
              
              <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-4">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search products..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  />
                </div>
                
                <select
                  value={filterCategory}
                  onChange={(e) => setFilterCategory(e.target.value)}
                  className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                >
                  {categories.map(category => (
                    <option key={category} value={category}>
                      {category === 'all' ? 'All' : (categoryLabels[category] ?? category)}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              {filteredRecommendations.map((product, index) => (
                <div 
                  key={product.productId} 
                  className="bg-gray-50 rounded-xl overflow-hidden hover:shadow-lg transition-shadow cursor-pointer group"
                  onClick={() => handleProductClick(product.productId, index + 1)}
                >
                  <div className="relative">
                    <img 
                      src={product.imageUrl} 
                      alt={product.title}
                      className="w-full h-48 object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                    <div className="absolute top-2 right-2 bg-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <Heart className="h-4 w-4 text-gray-600 hover:text-red-500" />
                    </div>
                    <div className="absolute top-2 left-2">
                      <div className="bg-indigo-600 text-white text-xs px-2 py-1 rounded-full font-medium">
                        {Math.round(product.confidenceScore * 100)}% Match
                      </div>
                    </div>
                  </div>
                  
                  <div className="p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-indigo-600 font-medium uppercase tracking-wider">
                        {product.category}
                      </span>
                      <div className="flex items-center space-x-1">
                        <Star className="h-3 w-3 text-yellow-400 fill-current" />
                        <span className="text-xs text-gray-600">{product.rating ?? '—'}</span>
                      </div>
                    </div>
                    
                    <h3 className="font-semibold text-gray-900 text-sm mb-2 line-clamp-2">
                      {product.title}
                    </h3>
                    
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-lg font-bold text-gray-900">
                        {utils.formatPrice(product.price, product.currency)}
                      </span>
                      <button className="text-gray-400 hover:text-gray-600">
                        <Share2 className="h-4 w-4" />
                      </button>
                    </div>
                    
                    <p className="text-xs text-gray-600 mb-3">
                      {product.reason}
                    </p>
                    
                    <button className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-2 px-4 rounded-lg font-medium transition-colors text-sm">
                      Add to Cart
                    </button>
                  </div>
                </div>
              ))}
            </div>
            
            {filteredRecommendations.length === 0 && (
              <div className="text-center py-8">
                <Filter className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">No products match your current filters.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoCommerceApp;
