import React, { useState, useEffect, useRef } from 'react';
import { Upload, Play, ShoppingBag, Star, TrendingUp, Filter, Search, Heart, Share2, Eye } from 'lucide-react';

const VideoCommerceApp = () => {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [processingVideo, setProcessingVideo] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterCategory, setFilterCategory] = useState('all');
  const [currentContentId, setCurrentContentId] = useState(null);
  const fileInputRef = useRef(null);

  // Sample data for demonstration
  const sampleRecommendations = [
    {
      id: 1,
      title: "Wireless Bluetooth Headphones",
      price: 129.99,
      category: "Electronics",
      rating: 4.5,
      image: "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=300",
      confidence: 0.92,
      reason: "Based on video content analysis"
    },
    {
      id: 2,
      title: "Premium Coffee Maker",
      price: 89.99,
      category: "Home",
      rating: 4.2,
      image: "https://images.unsplash.com/photo-1559056199-641a0ac8b55e?w=300",
      confidence: 0.87,
      reason: "Popular in similar videos"
    },
    {
      id: 3,
      title: "Fitness Tracker Watch",
      price: 199.99,
      category: "Sports",
      rating: 4.7,
      image: "https://images.unsplash.com/photo-1575311373937-040b8e1fd5b6?w=300",
      confidence: 0.85,
      reason: "Matches your viewing history"
    },
    {
      id: 4,
      title: "Leather Crossbody Bag",
      price: 159.99,
      category: "Fashion",
      rating: 4.3,
      image: "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=300",
      confidence: 0.83,
      reason: "Trending product"
    }
  ];

  const handleVideoUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setSelectedVideo(URL.createObjectURL(file));
    setProcessingVideo(true);
    setRecommendations([]);

    // Simulate API call to upload and process video
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('user_id', 'demo_user');

      // Simulate processing delay
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Set sample recommendations
      setRecommendations(sampleRecommendations);
      setCurrentContentId(`content_${Date.now()}`);
      
    } catch (error) {
      console.error('Error processing video:', error);
    } finally {
      setProcessingVideo(false);
    }
  };

  const handleProductClick = async (productId) => {
    // Log interaction
    try {
      const interaction = {
        user_id: 'demo_user',
        product_id: productId.toString(),
        action: 'click',
        context: {
          position: recommendations.findIndex(r => r.id === productId) + 1,
          content_id: currentContentId
        }
      };
      
      console.log('Logging interaction:', interaction);
      // In real implementation, make API call to /api/interactions
      
    } catch (error) {
      console.error('Error logging interaction:', error);
    }
  };

  const filteredRecommendations = recommendations.filter(item => {
    const matchesSearch = item.title.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = filterCategory === 'all' || item.category.toLowerCase() === filterCategory.toLowerCase();
    return matchesSearch && matchesCategory;
  });

  const categories = ['all', 'electronics', 'home', 'sports', 'fashion'];

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
                  <p className="text-sm text-gray-500">Supports MP4, MOV, AVI files up to 500MB</p>
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
                        setSelectedVideo(null);
                        setRecommendations([]);
                        setCurrentContentId(null);
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
          </div>

          {/* Sidebar - System Stats */}
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">System Performance</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Response Time</span>
                  <span className="text-sm font-medium text-green-600">245ms</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Accuracy</span>
                  <span className="text-sm font-medium text-blue-600">92.5%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Model Version</span>
                  <span className="text-sm font-medium text-gray-900">v1.0.0</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Status</span>
                  <div className="flex items-center space-x-1">
                    <div className="h-2 w-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm font-medium text-green-600">Healthy</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Stats</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Eye className="h-4 w-4 text-blue-500" />
                    <span className="text-sm text-gray-600">Videos Processed</span>
                  </div>
                  <span className="font-semibold text-gray-900">1,247</span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <ShoppingBag className="h-4 w-4 text-green-500" />
                    <span className="text-sm text-gray-600">Products Matched</span>
                  </div>
                  <span className="font-semibold text-gray-900">8,932</span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <TrendingUp className="h-4 w-4 text-purple-500" />
                    <span className="text-sm text-gray-600">Conversion Rate</span>
                  </div>
                  <span className="font-semibold text-gray-900">12.8%</span>
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
                      {category.charAt(0).toUpperCase() + category.slice(1)}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              {filteredRecommendations.map((product) => (
                <div 
                  key={product.id} 
                  className="bg-gray-50 rounded-xl overflow-hidden hover:shadow-lg transition-shadow cursor-pointer group"
                  onClick={() => handleProductClick(product.id)}
                >
                  <div className="relative">
                    <img 
                      src={product.image} 
                      alt={product.title}
                      className="w-full h-48 object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                    <div className="absolute top-2 right-2 bg-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <Heart className="h-4 w-4 text-gray-600 hover:text-red-500" />
                    </div>
                    <div className="absolute top-2 left-2">
                      <div className="bg-indigo-600 text-white text-xs px-2 py-1 rounded-full font-medium">
                        {Math.round(product.confidence * 100)}% Match
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
                        <span className="text-xs text-gray-600">{product.rating}</span>
                      </div>
                    </div>
                    
                    <h3 className="font-semibold text-gray-900 text-sm mb-2 line-clamp-2">
                      {product.title}
                    </h3>
                    
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-lg font-bold text-gray-900">
                        ${product.price}
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