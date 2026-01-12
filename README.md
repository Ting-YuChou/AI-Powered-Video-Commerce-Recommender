# 🎬 AI-Powered Video Commerce Recommender

A full-stack AI recommendation system that analyzes video content to deliver personalized product recommendations. Built with cutting-edge machine learning models including CLIP, FAISS vector search, and collaborative filtering.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2+-61DAFB.svg)](https://reactjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

### Core Capabilities
- **🎥 Intelligent Video Processing**: Automatically extracts keyframes, analyzes visual content, and generates embeddings using CLIP models
- **🤖 Multi-Strategy Recommendations**: Combines collaborative filtering, content-based filtering, and popularity-based ranking
- **⚡ Real-Time Vector Search**: Ultra-fast similarity search using FAISS with HNSW indexing
- **🎯 Personalized Ranking**: Neural ranking model with multi-objective optimization (CTR, CVR, GMV)
- **📊 Feature Store**: High-performance caching with Redis for user and content features
- **🔄 Adaptive Learning**: Continuously learns from user interactions to improve recommendations
- **🌐 RESTful API**: Production-ready FastAPI backend with automatic OpenAPI documentation
- **💻 Modern Frontend**: Beautiful React UI with Tailwind CSS

### Advanced Features
- **Content Understanding**: OCR text extraction, scene detection, and multimodal embeddings
- **Diversity & Exploration**: Balanced recommendations with category diversity
- **Trending Detection**: Time-decayed popularity scoring for trending items
- **Cold Start Handling**: Smart strategies for new users and new content
- **Health Monitoring**: Comprehensive health checks and performance metrics
- **Docker Support**: Complete containerization with multi-stage builds


## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10 or higher
- **Node.js**: 16.0 or higher (for frontend)
- **Redis**: 6.0 or higher (optional but recommended)
- **FFmpeg**: For video processing
- **System Memory**: 8GB+ recommended
- **GPU**: Optional but recommended for faster inference

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/Ting-YuChou/AI-Video-Commerce.git
cd AI-Video-Commerce
```

#### 2. Backend Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your configuration
```

#### 3. Frontend Setup

```bash
cd Code.Frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

#### 4. Start the Application

**Option A: Using the startup script (Recommended)**
```bash
# Makes the script executable
chmod +x startup.sh

# Start everything
./startup.sh start
```

**Option B: Manual start**
```bash
# Activate virtual environment
source venv/bin/activate

# Start backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, start frontend
cd Code.Frontend
npm run dev
```

#### 5. Access the Application

- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Frontend UI**: http://localhost:5173
- **Health Check**: http://localhost:8000/health

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Docker Build

```bash
# Build the image
docker build -t video-commerce-recommender .

# Run the container
docker run -p 8000:8000 \
  -e REDIS_HOST=redis \
  -e MODEL_DEVICE=cpu \
  video-commerce-recommender
```

## 📚 API Documentation

### Key Endpoints

#### Get Recommendations
```http
POST /api/recommendations
Content-Type: application/json

{
  "user_id": "user123",
  "context": {
    "category": "electronics",
    "page": "home"
  },
  "limit": 20
}
```

#### Upload Video Content
```http
POST /api/content/upload
Content-Type: multipart/form-data

file: video.mp4
metadata: {"product_id": "prod123", "category": "fashion"}
```

#### Track User Interaction
```http
POST /api/interactions
Content-Type: application/json

{
  "user_id": "user123",
  "content_id": "content456",
  "interaction_type": "click",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Health Check
```http
GET /health

Response:
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "api": "healthy",
    "redis": "healthy",
    "ml_models": "healthy"
  }
}
```

For complete API documentation, visit: http://localhost:8000/docs

## ⚙️ Configuration

### Environment Variables

Key configuration options (see `.env.example` for complete list):

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Model Configuration
MODEL_DEVICE=auto  # auto, cpu, or cuda
MODEL_CLIP_MODEL=openai/clip-vit-large-patch14
MODEL_BATCH_SIZE=32
MODEL_CACHE_DIR=/tmp/models

# Vector Search
VECTOR_INDEX_PATH=/tmp/vector_index.faiss
VECTOR_EMBEDDING_DIM=512

# Recommendation Weights
RECOMMENDATION_CF_WEIGHT=0.4
RECOMMENDATION_CONTENT_WEIGHT=0.3
RECOMMENDATION_POPULARITY_WEIGHT=0.3

# Caching
CACHE_ENABLE_CACHING=true
CACHE_DEFAULT_TTL=3600
CACHE_ADAPTIVE_TTL=true

# Environment
ENVIRONMENT=production  # development, production, or test
```

### Advanced Configuration

For detailed configuration options, edit `config.py` or provide a `config.json` file:

```json
{
  "model": {
    "clip_model": "openai/clip-vit-large-patch14",
    "embedding_dim": 512,
    "batch_size": 32
  },
  "recommendation": {
    "cf_weight": 0.4,
    "content_weight": 0.3,
    "popularity_weight": 0.3,
    "enable_diversity": true
  },
  "ranking": {
    "ctr_weight": 1.0,
    "cvr_weight": 2.0,
    "gmv_weight": 3.0
  }
}
```


### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint
flake8 .
mypy .

# Frontend linting
cd Code.Frontend
npm run lint
```

### Adding New Features

1. **New Recommendation Strategy**: Extend `RecommendationEngine` in `recommender.py`
2. **Custom Ranking**: Modify `RankingModel` in `ranking.py`
3. **Video Processing**: Enhance `ContentProcessor` in `content_processor.py`
4. **API Endpoints**: Add routes in `app.py`

## 📊 Performance


### Optimization Tips

1. **Enable GPU**: Set `MODEL_DEVICE=cuda` for 10x faster inference
2. **Use FAISS GPU**: Install `faiss-gpu` for faster vector search
3. **Scale Workers**: Increase `API_WORKERS` for more concurrent requests
4. **Redis Cluster**: Use Redis Cluster for distributed caching
5. **Model Quantization**: Enable `MODEL_ENABLE_QUANTIZATION=true`

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Write tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI CLIP**: For multimodal embeddings
- **Facebook FAISS**: For efficient vector search
- **FastAPI**: For the amazing web framework
- **Hugging Face**: For model hosting and transformers library

