#!/bin/bash

# AI-Powered Video Commerce Recommender - Startup Script
# ======================================================
# This script handles the complete startup process for the video commerce
# recommender system, including environment setup, dependency checks,
# service initialization, and health monitoring.

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$PROJECT_DIR/app.pid"
ENV_FILE="$PROJECT_DIR/.env"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8000"
DEFAULT_WORKERS="1"
DEFAULT_LOG_LEVEL="info"
DEFAULT_RELOAD="false"

# Print colored output
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}${NC}"
}

print_info() {
    print_message "$BLUE" "INFO: $1"
}

print_success() {
    print_message "$GREEN" "SUCCESS: $1"
}

print_warning() {
    print_message "$YELLOW" "WARNING: $1"
}

print_error() {
    print_message "$RED" "ERROR: $1"
}

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "============================================================"
    echo "    AI-Powered Video Commerce Recommender System"
    echo "============================================================"
    echo "    Starting up recommendation services..."
    echo "    Project Directory: $PROJECT_DIR"
    echo "    Timestamp: $(date)"
    echo "============================================================"
    echo -e "${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_system_requirements() {
    print_info "Checking system requirements..."
    
    local errors=0
    
    # Check Python
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
        if [[ $(echo "$PYTHON_VERSION >= 3.10" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
            print_success "Python $PYTHON_VERSION found"
        else
            print_error "Python 3.10+ required, found $PYTHON_VERSION"
            errors=$((errors + 1))
        fi
    else
        print_error "Python 3 not found"
        errors=$((errors + 1))
    fi
    
    # Check pip
    if command_exists pip3; then
        print_success "pip3 found"
    else
        print_error "pip3 not found"
        errors=$((errors + 1))
    fi
    
    # Check Redis (optional but recommended)
    if command_exists redis-server; then
        print_success "Redis server found"
    else
        print_warning "Redis server not found - feature store may not work"
    fi
    
    # Check Git (for dependency management)
    if command_exists git; then
        print_success "Git found"
    else
        print_warning "Git not found - may affect some installations"
    fi
    
    # Check system resources
    check_system_resources
    
    if [ $errors -gt 0 ]; then
        print_error "System requirements not met. Please install missing dependencies."
        exit 1
    fi
    
    print_success "System requirements check passed"
}

# Check system resources
check_system_resources() {
    print_info "Checking system resources..."
    
    # Check available memory (Linux/macOS)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        MEMORY_GB=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
    else
        MEMORY_GB="unknown"
    fi
    
    if [ "$MEMORY_GB" != "unknown" ]; then
        if [ "$MEMORY_GB" -lt 4 ]; then
            print_warning "Low memory detected: ${MEMORY_GB}GB (recommended: 8GB+)"
        else
            print_success "Memory: ${MEMORY_GB}GB available"
        fi
    fi
    
    # Check disk space
    DISK_SPACE=$(df -BG "$PROJECT_DIR" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')
    if [ "$DISK_SPACE" -lt 5 ]; then
        print_warning "Low disk space: ${DISK_SPACE}GB (recommended: 10GB+)"
    else
        print_success "Disk space: ${DISK_SPACE}GB available"
    fi
}

# Setup environment
setup_environment() {
    print_info "Setting up environment..."
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    print_success "Log directory created: $LOG_DIR"
    
    # Create necessary directories
    mkdir -p "$PROJECT_DIR/tmp"
    mkdir -p "$PROJECT_DIR/models"
    mkdir -p "$PROJECT_DIR/data"
    mkdir -p "$PROJECT_DIR/uploads"
    
    # Load environment variables
    if [ -f "$ENV_FILE" ]; then
        print_info "Loading environment variables from $ENV_FILE"
        export $(cat "$ENV_FILE" | grep -v '^#' | xargs)
        print_success "Environment variables loaded"
    else
        print_warning ".env file not found, using defaults"
        # Create default .env file
        create_default_env_file
    fi
    
    # Set environment variables with defaults
    export API_HOST="${API_HOST:-$DEFAULT_HOST}"
    export API_PORT="${API_PORT:-$DEFAULT_PORT}"
    export API_WORKERS="${API_WORKERS:-$DEFAULT_WORKERS}"
    export LOG_LEVEL="${LOG_LEVEL:-$DEFAULT_LOG_LEVEL}"
    export API_RELOAD="${API_RELOAD:-$DEFAULT_RELOAD}"
    
    print_success "Environment setup completed"
}

# Create default environment file
create_default_env_file() {
    print_info "Creating default .env file..."
    
    cat > "$ENV_FILE" << EOF
# AI Video Commerce Recommender Configuration
# ===========================================

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
API_RELOAD=false
API_WORKERS=1

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Model Configuration
MODEL_DEVICE=auto
MODEL_CACHE_DIR=/tmp/models
MODEL_BATCH_SIZE=32
MODEL_CLIP_MODEL=openai/clip-vit-large-patch14

# Vector Search Configuration
VECTOR_INDEX_PATH=/tmp/vector_index.faiss
VECTOR_EMBEDDING_DIM=512

# Data Configuration
DATA_LOAD_SAMPLE_DATA=true
DATA_UPLOAD_DIR=/tmp/uploads
DATA_MAX_FILE_SIZE=104857600

# Monitoring Configuration
MONITORING_LOG_LEVEL=INFO
MONITORING_ENABLE_METRICS=true

# Cache Configuration
CACHE_ENABLE_CACHING=true
CACHE_DEFAULT_TTL=3600
CACHE_ADAPTIVE_TTL=true

# Environment
ENVIRONMENT=development
EOF
    
    print_success "Default .env file created"
}

# Install Python dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        print_error "requirements.txt not found at $REQUIREMENTS_FILE"
        exit 1
    fi
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_info "Creating Python virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip > "$LOG_DIR/pip_upgrade.log" 2>&1
    
    # Install requirements
    print_info "Installing requirements (this may take a few minutes)..."
    pip install -r "$REQUIREMENTS_FILE" > "$LOG_DIR/pip_install.log" 2>&1
    
    if [ $? -eq 0 ]; then
        print_success "Dependencies installed successfully"
    else
        print_error "Failed to install dependencies. Check $LOG_DIR/pip_install.log"
        exit 1
    fi
}

# Check services
check_services() {
    print_info "Checking external services..."
    
    # Check Redis connection
    if [ -n "${REDIS_HOST}" ]; then
        REDIS_PORT="${REDIS_PORT:-6379}"
        print_info "Checking Redis connection at ${REDIS_HOST}:${REDIS_PORT}..."
        
        if command_exists redis-cli; then
            if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping >/dev/null 2>&1; then
                print_success "Redis connection successful"
            else
                print_warning "Redis connection failed - starting local Redis if available"
                start_redis_if_available
            fi
        else
            print_warning "redis-cli not found - cannot test Redis connection"
        fi
    fi
}

# Start Redis if available and not running
start_redis_if_available() {
    if command_exists redis-server; then
        if ! pgrep redis-server >/dev/null; then
            print_info "Starting Redis server..."
            redis-server --daemonize yes --port "${REDIS_PORT:-6379}" >/dev/null 2>&1
            sleep 2
            
            if pgrep redis-server >/dev/null; then
                print_success "Redis server started"
            else
                print_warning "Failed to start Redis server"
            fi
        else
            print_success "Redis server already running"
        fi
    fi
}

# Initialize application data
initialize_data() {
    print_info "Initializing application data..."
    
    # Check if sample data should be loaded
    if [ "${DATA_LOAD_SAMPLE_DATA:-true}" == "true" ]; then
        print_info "Generating sample data..."
        python3 -c "
import asyncio
import sys
import os
sys.path.append('$PROJECT_DIR')

async def init_data():
    try:
        from config import get_config
        from feature_store import FeatureStore
        from vector_search import VectorSearchEngine
        import data
        
        config = get_config()
        feature_store = FeatureStore(config.redis_config, config.cache_config)
        vector_search = VectorSearchEngine(config.vector_config)
        
        await feature_store.initialize()
        await vector_search.load_index()
        
        result = await data.initialize_sample_data(feature_store, vector_search)
        print(f'Sample data initialized: {result}')
        
    except Exception as e:
        print(f'Error initializing data: {e}')
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(init_data())
" > "$LOG_DIR/data_init.log" 2>&1

        if [ $? -eq 0 ]; then
            print_success "Sample data initialized"
        else
            print_warning "Sample data initialization failed. Check $LOG_DIR/data_init.log"
        fi
    fi
}

# Health check function
health_check() {
    local host="${1:-$API_HOST}"
    local port="${2:-$API_PORT}"
    local max_attempts="${3:-30}"
    local attempt=0
    
    print_info "Performing health check on http://${host}:${port}/health"
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "http://${host}:${port}/health" >/dev/null 2>&1; then
            print_success "Health check passed"
            return 0
        fi
        
        attempt=$((attempt + 1))
        if [ $attempt -lt $max_attempts ]; then
            sleep 2
        fi
    done
    
    print_error "Health check failed after $max_attempts attempts"
    return 1
}

# Start the application
start_application() {
    print_info "Starting AI Video Commerce Recommender..."
    
    # Check if already running
    if [ -f "$PID_FILE" ] && ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
        print_warning "Application already running (PID: $(cat $PID_FILE))"
        return 0
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Start the application
    print_info "Starting FastAPI server on ${API_HOST}:${API_PORT}..."
    
    if [ "$API_RELOAD" == "true" ]; then
        # Development mode with reload
        uvicorn app:app \
            --host "$API_HOST" \
            --port "$API_PORT" \
            --log-level "$LOG_LEVEL" \
            --reload \
            --reload-dir "$PROJECT_DIR" \
            > "$LOG_DIR/app.log" 2>&1 &
    else
        # Production mode
        uvicorn app:app \
            --host "$API_HOST" \
            --port "$API_PORT" \
            --log-level "$LOG_LEVEL" \
            --workers "$API_WORKERS" \
            > "$LOG_DIR/app.log" 2>&1 &
    fi
    
    APP_PID=$!
    echo $APP_PID > "$PID_FILE"
    
    print_success "Application started with PID: $APP_PID"
    print_info "Logs available at: $LOG_DIR/app.log"
    print_info "API Documentation: http://${API_HOST}:${API_PORT}/docs"
    
    # Wait a moment for startup
    sleep 3
    
    # Perform health check
    if health_check; then
        print_success "Application is healthy and ready!"
        print_info "API Endpoints:"
        print_info "  Health Check: http://${API_HOST}:${API_PORT}/health"
        print_info "  Recommendations: http://${API_HOST}:${API_PORT}/api/recommendations"
        print_info "  Documentation: http://${API_HOST}:${API_PORT}/docs"
    else
        print_error "Application failed health check"
        stop_application
        exit 1
    fi
}

# Stop the application
stop_application() {
    print_info "Stopping AI Video Commerce Recommender..."
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            print_info "Stopping application (PID: $PID)..."
            kill "$PID"
            
            # Wait for graceful shutdown
            local attempt=0
            while [ $attempt -lt 10 ] && ps -p "$PID" > /dev/null 2>&1; do
                sleep 1
                attempt=$((attempt + 1))
            done
            
            # Force kill if still running
            if ps -p "$PID" > /dev/null 2>&1; then
                print_warning "Force killing application..."
                kill -9 "$PID"
            fi
            
            print_success "Application stopped"
        else
            print_warning "Application not running"
        fi
        
        rm -f "$PID_FILE"
    else
        print_warning "PID file not found"
    fi
}

# Restart the application
restart_application() {
    print_info "Restarting AI Video Commerce Recommender..."
    stop_application
    sleep 2
    start_application
}

# Show application status
show_status() {
    print_info "Checking application status..."
    
    if [ -f "$PID_FILE" ] && ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
        PID=$(cat "$PID_FILE")
        print_success "Application is running (PID: $PID)"
        
        # Check if responding to HTTP requests
        if health_check "$API_HOST" "$API_PORT" 1; then
            print_success "Application is responding to requests"
        else
            print_warning "Application is running but not responding to requests"
        fi
    else
        print_info "Application is not running"
        return 1
    fi
}

# Show logs
show_logs() {
    local lines="${1:-50}"
    
    if [ -f "$LOG_DIR/app.log" ]; then
        print_info "Showing last $lines lines of application logs:"
        echo -e "${BLUE}========================================${NC}"
        tail -n "$lines" "$LOG_DIR/app.log"
        echo -e "${BLUE}========================================${NC}"
    else
        print_warning "Log file not found: $LOG_DIR/app.log"
    fi
}

# Main function
main() {
    case "${1:-start}" in
        "start")
            print_banner
            check_system_requirements
            setup_environment
            install_dependencies
            check_services
            initialize_data
            start_application
            ;;
        "stop")
            stop_application
            ;;
        "restart")
            restart_application
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "${2:-50}"
            ;;
        "health")
            if health_check; then
                print_success "Application is healthy"
                exit 0
            else
                print_error "Application is unhealthy"
                exit 1
            fi
            ;;
        "setup")
            print_banner
            check_system_requirements
            setup_environment
            install_dependencies
            print_success "Setup completed successfully!"
            ;;
        "help"|"-h"|"--help")
            echo "Usage: $0 {start|stop|restart|status|logs|health|setup|help}"
            echo ""
            echo "Commands:"
            echo "  start     Start the application (default)"
            echo "  stop      Stop the application"
            echo "  restart   Restart the application"
            echo "  status    Show application status"
            echo "  logs      Show application logs (default: 50 lines)"
            echo "  health    Perform health check"
            echo "  setup     Setup environment and dependencies only"
            echo "  help      Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  API_HOST      Host to bind to (default: 0.0.0.0)"
            echo "  API_PORT      Port to bind to (default: 8000)"
            echo "  API_WORKERS   Number of workers (default: 1)"
            echo "  LOG_LEVEL     Log level (default: info)"
            echo ""
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Trap signals for graceful shutdown
trap 'print_info "Received interrupt signal, shutting down..."; stop_application; exit 0' INT TERM

# Run main function
main "$@"