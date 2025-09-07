# AutoCare Backend - Concurrent Edition

AutoCare Backend is a high-performance, concurrency-optimized FastAPI-based backend service designed to power intelligent automotive care applications for multiple simultaneous users. It provides RESTful APIs and WebSocket support for real-time communication, leveraging an advanced concurrent architecture for enterprise-scale deployment.

## Project Description

AutoCare Backend Concurrent Edition is an enterprise-grade automotive diagnostic and care management system that combines artificial intelligence, machine learning, and automotive expertise to provide intelligent vehicle diagnostics, maintenance recommendations, and real-time support to thousands of concurrent users. The system is architected for high availability, horizontal scaling, and optimal performance under heavy load.

### Key Capabilities

**Concurrent User Support:**

- Handles 10,000+ concurrent user sessions with Redis-based session management
- Database connection pooling (10-100 connections) for optimal performance
- WebSocket connection management with user isolation and load balancing
- Agent pooling system for AI workload distribution
- Distributed caching with local and Redis-based multi-level caching

**Intelligent Diagnostic System:**

- AI-powered symptom analysis and diagnostic recommendations
- Multi-agent architecture with specialized diagnostic agents and load balancing
- Real-time decision tree navigation for systematic troubleshooting
- Integration with comprehensive automotive knowledge base
- Concurrent processing of multiple diagnostic sessions

**Advanced Performance Features:**

- Enhanced session management with automatic cleanup and resource optimization
- Rate limiting with adaptive algorithms and burst protection
- Real-time health monitoring and metrics collection
- Comprehensive error handling and recovery mechanisms
- Production-ready deployment with Docker containerization

**Real-time Communication:**

- WebSocket-powered live chat sessions supporting concurrent users
- Interactive diagnostic conversations with AI agents
- Real-time status updates for background processing tasks
- User-isolated connection tracking and message routing
- Scalable broadcast capabilities for system notifications

**Knowledge Management:**

- Extensive automotive reference library with PDF document processing
- Vector-based semantic search across automotive manuals and documentation
- Ground truth knowledge base with verified diagnostic procedures
- Concurrent access to knowledge base with optimized caching

**Enterprise Architecture:**

- Microservices-based design optimized for horizontal scaling
- Background task processing with agent pools for heavy operations
- Redis distributed caching for multi-instance deployments
- Enhanced database integration with connection pooling and failover
- Comprehensive monitoring, logging, and metrics collection

### Target Use Cases

- **Enterprise Service Centers:** Support hundreds of simultaneous diagnostic sessions
- **Automotive Platforms:** Power large-scale vehicle diagnostic applications
- **Educational Institutions:** Concurrent access for automotive technology training
- **Fleet Management:** Simultaneous monitoring and diagnosis of multiple vehicles
- **SaaS Providers:** White-label automotive diagnostic services
- **Fleet Management:** Centralized vehicle health monitoring and maintenance scheduling

## Version 1 Status

**Version 1 is complete and production-ready for MVP use.**

- All core features implemented and tested
- Robust error handling, dependency injection, and startup/shutdown checks
- Modular, SOLID-compliant architecture for easy extension

## Features

- **REST API**: Organized under `/api/v1` for versioning and modularity.
- **WebSocket Support**: Real-time communication endpoint at `/ws`.
- **Unified Error Handling**: Consistent error responses for HTTP and generic exceptions.
- **Concurrent User Support**: Built for 10,000+ simultaneous users with session management and connection pooling.
- **Enhanced Performance**: Database connection pooling, agent load balancing, and distributed caching.
- **Rate Limiting**: Advanced rate limiting with adaptive algorithms and burst protection.
- **Database Integration**: Enhanced Supabase integration with connection pooling and failover.
- **Redis Distributed Caching**: Multi-level caching with local and distributed storage.
- **Agent Pool Architecture**: Load-balanced AI agents for concurrent diagnostic processing.
- **Real-time WebSocket**: User-isolated WebSocket connections with heartbeat monitoring.
- **Background Task Processing**: Concurrent task processing with agent pool management.
- **Performance Monitoring**: Comprehensive metrics, health checks, and performance analytics.
- **Enterprise Logging**: Advanced logging with correlation IDs and audit trails.
- **Production Deployment**: Docker-based deployment with horizontal scaling support.
- **Session Management**: Redis-based distributed session storage for multi-instance deployments.
- **Health Monitoring**: Real-time component health checks and system metrics.

## Architecture Overview

The AutoCare Backend Concurrent Edition is designed with enterprise-grade concurrency patterns:

- **Session Manager**: Redis-based distributed session storage with automatic cleanup
- **Connection Pool**: Database connection pooling with automatic scaling (10-100 connections)
- **Agent Pool**: Load-balanced AI agent instances for concurrent processing
- **WebSocket Manager**: User-isolated WebSocket connections with connection limits
- **Chat Service**: Concurrent chat handling with conversation caching
- **Health Monitoring**: Real-time system health checks and performance metrics

## Project Structure

```
main.py                           # FastAPI app with concurrent architecture
app/
  api/v1/routes.py               # API route definitions
  core/config.py                 # Enhanced configuration with concurrent settings
  db/enhanced_connection_pool.py # Database connection pooling
  utils/session_manager.py       # Distributed session management
  utils/concurrent_websocket.py  # Enhanced WebSocket management
  services/concurrent_chat_service.py # Concurrent chat service
  agents/agent_pool.py           # Agent load balancing and pooling
  CRUD/                          # CRUD operations for entities
  schemas/                       # Data schemas (Pydantic)
  services/                      # Business logic services
  utils/                         # Enhanced utilities and middleware
car_data/                        # Automotive reference documents
tests/                           # Unit and integration tests
docker-compose.yml               # Production deployment with Redis and Milvus
Dockerfile                       # Optimized container for concurrent users
test_concurrent_system.py        # Comprehensive concurrent testing suite
DEPLOYMENT_GUIDE.md              # Production deployment guide
```

## Getting Started

### Quick Start with Docker (Recommended)

1. **Clone the repository:**

   ```sh
   git clone https://github.com/AbrehamGebremedhin/autocare-backend
   cd autocare-backend
   ```

2. **Configure environment variables:**

   ```sh
   cp .env.template .env
   # Edit .env with your actual API keys and credentials
   ```

3. **Deploy with Docker Compose:**

   ```sh
   docker-compose up -d
   ```

4. **Verify deployment:**
   ```sh
   curl http://localhost:8000/health
   curl http://localhost:8000/metrics
   ```

### Manual Installation

1. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**

   - Copy `.env.template` to `.env` and configure all required values
   - Set up Supabase, Redis, Milvus, Gemini, and YouTube credentials

3. **Start external services:**

   ```sh
   # Start Redis
   redis-server

   # Start Milvus (see Milvus documentation)
   # Configure Supabase database
   ```

4. **Run the application:**
   ```sh
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

### Testing Concurrent Performance

Run the comprehensive testing suite to validate concurrent user support:

```sh
# Install testing dependencies
pip install aiohttp websockets

# Run concurrent system tests
python test_concurrent_system.py --http-requests 200 --websocket-connections 50

# Run specific test types
python test_concurrent_system.py --test-type http
python test_concurrent_system.py --test-type websocket
python test_concurrent_system.py --test-type session
```

### API Endpoints

- **Root:** `GET /` — Welcome message
- **API v1:** `/api/v1/...` — Main REST endpoints (chat, car data, search, etc.)
- **WebSocket:** `ws://<host>/ws` — Real-time communication
- **Background Task Status:** `GET /api/v1/task-status/{task_id}` — Check Celery task status

### Testing

Run all tests with:

```sh
pytest
```

## Notable Dependencies

- fastapi, uvicorn, celery, slowapi, supabase, redis, pydantic, playwright, Crawl4AI, google-api-python-client, pypdf, pytest, and more (see `requirements.txt`).

## Notes

- For PDF and web scraping, the backend uses Crawl4AI and Playwright.
- The system is modular and can be extended with new agents, services, or integrations.
- For local development, ensure all environment variables are set and required services (Supabase, Redis) are running.
- The orchestrator agent and all major services are initialized and gracefully shut down on app startup/shutdown.
- Test coverage is provided for all major modules in the `tests/` directory.
