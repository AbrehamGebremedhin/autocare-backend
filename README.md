# AutoCare Backend

AutoCare Backend is a comprehensive FastAPI-based backend service designed to power intelligent automotive care applications. It provides RESTful APIs and WebSocket support for real-time communication, leveraging a modular architecture for scalability and maintainability.

## Project Description

AutoCare Backend is an advanced automotive diagnostic and care management system that combines artificial intelligence, machine learning, and automotive expertise to provide intelligent vehicle diagnostics, maintenance recommendations, and real-time support. The system is designed to bridge the gap between automotive professionals and vehicle owners by providing accessible, accurate, and actionable automotive insights.

### Key Capabilities

**Intelligent Diagnostic System:**

- AI-powered symptom analysis and diagnostic recommendations
- Multi-agent architecture with specialized diagnostic agents
- Real-time decision tree navigation for systematic troubleshooting
- Integration with comprehensive automotive knowledge base

**Knowledge Management:**

- Extensive automotive reference library with PDF document processing
- Vector-based semantic search across automotive manuals and documentation
- Ground truth knowledge base with verified diagnostic procedures
- Continuous learning from diagnostic interactions

**Real-time Communication:**

- WebSocket-powered live chat sessions for immediate support
- Interactive diagnostic conversations with AI agents
- Real-time status updates for background processing tasks
- Seamless integration between web and mobile interfaces

**Data Processing & Analytics:**

- Advanced text processing and embedding generation
- Web scraping capabilities for up-to-date automotive information
- Performance monitoring and analytics for system optimization
- Comprehensive logging and error tracking

**Scalable Architecture:**

- Microservices-based design with clear separation of concerns
- Background task processing with Celery for heavy operations
- Redis caching for improved performance and responsiveness
- Supabase integration for reliable data persistence and user management

### Target Use Cases

- **Vehicle Owners:** Get intelligent diagnostic help and maintenance guidance
- **Automotive Professionals:** Access comprehensive diagnostic tools and knowledge base
- **Service Centers:** Streamline diagnostic processes and customer communication
- **Educational Institutions:** Teaching platform for automotive technology
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
- **Rate Limiting**: Built-in rate limiting for API endpoints using SlowAPI.
- **Database Integration**: Uses Supabase for data storage and management.
- **Redis Caching**: Optional Redis cache integration for performance.
- **Extensible Agents & Services**: Modular agents and services for car data, chat, embedding, scraping (Crawl4AI), YouTube, and more.
- **Background Tasks**: Celery-based background task processing with status endpoints.
- **Performance Monitoring**: Service-level performance stats and caching.
- **Logging**: Centralized logging for easier debugging and monitoring.
- **Startup/Shutdown Checks**: Automatic health checks for Milvus, Supabase, Redis, and graceful shutdown of all services.
- **Dependency Injection**: FastAPI dependency injection for logger, DB, and WebSocket manager.
- **SOLID Architecture**: All major components depend on abstractions for testability and maintainability.

## Project Structure

```
main.py                  # FastAPI app entry point
app/
  api/v1/routes.py       # API route definitions
  api/v1/background_tasks.py # Background task endpoints
  api/v1/chat_route.py   # Chat session endpoints
  db/                    # Database handlers and migrations
  CRUD/                  # CRUD operations for entities
  core/                  # Core config, celery, and settings
  schemas/               # Data schemas (Pydantic)
  services/              # Business logic: chat, embedding, scraping, search, etc.
  utils/                 # Utilities (logger, websocket, redis, diagnosis tree, etc.)
car_data/                # Automotive reference documents (PDFs)
tests/                   # Unit and integration tests
requirements.txt         # Python dependencies
```

## Getting Started

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/AbrehamGebremedhin/autocare-backend
   cd autocare-backend
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Configure environment variables:**

   - Set up your Supabase, Redis, Gemini, and YouTube credentials as needed (see `app/core/config.py`).
   - Copy `.env.example` to `.env` and fill in the required values.

4. **Run the application:**
   ```sh
   uvicorn main:app --reload
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
