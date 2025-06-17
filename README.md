# AutoCare Backend

AutoCare Backend is a FastAPI-based backend service designed to power automotive care applications. It provides RESTful APIs and WebSocket support for real-time communication, leveraging a modular architecture for scalability and maintainability.

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

### Prerequisites

- Python 3.10+
- [Supabase](https://supabase.com/) account (for DB)
- Redis (optional, for caching)

### Installation

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
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
