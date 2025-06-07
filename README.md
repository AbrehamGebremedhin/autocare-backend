# AutoCare Backend

AutoCare Backend is a FastAPI-based backend service designed to power automotive care applications. It provides RESTful APIs and WebSocket support for real-time communication, leveraging a modular architecture for scalability and maintainability.

## Features

- **REST API**: Organized under `/api/v1` for versioning and modularity.
- **WebSocket Support**: Real-time communication endpoint at `/ws`.
- **Unified Error Handling**: Consistent error responses for HTTP and generic exceptions.
- **Database Integration**: Uses Supabase for data storage and management.
- **Redis Caching**: Optional Redis cache integration for performance.
- **Extensible Agents & Services**: Modular agents and services for car data, chat, embedding, scraping, and more.
- **Logging**: Centralized logging for easier debugging and monitoring.

## Project Structure

```
main.py                  # FastAPI app entry point
app/
  api/v1/routes.py       # API route definitions
  db/                    # Database handlers and migrations
  models/                # Pydantic models
  schemas/               # Data schemas
  services/              # Business logic and integrations
  utils/                 # Utilities (logger, websocket, redis, etc.)
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

   - Set up your Supabase and Redis credentials as needed (see `app/core/config.py`).

4. **Run the application:**
   ```sh
   uvicorn main:app --reload
   ```

### API Endpoints

- **Root:** `GET /` — Welcome message
- **API v1:** `/api/v1/...` — Main REST endpoints
- **WebSocket:** `ws://<host>/ws` — Real-time communication

### Testing

Run tests with:

```sh
pytest
```
