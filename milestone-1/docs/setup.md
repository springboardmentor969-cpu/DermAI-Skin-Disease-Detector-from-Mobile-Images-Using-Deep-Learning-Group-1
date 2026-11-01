# Setup guide

## Prerequisites

- Python 3.11+
- Node.js 20+
- PostgreSQL 15+ (recommended for production use)
- VS Code on Windows

## PostgreSQL setup

1. Install PostgreSQL and create a database named `dermai`.
2. Create a role or use the default `postgres` user.
3. Configure `DATABASE_URL` in your environment or `.env` file.
4. Generate a random `JWT_SECRET_KEY` of at least 32 characters. Do not commit a real secret.
5. Keep `CORS_ORIGINS` set to the frontend origin(s), normally `http://localhost:5173,http://127.0.0.1:5173`.

Example:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/dermai
```

If Postgres is not available, install it and configure it before running the application. Do not silently fall back to SQLite in the production configuration.

## Python setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set the values in `backend\.env` in the current PowerShell session before starting Uvicorn:

```powershell
$env:DATABASE_URL = "postgresql+psycopg://postgres:<password>@localhost:5432/dermai"
$env:JWT_SECRET_KEY = "<random-secret-at-least-32-characters>"
$env:CORS_ORIGINS = "http://localhost:5173"
python -m uvicorn app.main:app --reload --env-file .env
```

## Frontend setup

```powershell
cd frontend
npm install
npm run dev
```

Copy `frontend\.env.example` to `frontend\.env.local` to configure `VITE_API_BASE_URL`.

## Backend tests and database integration

Run backend tests from `backend` with:

```powershell
python -m pytest -q
```

By default, these API tests use a uniquely named SQLite database in the operating-system temporary directory. This is an explicit test-only override; the application runtime still defaults to PostgreSQL. To run the same API tests against PostgreSQL, set `DERMAI_TEST_DATABASE_URL` to a dedicated, disposable test database URL before pytest. PostgreSQL integration was not verified when no PostgreSQL service is running.

## Dataset report

The read-only dataset endpoint reads the actual inspection report and metadata. If those generated files are missing, run from the project root:

```powershell
python scripts\inspect_dataset.py
```

## Dataset

The real dataset is already present in the project root and is read-only. The application configuration points to the existing `HAM10000_images_part_1` and `HAM10000_images_part_2` directories without copying them.
