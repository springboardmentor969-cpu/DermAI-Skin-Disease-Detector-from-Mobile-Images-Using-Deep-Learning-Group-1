# DermAI

DermAI is a skin-disease detection project using the HAM10000 dermoscopic dataset already present in this repository. The initial milestone focuses on dataset inspection, architecture documentation, database design, and a working authentication and case-management skeleton.

## Dataset source

The project uses the existing Kaggle HAM10000 image set downloaded and extracted in this workspace. The real files are located in:

- `HAM10000_images_part_1/`
- `HAM10000_images_part_2/`
- `HAM10000_metadata.csv`

The application reads this dataset without creating a duplicate copy or altering the source files.

The HAM10000 image directories are not included in this upload because of their size. Download the dataset separately and set `DATASET_ROOT` in `backend/.env` to the image directory before running dataset-dependent features.

## Project structure

- `backend/` — FastAPI application, SQLAlchemy models, routes, and tests.
- `frontend/` — React + TypeScript + Vite + Tailwind UI.
- `scripts/` — dataset inspection utilities.
- `reports/` — generated dataset statistics and charts.
- `docs/` — setup and API notes.

## Quick start

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env with your PostgreSQL URL and a random JWT secret (at least 32 characters).
python -m uvicorn app.main:app --reload --env-file .env
```

PostgreSQL is the configured application database. The backend does not automatically fall back to SQLite.

### Frontend

Open a second PowerShell terminal:

```powershell
cd frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally `http://localhost:5173`).

## Dataset inspection

Run the dataset report generator:

```powershell
python scripts\inspect_dataset.py
```

This writes:

- `reports/class_distribution.csv`
- `reports/class_distribution.png`
- `reports/dataset_report.md`

## Architecture notes

See `architecture.md` and `database_schema.md`.
