# DermAI architecture

## High-level flow

React frontend -> FastAPI backend -> authentication and case-management services -> image validation and preprocessing -> PostgreSQL database

## Components

### Frontend

- React + TypeScript + Vite + Tailwind CSS for a responsive medical research dashboard.
- Views include authentication, dashboard, case management, image upload, and dataset overview pages.
- Frontend communicates with the FastAPI API over JSON.

### Backend

- FastAPI application handles user auth, case CRUD, upload validation, and future model endpoints.
- SQLAlchemy models define the relational schema for users, cases, uploaded images, and prediction records.
- Dataset inspection and preprocessing remain separate from future model-training logic.

### Storage and data flow

- The original Kaggle dataset remains read-only and is never duplicated.
- Uploaded user images are stored in a separate application upload directory.
- PostgreSQL is the primary database system. Local tests can use an explicit SQLite override only for isolated test execution.

## Milestone boundaries

This milestone covers:

- dataset inspection and reporting
- project scaffolding
- system design documentation
- database schema
- authentication and case APIs
- upload + preprocessing pipeline

Model training and inference are intentionally deferred to future milestones.
