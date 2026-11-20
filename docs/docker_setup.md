# DermAI — Docker Deployment & Containerization Guide

## 1. Overview

DermAI is fully containerized using a multi-container Docker Compose architecture:
1. **`db`**: PostgreSQL 16 (Alpine) with persistent volume storage for relational case records, user credentials, and analysis history.
2. **`backend`**: FastAPI Python 3.11-slim application serving deep learning inference (PyTorch, Torchvision), U-Net segmentation, Grad-CAM explainability, and PDF/CSV/JSON report generation.
3. **`frontend`**: Nginx 1.27 (Alpine) serving pre-compiled React 18 + TypeScript + Vite production assets and reverse-proxying `/api/` calls to the backend container.

---

## 2. Directory & Configuration Structure

```
DermAI/
├── backend/
│   ├── Dockerfile             # Multi-stage Python 3.11-slim build
│   ├── .dockerignore          # Excludes caches, venvs, and local git
│   ├── artifacts/models/      # Persistent PyTorch model weights (*.pth)
│   └── requirements.txt       # Frozen dependencies (including reportlab)
├── frontend/
│   ├── Dockerfile             # Multi-stage Node 20 builder -> Nginx runner
│   ├── nginx.conf             # Gzip compression & API reverse-proxy pass
│   └── .dockerignore          # Excludes node_modules and local build dist
├── docker-compose.yml         # Unified service coordination
└── .env.docker                # Production environment variable templates
```

---

## 3. Prerequisites

- **Docker Engine**: Version 24.0+ or **Docker Desktop** (Windows/macOS/Linux)
- **Docker Compose**: Version 2.20+
- Host RAM: Minimum 4GB (8GB recommended for deep learning and ViT models)
- Free Disk Space: Minimum 6GB (for base images and PyTorch model checkpoints)

---

## 4. Environment Configuration

Copy `.env.docker` to `.env` or set your environment variables:

```bash
# In project root:
cp .env.docker .env
```

Key environment variables in `.env.docker`:
| Variable | Value | Description |
|---|---|---|
| `POSTGRES_USER` | `dermai` | PostgreSQL database user |
| `POSTGRES_PASSWORD` | `dermai_secure_password` | PostgreSQL database password |
| `POSTGRES_DB` | `dermai` | PostgreSQL database name |
| `DATABASE_URL` | `postgresql://dermai:dermai_secure_password@db:5432/dermai` | Backend database connection string |
| `SECRET_KEY` | *(Strong 32+ char secret)* | JWT authentication encryption key |
| `UPLOAD_DIR` | `/app/uploads` | Container volume mount for uploads |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `1440` | JWT bearer token lifetime (24 hours) |

---

## 5. Local Docker Deployment (Step-by-Step)

### Step 1: Validate Compose Syntax
```bash
docker compose config
```
Ensure no syntax warnings or volume resolution errors appear.

### Step 2: Build the Containers
```bash
docker compose build --no-cache
```
This builds:
- The React TypeScript application via `node:20-alpine` and packages it into `nginx:1.27-alpine`.
- The FastAPI backend with system libraries `libgl1`, `libglib2.0-0` (for OpenCV/PIL) and PyTorch CPU wheels.

### Step 3: Launch the Stack
```bash
docker compose up -d
```

### Step 4: Verify Container Health
Check running container states:
```bash
docker compose ps
```
Expected output:
```
NAME               IMAGE            STATUS                   PORTS
dermai_db          postgres:16      Up (healthy)             0.0.0.0:5432->5432/tcp
dermai_backend     dermai-backend   Up (healthy)             0.0.0.0:8000->8000/tcp
dermai_frontend    dermai-frontend  Up                       0.0.0.0:5173->80/tcp
```

### Step 5: Access the Application
- **Frontend Web UI**: [http://localhost:5173](http://localhost:5173)
- **Backend Health Check**: [http://localhost:8000/health](http://localhost:8000/health) or [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 6. Persistent Volumes & Data Safety

Docker Compose provisions persistent named volumes:
1. `postgres_data` (`/var/lib/postgresql/data`): Preserves user accounts, clinical cases, uploaded image metadata, and doctor review notes across container restarts.
2. `backend_uploads` (`/app/uploads`): Stores uploaded dermoscopic JPEG/PNG image files and generated masks.
3. `./backend/artifacts` (Bind mount, read-only): Mounts local PyTorch weights (`mobilenet_v2.pth`, `resnet50.pth`, `vit_b_16.pth`, `efficientnet_b0.pth`, `unet.pth`) without modifying or duplicating them.

---

## 7. Useful Docker Management Commands

### Viewing Live Service Logs
```bash
# Follow all logs
docker compose logs -f

# Follow backend logs only (structured audit logging)
docker compose logs -f backend

# Follow database logs
docker compose logs -f db
```

### Stopping the Services
```bash
# Graceful stop
docker compose stop

# Stop and remove containers (preserves database and upload volumes)
docker compose down

# Stop and wipe ALL data (CAUTION: deletes postgres_data and uploads)
docker compose down -v
```

### Running Ad-Hoc Commands Inside the Backend
```bash
# Execute Pytest inside backend container
docker compose exec backend pytest tests -v

# Run database migration or inspect tables
docker compose exec db psql -U dermai -d dermai -c "\dt"
```

---

## 8. Troubleshooting

### Issue 1: Port 5432 Conflict
- **Cause**: A local PostgreSQL service is already running on port 5432 on the host machine.
- **Fix**: Change host port mapping in `docker-compose.yml` to `"5433:5432"`, or stop the local host PostgreSQL service (`net stop postgresql-x64-18`).

### Issue 2: Backend Reports `database: disconnected`
- **Cause**: Database container was not ready before backend attempted connection.
- **Fix**: The compose file includes `condition: service_healthy` on `db`. Verify db health with `docker inspect --format='{{json .State.Health}}' dermai_db`.

### Issue 3: Missing PyTorch Weights
- **Cause**: `./backend/artifacts/models` or segmentation checkpoints not present.
- **Fix**: Ensure `backend/artifacts/models/selection.json` and `.pth` files exist before building or mounting.
