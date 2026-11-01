# API overview

## Authentication

- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`

## Cases

- `POST /api/v1/cases`
- `GET /api/v1/cases`
- `GET /api/v1/cases/{case_id}`

## Uploads

- `POST /api/v1/upload/{case_id}`

## Dataset statistics

- `GET /api/v1/dataset/stats`
- Reads the generated class distribution and existing HAM10000 metadata.
- Returns dataset image/lesion counts and observed diagnosis class counts; it does not return predictions.

## Health

- `GET /health`

## Notes

- Authentication uses JWT bearer tokens.
- Protected routes require `Authorization: Bearer <token>`.
- Uploaded files are validated as real image content before storage.
- Access tokens remain in frontend memory and are not persisted in browser storage. Reloading the page requires signing in again.
