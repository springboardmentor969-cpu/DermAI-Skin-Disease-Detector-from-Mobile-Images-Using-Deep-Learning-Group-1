# Database schema

## Users

- `id`: primary key
- `email`: unique email address
- `full_name`: display name
- `password_hash`: secure password hash
- `is_active`: account status
- `role`: user or admin

## Cases

- `id`: primary key
- `user_id`: foreign key to `users.id`
- `title`: short case name
- `description`: optional notes
- `status`: lifecycle state such as `new`
- `created_at`: creation timestamp
- `updated_at`: last modification timestamp

## Images

- `id`: primary key
- `case_id`: foreign key to `cases.id`
- `filename`: original uploaded name
- `content_type`: MIME type
- `storage_path`: safe generated path
- `file_size`: bytes
- `width`, `height`, `color_mode`: image metadata
- `image_hash`: optional duplicate detection hash
- `status`: uploaded/processing/processed
- `created_at`: timestamp
- `metadata_json`: optional metadata payload

## Predictions

- `id`: primary key
- `case_id`: foreign key to `cases.id`
- `model_version`: reserved for future model versions
- `prediction_label`: reserved for model result
- `confidence`: reserved confidence score
- `status`: pending or completed
- `created_at`: timestamp

## Relationship notes

- One user may have many cases.
- One case may have many uploaded images.
- One case may have many prediction records.
- All timestamps use UTC and indexes are created on foreign keys and email lookups.
