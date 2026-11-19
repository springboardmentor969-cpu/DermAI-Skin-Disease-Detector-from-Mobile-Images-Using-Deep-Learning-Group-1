"""
S3-compatible Object Storage service using boto3 for MinIO.
Handles bucket verification/creation and image uploads.
"""

from typing import Optional
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from app.config import settings


class StorageService:
    def __init__(self):
        protocol = "https" if settings.MINIO_USE_SSL else "http"
        endpoint = f"{protocol}://{settings.MINIO_ENDPOINT}"
        self.bucket_name = settings.MINIO_BUCKET_NAME

        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=settings.MINIO_ROOT_USER,
            aws_secret_access_key=settings.MINIO_ROOT_PASSWORD,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )

    def ensure_bucket_exists(self, bucket_name: Optional[str] = None) -> str:
        """
        Verify if the target MinIO bucket exists, creating it automatically if missing.
        """
        target_bucket = bucket_name or self.bucket_name
        try:
            self.s3_client.head_bucket(Bucket=target_bucket)
        except ClientError as e:
            error_code = str(e.response.get("Error", {}).get("Code", ""))
            # 404 or NoSuchBucket means bucket doesn't exist yet
            if error_code in ("404", "NoSuchBucket"):
                self.s3_client.create_bucket(Bucket=target_bucket)
            else:
                # Some S3 implementations return 400 or other errors on head_bucket if missing
                try:
                    self.s3_client.create_bucket(Bucket=target_bucket)
                except Exception:
                    raise e
        return target_bucket

    def upload_file(
        self,
        file_bytes: bytes,
        storage_key: str,
        content_type: str,
        bucket_name: Optional[str] = None,
    ) -> str:
        """
        Upload binary data to MinIO under the specified storage key.
        Returns the storage key on success.
        """
        target_bucket = self.ensure_bucket_exists(bucket_name)
        self.s3_client.put_object(
            Bucket=target_bucket,
            Key=storage_key,
            Body=file_bytes,
            ContentType=content_type,
        )
        return storage_key

    def get_file(self, storage_key: str, bucket_name: Optional[str] = None) -> bytes:
        """
        Retrieve binary data from MinIO by storage key.
        """
        target_bucket = bucket_name or self.bucket_name
        response = self.s3_client.get_object(Bucket=target_bucket, Key=storage_key)
        return response["Body"].read()

    def get_presigned_url(
        self,
        storage_key: str,
        expires_in: int = 3600,
        bucket_name: Optional[str] = None,
    ) -> str:
        """
        Generate a presigned GET URL for an image object in MinIO.
        """
        target_bucket = bucket_name or self.bucket_name
        try:
            return self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": target_bucket, "Key": storage_key},
                ExpiresIn=expires_in,
            )
        except Exception:
            return f"/{storage_key}"


# Global singleton instance
storage_service = StorageService()
