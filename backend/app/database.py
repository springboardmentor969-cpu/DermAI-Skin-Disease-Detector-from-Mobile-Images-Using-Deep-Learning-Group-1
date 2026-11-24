from pymongo import MongoClient

from .config import settings


# ============================================================
# MONGODB CONNECTION
# ============================================================

client = MongoClient(
    settings.MONGODB_URL
)

db = client[
    settings.DATABASE_NAME
]


# ============================================================
# COLLECTIONS
# ============================================================

users_collection = db["users"]

sessions_collection = db["sessions"]

cases_collection = db["cases"]

images_collection = db["images"]

analyses_collection = db["analyses"]

predictions_collection = db["predictions"]

model_versions_collection = db["model_versions"]


# ============================================================
# DATABASE INDEXES
# ============================================================

users_collection.create_index(
    "email",
    unique=True
)


# ============================================================
# CONNECTION TEST
# ============================================================

def test_database_connection():

    try:

        client.admin.command("ping")

        print("MongoDB connection successful.")
        print(f"Database: {settings.DATABASE_NAME}")

        return True

    except Exception as error:

        print("MongoDB connection failed.")
        print(f"Error: {error}")

        return False