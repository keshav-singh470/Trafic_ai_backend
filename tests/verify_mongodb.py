import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    print("FAILED: MONGO_URI not found in .env")
else:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("SUCCESS: MongoDB connected successfully")
    except Exception as e:
        print(f"FAILED: Connection failed: {e}")
