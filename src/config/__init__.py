"""configuration"""
import os
from dotenv import load_dotenv

path = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(path) != "src":
    path = os.path.abspath(os.path.join(path, "../"))
    break
PROJECT_FOLDER = os.path.abspath(os.path.join(path, "../"))
APISRC_FOLDER: str = os.path.join(PROJECT_FOLDER, "src/api")
DOTENV_PATH: str = os.path.join(PROJECT_FOLDER, ".env")
# Load environment variables from .env file
load_dotenv(dotenv_path=DOTENV_PATH)
DATABASE_URL:str = os.getenv("DATABASE_URL") or ""
