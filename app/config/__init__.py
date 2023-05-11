"""configuration"""
import os
from dotenv import load_dotenv

path = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(path) != "app":
    path = os.path.abspath(os.path.join(path, "../"))
    break
PROJECT_FOLDER = os.path.abspath(os.path.join(path, "../"))
SOURCE_FOLDER: str = os.path.join(PROJECT_FOLDER, "app")
API_FOLDER: str = os.path.join(SOURCE_FOLDER, "api")
DB_FOLDER: str = os.path.join(SOURCE_FOLDER, "database")
DOTENV_PATH: str = os.path.join(PROJECT_FOLDER, ".env")
# Load environment variables from .env file
load_dotenv(dotenv_path=DOTENV_PATH)
DATABASE_URL: str = (
    os.getenv("DATABASE_URL")
    or f"sqlite:///{os.path.join(DB_FOLDER, 'database.db')}"
)
DATABASE_EXC: str = os.path.join(DB_FOLDER, "database.xlsx")
LOG_LEVEL: str = os.getenv("LOG_LEVEL") or "INFO"
EDGAR_API: str = os.getenv("EDGAR_API")
