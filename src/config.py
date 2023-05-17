"""configuration"""
import os
from dotenv import load_dotenv

path = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(path) != "src":
    path = os.path.abspath(os.path.join(path, "../"))
    break
TOP_FOLDER = os.path.abspath(os.path.join(path, "../"))
SRC_FOLDER: str = os.path.join(TOP_FOLDER, "src")
API_FOLDER: str = os.path.join(SRC_FOLDER, "api")
ENV_FILE: str = os.path.join(TOP_FOLDER, ".env")
# Load environment variables from .env file
load_dotenv(dotenv_path=ENV_FILE)
