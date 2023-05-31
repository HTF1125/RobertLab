"""ROBERT"""
import os
import sys
from datetime import date
from dotenv import load_dotenv
import click

"""configuration"""

path = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(path) != "src":
    path = os.path.abspath(os.path.join(path, "../"))
    break
TOP_FOLDER = os.path.abspath(os.path.join(path, "../"))
SRC_FOLDER: str = os.path.join(TOP_FOLDER, "src")
API_FOLDER: str = os.path.join(SRC_FOLDER, "api")
ENV_FILE: str = os.path.join(SRC_FOLDER, ".env")
# Load environment variables from .env file
load_dotenv(dotenv_path=ENV_FILE)


@click.command()
@click.argument("task", default="api")
@click.option("--asofdate", default=str(date.today()), help="As of date")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload.")
def start(task: str, reload: bool, asofdate: str):
    """main cli function"""
    print(task, asofdate)
    if task == "api":
        try:
            import uvicorn
            uvicorn.run(app="api.main:app", reload=reload)
        except ImportError as exc:
            raise ImportError() from exc

    elif task == "web":
        from streamlit.web import cli as stcli

        file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "web/main.py"
        )

        sys.argv = ["streamlit", "run", file, ]
        sys.exit(stcli.main())
