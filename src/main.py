"""ROBERT"""
import os
from datetime import date
import click

@click.command()
@click.argument("task", default="api")
@click.option("--asofdate", default=str(date.today()), help="As of date")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload.")
def start(task: str, reload: bool, asofdate: str):
    """main cli function"""
    print(task, asofdate)
    if task == "api":
        import uvicorn
        uvicorn.run(app="api.main:app", reload=reload)
    elif task == "streamlit":
        import streamlit.web.bootstrap
        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dirname, "web/stream/main.py")
        streamlit.web.bootstrap.run(filename, "", [], flag_options={})
        # streamlit.bootstrap.run(filename, "", args, flag_option   s={})


if __name__ == "__main__":

    start()
