"""ROBERT"""

from datetime import date
import click
import uvicorn


@click.command()
@click.option("--asofdate", default=str(date.today()), help="As of date")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload.")
def start(reload: bool, asofdate: str):
    """main cli function"""
    print(asofdate)
    uvicorn.run(app="api.main:app", reload=reload)



if __name__ == "__main__":

    start()