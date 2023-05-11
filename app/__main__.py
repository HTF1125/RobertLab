"""ROBERT"""
import os
import sys
import click
import uvicorn
from datetime import date
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../..")))


@click.command()
@click.option("--asofdate", default=str(date.today()), type=str)
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload.")
def main(reload: bool, asofdate: str):
    """main cli function"""
    print(asofdate)
    uvicorn.run(app="app.api.main:app", reload=reload)


if __name__ == "__main__":
    main()  # pragma: no cover  