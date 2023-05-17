"""ROBERT"""
import click
import uvicorn

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), "../..")))


@click.command()
# @click.option("--asofdate", default=str(date.today()), help="As of date")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload.")
def main(reload: bool):
    """main cli function"""
    uvicorn.run(app="api.main:app", reload=reload)


if __name__ == "__main__":
    main()  # pragma: no cover
