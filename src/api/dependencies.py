
from database import engine, Session


# Dependency
def get_session():
    with Session(engine) as session:
        yield session
