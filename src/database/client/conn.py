from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from src.config import DATABASE_URL
Base = declarative_base()

print("*"*100)
print(DATABASE_URL)
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(url=DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(
        url=DATABASE_URL,
        pool_size=10,
        max_overflow=2,
        pool_recycle=300,
        pool_pre_ping=True,
        pool_use_lifo=True,
        echo=False,
    )
SessionMaker = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def SessionContext():
    """contexted session maker"""
    session = SessionMaker()
    try:
        yield session
    except:
        session.rollback()
        raise
    finally:
        session.close()

def create_all():
    """"create all database tables"""
    Base.metadata.create_all(engine)


def drop_all():
    """drop all database tables"""
    Base.metadata.drop_all(engine)