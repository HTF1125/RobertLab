"""ROBERT"""
import os
from contextlib import contextmanager
from sqlalchemy import Column, String, Integer, Date, ForeignKey
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.declarative import declarative_base

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.db")
DATABASE_URL: str = os.getenv("DATABASE_URL") or f"sqlite:///{path}"

Base = declarative_base()

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        url=DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
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
def Session():
    """contexted session maker"""
    session = SessionMaker()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()



Base = declarative_base()

class Meta(Base):
    __tablename__ = "meta1"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, index=True)
    name = Column(String)
    source = Column(String)
    category = Column(String)


class PxLast(Base):
    __tablename__ = "px_index"
    date = Column(Date, primary_key=True)
    meta_id = Column(ForeignKey("meta1.id"))