"""ROBERT"""
import os
from typing import List, Dict
from datetime import datetime
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy import Column, String, Integer, Float, Date, DateTime, ForeignKey
from sqlalchemy import func
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


class Mixins(Base):
    __abstract__ = True

    @classmethod
    def insert(cls, records: List[Dict]) -> bool:
        with Session() as session:
            session.bulk_insert_mappings(cls, records)
        return True


class TimeMixins(Mixins):
    __abstract__ = True
    created_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        server_default=func.now(),
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        server_default=func.now(),
    )


class Investable(TimeMixins):
    __tablename__ = "investable"

    ticker = Column(String, index=True)
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    description = Column(String)


class Universe(TimeMixins):
    __tablename__ = "universe"
    ticker = Column(String, index=True)
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    description = Column(String)


class UniverseInvestable(Mixins):
    __tablename__ = "universe_investable"
    date = Column(Date, primary_key=True)
    universe_id = Column(ForeignKey("universe.id"), primary_key=True)
    investable_id = Column(ForeignKey("investable.id"), primary_key=True)


class PxLast(Base):
    __tablename__ = "px_last"
    date = Column(Date, primary_key=True)
    investable_id = Column(ForeignKey("investable.id"), primary_key=True)
    px_last = Column(Float)


class PxVolume(Base):
    __abstract__ = True
    date = Column(Date, primary_key=True)
    investable_id = Column(ForeignKey("investable.id"), primary_key=True)
    px_volume = Column(Integer)

class PxDvd(Base):
    __abstract__ = True
    date = Column(Date, primary_key=True)
    investable_id = Column(ForeignKey("investable.id"), primary_key=True)
    px_dvd = Column(Float)


class PxSplit(Base):
    __abstract__ = True
    date = Column(Date, primary_key=True)
    investable_id = Column(ForeignKey("investable.id"), primary_key=True)
    px_split = Column(Float)


def create_all() -> None:
    Base.metadata.create_all(engine)
