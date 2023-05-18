"""ROBERT"""
import os
from typing import Optional, List
from datetime import datetime, date
from sqlmodel import Session, SQLModel, Field, Relationship, create_engine
from sqlalchemy import Column, DateTime, String, Float, func
from pydantic import BaseModel

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.db")
DATABASE_URL: str = os.getenv("DATABASE_URL") or f"sqlite:///{path}"
class DatePkBase(BaseModel):
    dt: date = Field(default=None, primary_key=True)


class IdPkBase(SQLModel):
    id: Optional[int] = Field(default=None, primary_key=True)


class TimeMixin(BaseModel):
    created_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True),
            default=datetime.utcnow,
            nullable=False,
        )
    )
    modified_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True),
            default=datetime.utcnow,
            onupdate=datetime.utcnow,
            server_default=func.now(),
            nullable=False,
        )
    )


class StaticBase(BaseModel):
    ticker: str = Field(sa_column=Column(String, unique=True, index=True))
    name: Optional[str] = Field(sa_column=Column(String))
    description: Optional[str] = Field(sa_column=Column(String))


class MetaBase(StaticBase):
    """MetaBase"""

    category: Optional[str] = Field(sa_column=Column(String))
    instrument: Optional[str] = Field(sa_column=Column(String))
    isin: Optional[str] = Field(sa_column=Column(String))
    source: Optional[str] = Field(sa_column=Column(String))
    yahoo: Optional[str] = Field(sa_column=Column(String))
    bloomberg: Optional[str] = Field(sa_column=Column(String))
    naver: Optional[str] = Field(sa_column=Column(String))
    morningstar: Optional[str] = Field(sa_column=Column(String))
    reuters: Optional[str] = Field(sa_column=Column(String))


class UniverseBase(StaticBase, TimeMixin):
    pass


class Meta(MetaBase, TimeMixin, IdPkBase, SQLModel, table=True):
    pxlasts: List["PxLast"] = Relationship(back_populates="meta")
    pxdvds: List["PxDvd"] = Relationship(back_populates="meta")

class Universe(UniverseBase, IdPkBase, table=True):
    pass


class PxLast(SQLModel, table=True):
    dt: date = Field(default=None, primary_key=True)
    meta_id: int = Field(foreign_key="meta.id")
    val: float = Field(sa_column=Column(Float))
    meta: Optional["Meta"] = Relationship(back_populates="pxlasts")

class PxDvd(SQLModel, table=True):
    dt: date = Field(default=None, primary_key=True)
    meta_id: int = Field(foreign_key="meta.id")
    val: float = Field(sa_column=Column(Float))
    meta: Optional["Meta"] = Relationship(back_populates="pxdvds")


class UniverseAsset(SQLModel, table=True):
    __tablename__ = "universe_asset"
    dt: date = Field(default=None, primary_key=True)
    universe_id: int = Field(foreign_key="universe.id", primary_key=True)
    meta_id: int = Field(foreign_key="meta.id", primary_key=True)


engine = create_engine(
    url=DATABASE_URL,
    pool_size=10,
    max_overflow=2,
    pool_recycle=300,
    pool_pre_ping=True,
    pool_use_lifo=True,
    echo=False,
)

# SQLModel.metadata.create_all(engine)
