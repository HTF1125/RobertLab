import sqlalchemy as sa
from .mixins import StaticBase


class Meta(StaticBase):
    """table meta"""

    __tablename__ = "tbmeta"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    ticker = sa.Column(sa.VARCHAR(255))
    name = sa.Column(sa.VARCHAR(255))
    description = sa.Column(sa.VARCHAR(1000), nullable=True)

    __table_args__ = (
        sa.UniqueConstraint("id"),
        sa.UniqueConstraint("ticker"),
        sa.UniqueConstraint("name"),
    )


class Source(StaticBase):
    """ticker mappings"""

    __tablename__ = "tbsource"
    meta_id = sa.Column(sa.ForeignKey("tbmeta.id"), primary_key=True)
    source = sa.Column(sa.VARCHAR(255), nullable=False, default="NOTSET")
    bloomberg = sa.Column(sa.VARCHAR(255), nullable=True, default=None)
    yahoo = sa.Column(sa.VARCHAR(255), nullable=True, default=None)
    naver = sa.Column(sa.VARCHAR(255), nullable=True, default=None)
    reuter = sa.Column(sa.VARCHAR(255), nullable=True, default=None)
    morningstar = sa.Column(sa.VARCHAR(255), nullable=True, default=None)


class Strategy(StaticBase):
    """strategy"""

    __tablename__ = "tbstrategy"
    meta_id = sa.Column(sa.ForeignKey("tbmeta.id"), primary_key=True)
    name = sa.Column(sa.VARCHAR(255), nullable=False)
    frequency = sa.Column(sa.VARCHAR(255), nullable=False, default="NOTSET")


class Equity(StaticBase):
    """strategy"""

    __tablename__ = "tbequity"
    meta_id = sa.Column(sa.ForeignKey("tbmeta.id"), primary_key=True)


class Index(StaticBase):
    """index"""

    __tablename__ = "tbindex"
    meta_id = sa.Column(sa.ForeignKey("tbmeta.id"), primary_key=True)


class Universe(StaticBase):
    """universe"""

    __tablename__ = "tbuniverse"
    strategy_id = sa.Column(sa.ForeignKey("tbstrategy.meta_id"), primary_key=True)
    meta_id = sa.Column(sa.ForeignKey("tbequity.meta_id"), primary_key=True)

