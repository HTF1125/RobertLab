import sqlalchemy as sa
from .mixins import StaticBase, TimeSeriesBase

class Meta(StaticBase):
    """table meta"""

    __tablename__ = "tb_meta"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    ticker = sa.Column(sa.VARCHAR(255), nullable=False)
    name = sa.Column(sa.VARCHAR(255), nullable=False)
    description = sa.Column(sa.VARCHAR(1000), nullable=True)
    deactive = sa.Column(sa.Boolean, default=False, nullable=False)


class Source(StaticBase):
    """ticker mappings"""

    __tablename__ = "tb_source"
    meta_id = sa.Column(sa.ForeignKey("tb_meta.id"), primary_key=True)
    source = sa.Column(sa.VARCHAR(255), nullable=False, default="NOTSET")
    bloomberg = sa.Column(sa.VARCHAR(255), nullable=True, default=None)
    yahoo = sa.Column(sa.VARCHAR(255), nullable=True, default=None)
    naver = sa.Column(sa.VARCHAR(255), nullable=True, default=None)
    reuter = sa.Column(sa.VARCHAR(255), nullable=True, default=None)
    morningstar = sa.Column(sa.VARCHAR(255), nullable=True, default=None)


class Strategy(StaticBase):
    """strategy"""

    __tablename__ = "tb_strategy"
    meta_id = sa.Column(sa.ForeignKey("tb_meta.id"), primary_key=True)
    name = sa.Column(sa.VARCHAR(255), nullable=False)
    frequency = sa.Column(sa.VARCHAR(255), nullable=False, default="NOTSET")


class Equity(StaticBase):
    """strategy"""

    __tablename__ = "tb_equity"
    meta_id = sa.Column(sa.ForeignKey("tb_meta.id"), primary_key=True)


class Index(StaticBase):
    """index"""

    __tablename__ = "tb_index"
    meta_id = sa.Column(sa.ForeignKey("tb_meta.id"), primary_key=True)


class Universe(StaticBase):
    """universe"""

    __tablename__ = "tb_universe"
    strategy_id = sa.Column(sa.ForeignKey("tb_strategy.meta_id"), primary_key=True)
    meta_id = sa.Column(sa.ForeignKey("tb_equity.meta_id"), primary_key=True)


class EquityDailyBar(TimeSeriesBase):
    """equity daily bar"""
    __tablename__ = "tb_equity_daly_bar"
    meta_id = sa.Column(
        sa.ForeignKey("tb_equity.meta_id"), primary_key=True, index=True
    )
    date = sa.Column(sa.Date, primary_key=True, nullable=False, index=True)
    open = sa.Column(sa.Numeric(20, 5), nullable=True)
    high = sa.Column(sa.Numeric(20, 5), nullable=True)
    low = sa.Column(sa.Numeric(20, 5), nullable=True)
    close = sa.Column(sa.Numeric(20, 5), nullable=True)
    volume = sa.Column(sa.Numeric(20, 5), nullable=True)
    dvds = sa.Column(sa.Numeric(20, 5), nullable=True)
    splits = sa.Column(sa.Numeric(20, 5), nullable=True)
    tot_return = sa.Column(sa.Numeric(20, 10), nullable=True)
