"""ROBERT"""
import sqlalchemy as sa
from .mixins import StaticBase, TimeSeriesBase


class Meta(StaticBase):
    """
    Represents a metadata record in the database.
    """

    __tablename__ = "tb_meta"
    __table_args__ = (sa.UniqueConstraint("code", "name", name="uc_code_name"),)
    id = sa.Column(
        sa.Integer,
        primary_key=True,
        comment="Internal MetaID",
        doc="Internal Meta ID (UNIVERSAL)",
    )
    code = sa.Column(
        sa.VARCHAR(100),
        unique=True,
        nullable=False,
        index=True,
        comment="Code that could be recognized by Users.",
        doc="Code for user interface",
    )
    category = sa.Column(
        sa.VARCHAR(20),
        nullable=True,
        unique=False,
        comment="Metadata Type",
        doc="Metadata Type",
    )
    instrument = sa.Column(
        sa.VARCHAR(20),
        nullable=True,
        unique=False,
        comment="Metadata Instrument",
        doc="Metadata Instrument",
    )
    isin = sa.Column(
        sa.CHAR(12),
        nullable=True,
        comment="International Securities Identification Code (12)",
        doc="International Securities Identification Code (12)",
    )
    name = sa.Column(
        sa.VARCHAR(255),
        nullable=False,
        unique=True,
        comment="Abbreviated Name",
        doc="Short Name",
    )
    description = sa.Column(
        sa.VARCHAR(1000),
        unique=False,
        nullable=True,
        comment="Description for the Meta",
        doc="Description for the Meta",
    )
    deactive = sa.Column(
        sa.Boolean,
        nullable=False,
        # Use a SQLAlchemy text clause for default value
        default=sa.text("false"),
        server_default=sa.text("false"),
    )
    source = sa.Column(sa.VARCHAR(255), nullable=True, comment="source", doc="source")
    yahoo = sa.Column(
        sa.VARCHAR(100),
        nullable=True,
        comment="Code that could be recognized by Yahoo.",
        doc="Code for Yahoo",
    )
    naver = sa.Column(
        sa.VARCHAR(100),
        nullable=True,
        comment="Code that could be recognized by Naver.",
        doc="Code for Naver.",
    )
    bloomberg = sa.Column(
        sa.VARCHAR(100),
        nullable=True,
        comment="Code that could be recognized by Naver.",
        doc="Code for Naver.",
    )
    morningstar = sa.Column(
        sa.VARCHAR(100),
        nullable=True,
        comment="Code that could be recognized by Morningstar.",
        doc="Code for Morningstar.",
    )
    reuters = sa.Column(
        sa.VARCHAR(100),
        nullable=True,
        comment="Code that could be recognized by Reuters.",
        doc="Code for Reuters.",
    )


class DailyBar(TimeSeriesBase):
    """daily bar series"""

    __tablename__ = "tb_dailybar"
    meta_id = sa.Column(sa.ForeignKey("tb_meta.id"), primary_key=True)
    date = sa.Column(sa.Date, primary_key=True)
    open = sa.Column(sa.Numeric(30, 5), nullable=True)
    high = sa.Column(sa.Numeric(30, 5), nullable=True)
    low = sa.Column(sa.Numeric(30, 5), nullable=True)
    close = sa.Column(sa.Numeric(30, 5), nullable=True)
    volume = sa.Column(sa.Numeric(30, 5), nullable=True)
    dividends = sa.Column(sa.Numeric(30, 5), nullable=True)
    stock_splits = sa.Column(sa.Numeric(30, 5), nullable=True)
    pri_return = sa.Column(sa.Numeric(30, 5), nullable=True)
    dvd_return = sa.Column(sa.Numeric(30, 5), nullable=True)
    tot_return = sa.Column(sa.Numeric(30, 5), nullable=True)



class Universe(StaticBase):
    """InvestmentUniverse"""
    __tablename__ = "tb_universe"
    id = sa.Column(
        sa.Integer,
        primary_key=True,
        comment="Internal Universe ID",
        doc="Internal Universe ID (UNIVERSAL)",
    )
    name = sa.Column(sa.String(255), nullable=False)
class UniverseMeta(TimeSeriesBase):
    """investment universe"""
    __tablename__ = "tb_universe_meta"
    date = sa.Column(sa.Date, primary_key=True)
    universe_id = sa.Column(sa.ForeignKey("tb_universe.id"), primary_key=True)
    meta_id = sa.Column(sa.ForeignKey("tb_meta.id"), primary_key=True)
