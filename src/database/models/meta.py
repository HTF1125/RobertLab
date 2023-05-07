import sqlalchemy as sa
from .mixins import StaticBase, TimeSeriesBase

class Meta(StaticBase):
    """
    Represents a metadata record in the database.
    """

    __tablename__ = "tb_meta"
    __table_args__ = (sa.UniqueConstraint("code", "name", name="uc_code_name"),)
    meta_id = sa.Column(
        sa.Integer,
        primary_key=True,
        unique=True,
        nullable=False,
        comment="Internal ID",
        doc="Internal ID (UNIVERSAL)",
    )
    code = sa.Column(
        sa.VARCHAR(100),
        unique=True,
        nullable=False,
        index=True,
        comment="Code that could be recognized by Users.",
        doc="Code for user interface",
    )
    meta_type = sa.Column(
        sa.VARCHAR(20),
        nullable=True,
        unique=False,
        comment="Metadata Type",
        doc="Metadata Type",
    )
    meta_instrument = sa.Column(
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
    inception_date = sa.Column(
        sa.Date, nullable=True, comment="Inception Date", doc="Inception Date"
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
    source = sa.Column(
        sa.VARCHAR(255),
        nullable=True,
        comment="source",
        doc="source"
    )
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


    def source_data(self):

        if self.source == "YAHOO":

            import yfinance as yf

            data = yf.download(self.yahoo, start="1990-1-1")

            return data

        return None







class DailyBar(TimeSeriesBase):
    """daily bar series"""

    __tablename__ = "tb_dailybar"
    meta_id = sa.Column(
        sa.ForeignKey("tb_meta.meta_id"),
        primary_key=True,
        comment="Internal ID (FK from tb_meta)",
        doc="Internal ID (UNIVERSAL) (FK from tb_meta)",
    )
    date = sa.Column(sa.Date, primary_key=True, nullable=False)
    open = sa.Column(sa.Numeric(30, 5), nullable=False)
    high = sa.Column(sa.Numeric(30, 5), nullable=False)
    low = sa.Column(sa.Numeric(30, 5), nullable=False)
    close = sa.Column(sa.Numeric(30, 5), nullable=False)
    dvds = sa.Column(sa.Numeric(30, 5), nullable=False)
    volume = sa.Column(sa.Numeric(30, 5), nullable=False)
    gross_rtn = sa.Column(sa.Numeric(30, 5), nullable=False)
