import sqlalchemy as sa
from .mixins import StaticBase, TimeSeriesBase


class Meta(StaticBase):
    """table meta"""
    __tablename__ = "tbmeta"

    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    ticker = sa.Column(sa.VARCHAR(255))
    name = sa.Column(sa.VARCHAR(255))
    description = sa.Column(sa.VARCHAR(1000), nullable=True)

    __table_args__ = (sa.UniqueConstraint('id'), sa.UniqueConstraint(
        "ticker"), sa.UniqueConstraint("name"))


class Source(StaticBase):
    """ticker mappings"""
    __tablename__ = "tbsource"
    meta_id = sa.Column(sa.ForeignKey("tbmeta.id"))
    