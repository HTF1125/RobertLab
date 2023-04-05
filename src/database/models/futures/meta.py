import enum
import sqlalchemy as sa
from ..mixins import StaticBase
from ...client import SessionContext


class MetaCategory(enum.Enum):
    """metadata categories"""

    MACRO = "MACRO"
    ETF = "ETF"
    STOCK = "STOCK"
    INDEX = "INDEX"
    NOTSET = "NOTSET"


class Meta(StaticBase):
    """
    Represents a metadata record in the database.
    """

    __tablename__ = "tb_meta"
    __table_args__ = sa.UniqueConstraint("code", "name", name="uc_code_name")

    meta_id = sa.Column(
        sa.Integer,
        sa.Identity(start=100_000),
        primary_key=True,
        comment="Internal ID",
        docs="Internal ID (UNIVERSAL)",
    )
    code = sa.Column(
        sa.VARCHAR(100),
        unique=True,
        nullable=False,
        index=True,
        comment="Code that could be recognized by Users.",
        docs="Code for user interface",
    )
    categ = sa.Column(
        sa.Enum(MetaCategory),
        nullable=False,
        unique=False,
        comment="Category of the Metadata",
        docs="Category of the Metadata",
        default=MetaCategory.NOTSET,
    )
    name = sa.Column(
        sa.VARCHAR(255),
        nullable=False,
        unique=True,
        comment="Abbreviated Name",
        docs="Short Name",
    )
    description = sa.Column(
        sa.VARCHAR(1000),
        unique=False,
        nullable=True,
        comment="Description for the Meta",
        docs="Description for the Meta",
    )
    deactive = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        server_default=False,
    )

    def __str__(self) -> str:
        return f"<Meta id={self.meta_id} code={self.code} name={self.name}>"
