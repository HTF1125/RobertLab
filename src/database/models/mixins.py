import logging
from typing import Union, Dict, List
from datetime import date, datetime
from dateutil import parser
from sqlalchemy.orm import Query
import numpy as np
import pandas as pd
import sqlalchemy as sa
from ..client import SessionContext, Base


logger = logging.getLogger(__name__)


def read_sql_query(query: Query, **kwargs) -> pd.DataFrame:
    """Read sql query

    Args:
        query (Query): sqlalchemy.Query

    Returns:
        pd.DataFrame: read the query into dataframe.
    """
    return pd.read_sql_query(
        sql=query.statement,
        con=query.session.bind,
        index_col=kwargs.get("index_col", None),
        parse_dates=kwargs.get("parse_dates", None),
    )


class Mixins(Base):
    """mixins for models"""

    __abstract__ = True

    @staticmethod
    def parse_datetime(table: sa.Table, records: List[Dict]) -> List[Dict]:
        out = []

        mapper = sa.inspect(table).columns

        for record in records:
            parsed = record.copy()
            for column in mapper:
                if isinstance(getattr(table, column.key), date):
                    parsed[column.key] = parser.parse(str(record[column.key])).date()
                elif isinstance(getattr(table, column.key), datetime):
                    parsed[column.key] = parser.parse(str(record[column.key]))

                out.append(parsed)

        return out

    @classmethod
    def add(cls, **kwargs) -> None:
        """add an object"""
        session = kwargs.pop("session", None)
        if session is None:
            with SessionContext() as session:
                session.add(cls(**kwargs))
                session.commit()
                return
        session.add(cls(**kwargs))

    @classmethod
    def insert(
        cls, records: Union[List[Dict], pd.Series, pd.DataFrame], **kwargs
    ) -> None:
        """insert bulk"""

        if isinstance(records, pd.DataFrame):
            records = records.replace({np.NaN: None}).to_dict("records")
        elif isinstance(records, pd.Series):
            records = [records.replace({np.NaN: None}).to_dict()]
        elif isinstance(records, list):
            ...
        elif isinstance(records, dict):
            records = [records]
        else:
            raise TypeError(
                "insert only takes pd.Series or pd.DataFrame,"
                + " but {type(records)} was given."
            )

        session = kwargs.pop("session", None)
        if session is None:
            with SessionContext() as session:
                session.bulk_insert_mappings(cls, records)
                session.commit()
                return
        session.bulk_insert_mappings(cls, records)
        print(f"insert into {cls.__tablename__}: {len(records)} records complete.")

    @classmethod
    def update(
        cls, records: Union[Dict, List[Dict], pd.Series, pd.DataFrame], **kwargs
    ) -> None:
        """insert bulk"""

        if isinstance(records, pd.DataFrame):
            records = records.replace({np.NaN: None}).to_dict("records")
        elif isinstance(records, pd.Series):
            records = [records.replace({np.NaN: None}).to_dict()]
        else:
            raise TypeError(
                "insert only takes pd.Series or pd.DataFrame,"
                + " but {type(records)} was given."
            )

        session = kwargs.pop("session", None)
        if session is None:
            with SessionContext() as session:
                session.bulk_update_mappings(cls, records)
                session.commit()
                return
        session.bulk_update_mappings(cls, records)

    @classmethod
    def from_dict(cls, data: Dict):
        """instance construct from dict"""
        return cls(**data)

    def to_dict(self) -> Dict:
        """Convert database table row to dictionary."""
        return {
            column.key: getattr(self, column.key).isoformat()
            if isinstance(getattr(self, column.key), (date, datetime))
            else getattr(self, column.key)
            for column in sa.inspect(self.__class__).columns
        }

    @classmethod
    def query(cls, **kwargs) -> Query:
        """make a query"""
        session = kwargs.pop("session", None)
        if session is None:
            with SessionContext() as session:
                return session.query(cls).filter_by(**kwargs)
        return session.query(cls).filter_by(**kwargs)

    @classmethod
    def query_df(cls, **kwargs) -> pd.DataFrame:
        """query table with dataframe"""
        read_kwargs = {
            "index_col": kwargs.pop("index_col", None),
            "parse_dates": kwargs.pop("parse_dates", None),
        }
        return read_sql_query(cls.query(**kwargs), **read_kwargs)

    @classmethod
    def delete(cls, **kwargs) -> None:
        """delete recrods"""
        with SessionContext() as session:
            session.query(cls).filter_by(**kwargs).delete()
            session.commit()


class StaticBase(Mixins):
    """abstract static mixins"""

    __abstract__ = True
    created_date = sa.Column(
        sa.DateTime,
        server_default=sa.func.now(),
        nullable=False,
        comment="Last Modified Datetime.",
        doc="Last Modified Datetime.",
    )
    last_modified_date = sa.Column(
        sa.DateTime,
        server_default=sa.func.now(),
        server_onupdate=sa.func.now(),
        nullable=False,
        comment="Last Modified Datetime.",
        doc="Last Modified Datetime.",
    )
    memo = sa.Column(
        sa.VARCHAR(1000),
        unique=False,
        nullable=True,
        comment="Additional Comments.",
        doc="Additional Comments.",
    )


class TimeSeriesBase(Mixins):
    """abstract timeseries mixins"""

    __abstract__ = True
