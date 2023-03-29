import logging
from datetime import date, datetime
from typing import Union, Dict, List
from sqlalchemy.orm import Query
import sqlalchemy as sa
import numpy as np
import pandas as pd
from ..client import Base, SessionContext

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
    def insert(cls, records: Union[List[Dict], pd.Series, pd.DataFrame], **kwargs) -> None:
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
                session.bulk_insert_mappings(cls, records)
                session.commit()
                return
        session.bulk_insert_mappings(cls, records)

    @classmethod
    def update(cls, records: Union[List[Dict], pd.Series, pd.DataFrame], **kwargs) -> None:
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

    def dict(self) -> Dict:
        """converty database table row to dict"""
        return {
            c.key: getattr(self, c.key).isoformat()
            if isinstance(getattr(self, c.key), (date, datetime))
            else getattr(self, c.key)
            for c in sa.inspect(self).mapper.sa.column_attrs
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
    created_date = sa.Column(sa.DateTime, server_default=sa.func.now(), nullable=False)
    last_modified_date = sa.Column(
        sa.DateTime, server_default=sa.func.now(), nullable=False
    )


class TimeSeriesBase(StaticBase):
    """abstract timeseries mixins"""

    __abstract__ = True
    created_date = sa.Column(sa.DateTime, server_default=sa.func.now(), nullable=False)
    last_modified_date = sa.Column(
        sa.DateTime, server_default=sa.func.now(), nullable=False
    )
