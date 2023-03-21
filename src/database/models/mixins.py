
from datetime import date, datetime
from typing import Union, Dict, List
from sqlalchemy.orm import Query
import sqlalchemy as sa
import numpy as np
import pandas as pd
from ..client import Base, SessionContext


def read_sql_query(query: Query, **kwargs) -> pd.DataFrame:
    """_summary_

    Args:
        query (Query): _description_

    Returns:
        pd.DataFrame: _description_
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
        with SessionContext() as session:
            session.add(cls(**kwargs))
            session.commit()
        
    
    @classmethod
    def insert(cls, records: Union[List[Dict], pd.Series, pd.DataFrame]) -> None:
        """insert bulk"""

        if isinstance(records, pd.DataFrame):
            records = records.replace({np.NaN: None}).to_dict("records")
        elif isinstance(records, pd.Series):
            records = [records.replace({np.NaN: None}).to_dict()]

        with SessionContext() as session:
            session.bulk_insert_mappings(cls, records)
            session.commit()

    @classmethod
    def from_dict(cls, data: Dict):
        """construct cls object from dict"""
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
        with SessionContext() as session:
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
    """timestamp mixins"""
    __abstract__ = True
    created_date = sa.Column(
        sa.DateTime, server_default=sa.func.now(), nullable=False)
    last_modified_date = sa.Column(
        sa.DateTime, server_default=sa.func.now(), nullable=False)


class TimeSeriesBase(Mixins):
    """abstract TimeSeries class"""

    __abstract__ = True