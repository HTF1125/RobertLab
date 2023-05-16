import pandas as pd
from sqlalchemy.orm import Query
from .client import SessionContext
from .models import Meta, Universe, UniverseMeta


def read_sql_query(query: Query, **kwargs) -> pd.DataFrame:
    """This is pass through function to read query into dataframe."""
    return pd.read_sql_query(sql=query.statement, con=query.session.bind, **kwargs)


def universe(name: str) -> pd.DataFrame:

    with SessionContext() as session:

        query = (session.query(
            UniverseMeta.date,
            Meta.code,
            Meta.name,
        ).select_from(UniverseMeta)
        .join(Meta, UniverseMeta.meta_id == Meta.id).join(
            Universe, UniverseMeta.universe_id == Universe.id
        ).filter(Universe.name == name))

        return read_sql_query(query=query)