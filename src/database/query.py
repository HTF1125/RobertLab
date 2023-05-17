import pandas as pd
from .base import Session, engine, Meta


def get_meta(**kwargs):
    with Session(engine) as session:
        query = session.query(Meta)
        return pd.read_sql_query(sql=query.statement, con=engine, **kwargs)


# def read_sql_query(query, **kwargs) -> pd.DataFrame:
#     """This is pass through function to read query into dataframe."""
#     return pd.read_sql_query(sql=query.statement, con=query.session.bind, **kwargs)


# def universe(name: str) -> pd.DataFrame:
#     with Session(engine) as session:
#         query = (
#             session.query(Universe.date, Meta.ticker, Meta.name)
#             .select_from(Universe)
#             .join(Meta, Universe.asset == Meta.id)
#             .filter()
#         )

#     with SessionContext() as session:
#         query = (
#             session.query(
#                 UniverseMeta.date,
#                 Meta.code,
#                 Meta.name,
#             )
#             .select_from(UniverseMeta)
#             .join(Meta, UniverseMeta.meta_id == Meta.id)
#             .join(Universe, UniverseMeta.universe_id == Universe.id)
#             .filter(Universe.name == name)
#         )

#         return read_sql_query(query=query)
