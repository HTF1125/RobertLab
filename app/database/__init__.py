from .client import SessionContext, engine, Base
from . import models


def create_all():
    """ "create all database tables"""
    Base.metadata.create_all(engine)


def drop_all():
    """drop all database tables"""
    Base.metadata.drop_all(engine)
