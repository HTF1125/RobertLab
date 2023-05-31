import sqlalchemy as sa
from .mixins import Mixins

class User(Mixins):

    __tablename__ = "tb_user"
    id = sa.Column(sa.Interval, primary_key=True, index=True)
    email = sa.Column(sa.String, unique=True, index = True)
    password = sa.Column(sa.String)

