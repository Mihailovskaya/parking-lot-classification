import sqlalchemy as sa
from .base import Base

class PhotoCreationDate(Base):
    __tablename__ = 'photo_creation_date'

    DateTime = sa.Column(sa.DateTime)
    Camera_ID = sa.Column(sa.Integer)
    __table_args__ = (
        sa.PrimaryKeyConstraint(DateTime, Camera_ID),
        {},
    )
    