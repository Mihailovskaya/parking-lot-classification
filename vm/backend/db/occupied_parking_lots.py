import sqlalchemy as sa
from .base import Base

class OccupiedParkingLots(Base):
    __tablename__ = 'occupied_parking_lots'

    DateTime = sa.Column(sa.DateTime)
    Place = sa.Column(sa.Integer)
    Camera_ID = sa.Column(sa.Integer)
    Confidence = sa.Column(sa.Float, nullable=True)
    x = sa.Column(sa.Float, nullable=True)
    y = sa.Column(sa.Float, nullable=True)
    w = sa.Column(sa.Float, nullable=True)
    h = sa.Column(sa.Float, nullable=True)
    __table_args__ = (
        sa.PrimaryKeyConstraint(DateTime, Place, Camera_ID),
        {},
    )
    