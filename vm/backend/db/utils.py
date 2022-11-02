from .base import engine as engine
from sqlalchemy.orm import Session

def add_element(element):
    with Session(engine) as session:
        session.add(element)
        session.commit()

