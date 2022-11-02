from sqlalchemy import MetaData, create_engine
from sqlalchemy.ext.declarative import as_declarative



engine = create_engine('postgresql://postgres:111@10.216.0.131:5432/parking_lot_classification')#, echo = True)
metadata = MetaData(bind=engine)

@as_declarative(metadata=metadata)
class Base:
    pass