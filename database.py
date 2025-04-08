from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./ml_data.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class ProductData(Base):
    __tablename__ = "product_data"

    id = Column(Integer, primary_key=True, index=True)
    imageid = Column(String, index=True)
    description = Column(String)
    prdtypecode = Column(Integer)

Base.metadata.create_all(bind=engine)