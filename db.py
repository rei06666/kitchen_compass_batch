from sqlalchemy import create_engine, Column, String, PrimaryKeyConstraint, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

# SQLiteのDB接続
DATABASE_URL = "sqlite:///recipes.db"
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()


class Category(Base):
    __tablename__ = "categories"

    category = Column(String, nullable=False)
    category_id = Column(String, nullable=False)
    parent_category = Column(String, nullable=True)
    category_name = Column(String, nullable=False)
    category_url = Column(String, nullable=False)


    # 複合主キー
    __table_args__ = (
        PrimaryKeyConstraint("category_id", "parent_category", "category_name"),
    )


class Recipe(Base):
    __tablename__ = "recipes"

    recipe_id = Column(String, primary_key=True)
    recipe_name = Column(String, nullable=False)
    recipe_url = Column(String, nullable=False)
    recipe_photo = Column(String, nullable=True)
    material = Column(String, nullable=True)
    description = Column(String, nullable=True)
    preprocessed_description = Column(String, nullable=True)
    vector = Column(LargeBinary, nullable=True)



# テーブル作成
Base.metadata.create_all(engine)