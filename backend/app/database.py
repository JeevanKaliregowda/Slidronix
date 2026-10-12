import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / '.env')
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# For a college demo, SQLite works immediately. Set DATABASE_URL to PostgreSQL
# when PostgreSQL is installed, e.g. postgresql+psycopg://postgres:password@localhost:5432/slidronix
_DEFAULT_DB = Path(__file__).resolve().parents[2] / "slidronix_demo.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_DEFAULT_DB.as_posix()}")

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()