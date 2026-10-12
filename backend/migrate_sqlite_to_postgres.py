import os, sqlite3
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from app.database import Base
from app.db_models import IncidentReport, SOSRequest, ProjectAlert
url = os.getenv("DATABASE_URL")
if not url or not url.startswith("postgresql"):
    raise SystemExit("DATABASE_URL must point to PostgreSQL in backend/.env")
engine = create_engine(url, pool_pre_ping=True)
Base.metadata.create_all(engine)
source = Path(__file__).resolve().parents[1] / "slidronix_demo.db"
if not source.exists():
    raise SystemExit(f"SQLite source not found: {source}")
sqlite = sqlite3.connect(source)
sqlite.row_factory = sqlite3.Row
mapping = [
    ("incident_reports", IncidentReport),
    ("sos_requests", SOSRequest),
    ("project_alerts", ProjectAlert),
]
with Session(engine) as session:
    for table, model in mapping:
        rows = sqlite.execute(f'SELECT * FROM "{table}"').fetchall()
        cols = {c.name for c in model.__table__.columns}
        existing = set(session.scalars(select(model.id)).all())
        copied = 0
        for row in rows:
            data = {k: row[k] for k in row.keys() if k in cols}
            # SQLite stores timestamps as text; normalize them for PostgreSQL DateTime columns.
            if isinstance(data.get("created_at"), str):
                from datetime import datetime
                raw = data["created_at"].strip()
                try:
                    data["created_at"] = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                except ValueError:
                    data["created_at"] = None
            if data.get("id") in existing:
                continue
            session.add(model(**data)); copied += 1
        session.flush()
        print(f"{table}: source={len(rows)}, already_present={len(existing)}, inserted={copied}")
    session.commit()
sqlite.close()
with engine.connect() as conn:
    print("PostgreSQL tables:", ", ".join(Base.metadata.tables.keys()))
print("Migration finished. SQLite source was not modified.")

