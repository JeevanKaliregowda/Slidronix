from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, inspect, select, text
from sqlalchemy.orm import Session
from app.database import Base, engine, get_db
from app.db_models import IncidentReport, SOSRequest, ProjectAlert
from app.auth import require_roles

router = APIRouter(tags=["project-workflows"])

# Creates the small demo schema on startup. Existing tables/data are preserved.
Base.metadata.create_all(bind=engine)

def ensure_ownership_columns(bind=engine):
    """Add nullable account ownership to legacy SQLite/Postgres tables."""
    with bind.begin() as connection:
        inspector = inspect(connection)
        for table in ("incident_reports", "sos_requests"):
            if table in inspector.get_table_names() and "user_id" not in {
                column["name"] for column in inspector.get_columns(table)
            }:
                connection.execute(text(f"ALTER TABLE {table} ADD COLUMN user_id INTEGER"))


# create_all does not alter an existing demo database. Add nullable ownership
# columns in place so old rows and all existing data remain intact.
ensure_ownership_columns()

class IncidentCreate(BaseModel):
    title: str = Field(min_length=3, max_length=160)
    description: str = Field(default="", max_length=5000)
    category: str = Field(default="landslide", max_length=40)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    reporter_name: str = Field(default="Demo Citizen", max_length=100)

class IncidentStatusUpdate(BaseModel):
    status: str = Field(pattern="^(reported|under_review|responding|resolved|closed)$")

class SOSCreate(BaseModel):
    message: str = Field(default="Emergency assistance requested", max_length=2000)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    contact_name: str = Field(default="Demo Citizen", max_length=100)
    contact_phone: str = Field(default="", max_length=30)

class SOSStatusUpdate(BaseModel):
    status: str = Field(pattern="^(open|acknowledged|dispatched|resolved|cancelled)$")

class AlertCreate(BaseModel):
    title: str = Field(min_length=3, max_length=160)
    message: str = Field(default="", max_length=5000)
    risk_level: str = Field(default="Moderate", pattern="^(Low|Moderate|High|Critical)$")
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)

def as_dict(obj):
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}

@router.get("/db/status")
def database_status(db: Session = Depends(get_db)):
    try:
        db.execute(select(1))
        return {"status": "ok", "database": engine.dialect.name, "tables": ["incident_reports", "sos_requests", "project_alerts"]}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")

@router.post("/incidents", status_code=201)
def create_incident(payload: IncidentCreate, db: Session = Depends(get_db), user=Depends(require_roles("citizen", "rescue", "admin"))):
    obj = IncidentReport(**payload.model_dump(exclude={"reporter_name"}), user_id=user.id, reporter_name=user.name)
    db.add(obj); db.commit(); db.refresh(obj)
    return as_dict(obj)

@router.get("/incidents")
def list_incidents(status: str | None = None, limit: int = Query(default=100, ge=1, le=500), db: Session = Depends(get_db), user=Depends(require_roles("citizen", "researcher", "rescue", "admin"))):
    stmt = select(IncidentReport).order_by(desc(IncidentReport.created_at)).limit(limit)
    if user.role == "citizen":
        stmt = stmt.where(IncidentReport.user_id == user.id)
    if status:
        stmt = stmt.where(IncidentReport.status == status)
    return [as_dict(x) for x in db.scalars(stmt).all()]

@router.patch("/incidents/{incident_id}/status")
def update_incident_status(incident_id: int, payload: IncidentStatusUpdate, db: Session = Depends(get_db), user=Depends(require_roles("rescue", "admin"))):
    obj = db.get(IncidentReport, incident_id)
    if not obj: raise HTTPException(status_code=404, detail="Incident not found")
    obj.status = payload.status; db.commit(); db.refresh(obj)
    return as_dict(obj)

@router.post("/sos", status_code=201)
def create_sos(payload: SOSCreate, db: Session = Depends(get_db), user=Depends(require_roles("citizen", "rescue", "admin"))):
    obj = SOSRequest(**payload.model_dump(exclude={"contact_name"}), user_id=user.id, contact_name=user.name)
    db.add(obj); db.commit(); db.refresh(obj)
    return as_dict(obj)

@router.get("/sos")
def list_sos(status: str | None = None, limit: int = Query(default=100, ge=1, le=500), db: Session = Depends(get_db), user=Depends(require_roles("citizen", "rescue", "admin"))):
    stmt = select(SOSRequest).order_by(desc(SOSRequest.created_at)).limit(limit)
    if user.role == "citizen":
        stmt = stmt.where(SOSRequest.user_id == user.id)
    if status:
        stmt = stmt.where(SOSRequest.status == status)
    return [as_dict(x) for x in db.scalars(stmt).all()]

@router.patch("/sos/{sos_id}/status")
def update_sos_status(sos_id: int, payload: SOSStatusUpdate, db: Session = Depends(get_db), user=Depends(require_roles("rescue", "admin"))):
    obj = db.get(SOSRequest, sos_id)
    if not obj: raise HTTPException(status_code=404, detail="SOS request not found")
    obj.status = payload.status; db.commit(); db.refresh(obj)
    return as_dict(obj)

@router.post("/alerts", status_code=201)
def create_alert(payload: AlertCreate, db: Session = Depends(get_db), user=Depends(require_roles("rescue", "admin"))):
    obj = ProjectAlert(**payload.model_dump())
    db.add(obj); db.commit(); db.refresh(obj)
    return as_dict(obj)

@router.get("/alerts")
def list_alerts(status: str | None = None, limit: int = Query(default=100, ge=1, le=500), db: Session = Depends(get_db), user=Depends(require_roles("citizen", "researcher", "rescue", "admin"))):
    stmt = select(ProjectAlert).order_by(desc(ProjectAlert.created_at)).limit(limit)
    if status:
        stmt = stmt.where(ProjectAlert.status == status)
    return [as_dict(x) for x in db.scalars(stmt).all()]
