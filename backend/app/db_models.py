from datetime import datetime, timezone
from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base

def utc_now():
    return datetime.now(timezone.utc)

class IncidentReport(Base):
    __tablename__ = "incident_reports"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # Nullable so existing records remain valid after the additive migration.
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    category: Mapped[str] = mapped_column(String(40), default="landslide")
    status: Mapped[str] = mapped_column(String(30), default="reported", index=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    reporter_name: Mapped[str] = mapped_column(String(100), default="Demo Citizen")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

class SOSRequest(Base):
    __tablename__ = "sos_requests"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # Nullable so existing records remain valid after the additive migration.
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    message: Mapped[str] = mapped_column(Text, default="Emergency assistance requested")
    status: Mapped[str] = mapped_column(String(30), default="open", index=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    contact_name: Mapped[str] = mapped_column(String(100), default="Demo Citizen")
    contact_phone: Mapped[str] = mapped_column(String(30), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

class ProjectAlert(Base):
    __tablename__ = "project_alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    message: Mapped[str] = mapped_column(Text, default="")
    risk_level: Mapped[str] = mapped_column(String(20), default="Moderate")
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
