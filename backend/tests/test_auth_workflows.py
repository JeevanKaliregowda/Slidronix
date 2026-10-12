"""Integration checks for auth and record access, using a disposable test DB."""

import os
import tempfile
import uuid
from pathlib import Path

_test_db = Path(tempfile.gettempdir()) / f"slidronix_test_{uuid.uuid4().hex}.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_test_db.as_posix()}"
os.environ["JWT_SECRET_KEY"] = "test-only-secret-that-is-not-used-outside-this-process"

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text

from app.auth import Role, User, password_hash
from app.database import SessionLocal
from app.main import app
from app.incidents import ensure_ownership_columns


client = TestClient(app)


def test_additive_migration_preserves_legacy_incident_and_sos_rows():
    legacy_path = Path(tempfile.gettempdir()) / f"slidronix_legacy_{uuid.uuid4().hex}.db"
    legacy_engine = create_engine(f"sqlite:///{legacy_path.as_posix()}")
    with legacy_engine.begin() as connection:
        connection.execute(text("CREATE TABLE incident_reports (id INTEGER PRIMARY KEY, title TEXT NOT NULL)"))
        connection.execute(text("CREATE TABLE sos_requests (id INTEGER PRIMARY KEY, message TEXT NOT NULL)"))
        connection.execute(text("INSERT INTO incident_reports (id, title) VALUES (7, 'Legacy incident')"))
        connection.execute(text("INSERT INTO sos_requests (id, message) VALUES (9, 'Legacy SOS')"))

    ensure_ownership_columns(legacy_engine)
    with legacy_engine.connect() as connection:
        assert "user_id" in {column["name"] for column in inspect(connection).get_columns("incident_reports")}
        assert "user_id" in {column["name"] for column in inspect(connection).get_columns("sos_requests")}
        assert connection.execute(text("SELECT title FROM incident_reports WHERE id=7")).scalar_one() == "Legacy incident"
        assert connection.execute(text("SELECT message FROM sos_requests WHERE id=9")).scalar_one() == "Legacy SOS"
    legacy_engine.dispose()
    legacy_path.unlink(missing_ok=True)


def register(name: str):
    email = f"{uuid.uuid4().hex}@example.com"
    password = "safe-test-password-123"
    response = client.post("/auth/register", json={
        "name": name,
        "email": email,
        "password": password,
        "role": "admin",
    })
    assert response.status_code == 201
    assert response.json()["role"] == Role.citizen.value
    token_response = client.post("/auth/token", data={"username": email, "password": password})
    assert token_response.status_code == 200
    return response.json(), token_response.json()["access_token"]


def test_registration_login_and_me():
    account, token = register("First Citizen")
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/auth/me", headers=headers)
    assert response.status_code == 200
    assert response.json() == account
    summary = client.get("/predictions/summary", headers=headers)
    assert summary.status_code == 200
    assert summary.json()["total_locations"] > 0
    assert client.get("/auth/admin/users", headers=headers).status_code == 403


def test_prediction_api_requires_authentication():
    assert client.get("/predictions/summary").status_code == 401
    assert client.get("/predict/coordinate/status").status_code == 401


def test_citizen_records_are_owned_and_names_come_from_account():
    first, first_token = register("First Citizen")
    second, second_token = register("Second Citizen")
    first_headers = {"Authorization": f"Bearer {first_token}"}
    second_headers = {"Authorization": f"Bearer {second_token}"}

    created_incident = client.post("/incidents", headers=first_headers, json={
        "title": "Observed rockfall",
        "description": "A small fall near the road.",
        "latitude": 13.1,
        "longitude": 75.2,
        "reporter_name": "Spoofed name",
    })
    assert created_incident.status_code == 201
    assert created_incident.json()["reporter_name"] == first["name"]
    assert [x["id"] for x in client.get("/incidents", headers=first_headers).json()] == [created_incident.json()["id"]]
    assert client.get("/incidents", headers=second_headers).json() == []

    created_sos = client.post("/sos", headers=first_headers, json={
        "message": "Please check the road.",
        "latitude": 13.1,
        "longitude": 75.2,
        "contact_name": "Spoofed name",
    })
    assert created_sos.status_code == 201
    assert created_sos.json()["contact_name"] == first["name"]
    assert [x["id"] for x in client.get("/sos", headers=first_headers).json()] == [created_sos.json()["id"]]
    assert client.get("/sos", headers=second_headers).json() == []
    assert client.patch(f"/sos/{created_sos.json()['id']}/status", headers=first_headers, json={"status": "resolved"}).status_code == 403


def test_admin_can_change_other_account_role_but_not_self_demote():
    target, _ = register("Rescue Operator")
    with SessionLocal() as db:
        admin = User(
            name="Test Administrator",
            email=f"admin-{uuid.uuid4().hex}@example.com",
            password_hash=password_hash.hash("admin-test-password-123"),
            role=Role.admin.value,
        )
        db.add(admin)
        db.commit()
        db.refresh(admin)
        admin_id = admin.id
        from app.auth import token_for
        admin_token = token_for(admin)

    headers = {"Authorization": f"Bearer {admin_token}"}
    changed = client.patch(f"/auth/admin/users/{target['id']}/role", headers=headers, json={"role": "rescue"})
    assert changed.status_code == 200
    assert changed.json()["role"] == Role.rescue.value
    assert client.get("/auth/admin/users", headers=headers).status_code == 200
    assert client.patch(f"/auth/admin/users/{admin_id}/role", headers=headers, json={"role": "citizen"}).status_code == 409


def test_rescue_workspace_can_manage_shared_queue_and_publish_alerts():
    _, citizen_token = register("Reporting Citizen")
    rescue, rescue_token = register("Rescue Operator")
    with SessionLocal() as db:
        rescue_user = db.get(User, rescue["id"])
        rescue_user.role = Role.rescue.value
        db.commit()

    # The role claim comes from the database at token validation time, so a
    # session issued before a role change sees the newly assigned workspace.
    citizen_headers = {"Authorization": f"Bearer {citizen_token}"}
    rescue_headers = {"Authorization": f"Bearer {rescue_token}"}
    incident = client.post("/incidents", headers=citizen_headers, json={
        "title": "Blocked access road",
        "latitude": 13.2,
        "longitude": 75.3,
    })
    assert incident.status_code == 201
    queue = client.get("/incidents", headers=rescue_headers).json()
    assert any(record["id"] == incident.json()["id"] for record in queue)
    updated = client.patch(
        f"/incidents/{incident.json()['id']}/status",
        headers=rescue_headers,
        json={"status": "responding"},
    )
    assert updated.status_code == 200
    alert = client.post("/alerts", headers=rescue_headers, json={
        "title": "Road access update",
        "message": "Response team is reviewing the blocked road.",
        "risk_level": "Critical",
    })
    assert alert.status_code == 201
    alerts = client.get("/alerts", headers=citizen_headers).json()
    assert any(item["id"] == alert.json()["id"] and item["risk_level"] == "Critical" for item in alerts)

