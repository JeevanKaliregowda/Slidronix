import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from pwdlib import PasswordHash
import jwt
from sqlalchemy import Boolean, DateTime, Integer, String, select
from sqlalchemy.orm import Mapped, mapped_column, Session
from app.database import Base, engine, get_db

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
SECRET = os.getenv("JWT_SECRET_KEY")
if not SECRET:
    raise RuntimeError("JWT_SECRET_KEY missing in backend/.env")
ALGORITHM = "HS256"
EXPIRE = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
password_hash = PasswordHash.recommended()
oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/token")
router = APIRouter(prefix="/auth", tags=["authentication"])

class Role(str, Enum):
    citizen = "citizen"
    researcher = "researcher"
    rescue = "rescue"
    admin = "admin"

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(500), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="citizen")
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class RegisterIn(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)

class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    email: str
    role: str

class RoleUpdate(BaseModel):
    role: Role

def token_for(user):
    now = datetime.now(timezone.utc)
    return jwt.encode({"sub": str(user.id), "role": user.role, "iat": now, "exp": now + timedelta(minutes=EXPIRE)}, SECRET, algorithm=ALGORITHM)

def current_user(token: str = Depends(oauth2), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
        user_id = int(payload["sub"])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token", headers={"WWW-Authenticate":"Bearer"})
    user = db.get(User, user_id)
    if not user or not user.active:
        raise HTTPException(status_code=401, detail="Account unavailable")
    return user

def require_roles(*roles):
    def guard(user=Depends(current_user)):
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="Insufficient role permissions")
        return user
    return guard

@router.post("/register", response_model=UserOut, status_code=201)
def register(payload: RegisterIn, db: Session = Depends(get_db)):
    # Public registration always creates a citizen; privilege escalation is not accepted.
    email = str(payload.email).lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(name=payload.name.strip(), email=email, password_hash=password_hash.hash(payload.password), role=Role.citizen.value)
    db.add(user); db.commit(); db.refresh(user)
    return UserOut(id=user.id, name=user.name, email=user.email, role=user.role)

@router.post("/token")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == form.username.lower()))
    if not user or not user.active or not password_hash.verify(form.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password", headers={"WWW-Authenticate":"Bearer"})
    return {"access_token": token_for(user), "token_type":"bearer", "role":user.role, "name":user.name}

@router.get("/me", response_model=UserOut)
def me(user=Depends(current_user)):
    return UserOut(id=user.id, name=user.name, email=user.email, role=user.role)

@router.get("/admin/roles")
def roles_admin(user=Depends(require_roles("admin"))):
    return {"roles":[r.value for r in Role]}

@router.get("/admin/users", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db), user=Depends(require_roles("admin"))):
    return db.scalars(select(User).order_by(User.id)).all()

@router.patch("/admin/users/{user_id}/role", response_model=UserOut)
def update_user_role(user_id: int, payload: RoleUpdate, db: Session = Depends(get_db), user=Depends(require_roles("admin"))):
    target = db.get(User, user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if target.id == user.id and payload.role.value != Role.admin.value:
        raise HTTPException(status_code=409, detail="You cannot remove your own administrator role")
    target.role = payload.role.value
    db.commit()
    db.refresh(target)
    return target

Base.metadata.create_all(bind=engine)
