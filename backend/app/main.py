import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

app = FastAPI(
    title="Slidronix API",
    description="API for the Slidronix landslide risk assessment platform.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["System"])
def root():
    return {"name": "Slidronix API", "status": "running", "version": app.version}


@app.get("/health", tags=["System"])
def health():
    return {"status": "healthy", "service": "slidronix-backend"}


from app.predictions import router as predictions_router
from app.lazy_live_inference import router as live_inference_router
from app.incidents import router as incidents_router
from app.auth import router as auth_router

app.include_router(predictions_router)
app.include_router(live_inference_router)
app.include_router(incidents_router)
app.include_router(auth_router)
