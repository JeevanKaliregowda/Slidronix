
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.auth import current_user

router = APIRouter(tags=["live-inference"])


class CoordinateRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


@router.get("/predict/coordinate/status")
def coordinate_status(user=Depends(current_user)):
    # Import the heavy ML stack only when this endpoint is called.
    from app.live_inference import coordinate_status as run_status

    return run_status(user=user)


@router.post("/predict/coordinate")
def predict_coordinate(
    request: CoordinateRequest,
    user=Depends(current_user),
):
    # Import the heavy ML stack only when this endpoint is called.
    from app.live_inference import predict_coordinate as run_prediction
    from app.live_inference import CoordinateRequest as InferenceRequest

    inference_request = InferenceRequest(
        latitude=request.latitude,
        longitude=request.longitude,
    )
    return run_prediction(request=inference_request, user=user)
