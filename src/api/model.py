"""Data formats for predictions."""
from datetime import datetime

from pydantic import BaseModel, Field


class VisionModelRequest(BaseModel):
    """Vision Request model."""

    file_path: str = Field(
        ...,
        title="absolute path to the uploaded video file",
        example="/home/pensiv/test.mp4",
    )


class AudioModelRequest(BaseModel):
    """Audio Request model"""

    file_path: str = Field(
        ...,
        title="absolute path to the uploaded video file",
        example="/home/pensiv/test.mp4",
    )


class PensivModelRequest(BaseModel):
    """Full Pensiv Request model"""

    id: int = Field(
        ...,
        title="user-specific ID",
    )
    date: str = Field(
        default=datetime.now().strftime("%Y%m%d%H%M"),
        title="datetime at time of request",
    )
    filename: str = Field(
        ...,
        title="name of input video",
        example="test.mp4",
    )
    path: str = Field(
        default=f"/data/service/shorts/admin/model/input/{datetime.now().strftime('%Y%m%d')}",
        title="absolute path of the parent directory of the file",
        example="/data/service/shorts/admin/model/input/test",
    )
    threshold: float = Field(
        ...,
        title="cutoff threshold for highlight detection",
        example=0.6,
    )


class PredictionResponse(BaseModel):
    """Prediction Response model."""

    prediction: list[float] = (Field(..., title="Prediction result", example=[]),)
