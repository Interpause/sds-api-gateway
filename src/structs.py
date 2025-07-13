"""Structs for API."""

from typing import List, Optional

from fastapi import UploadFile
from pydantic import BaseModel

# btw, can use Annotated even in pydantic to attach metadata for fastapi.


class RequestGenerate3D(BaseModel):
    """Request model for generating 3D content."""

    client_id: str
    # TODO: Support optional image in workflow.
    image: UploadFile
    prompt: Optional[str] = None


class ResponseGenerate3D(BaseModel):
    """Response model for 3D generation."""

    task_id: str


class RequestGenerationEvents(BaseModel):
    """Request model for checking generation event log."""

    client_id: str
    task_id: str
    n_received: int = 0


class ResponseGenerationEvents(BaseModel):
    """Response model for generation events."""

    events: List[str]
    n_received: int


class RequestGenerationStatus(BaseModel):
    """Request model for checking generation status."""

    client_id: str
    task_id: str


class ResponseGenerationStatus(BaseModel):
    """Response model for generation status."""

    status: str


class RequestGenerationResult(BaseModel):
    """Request model for getting generation result."""

    client_id: str
    task_id: str


class ResponseGenerationResult(BaseModel):
    """Response model for generation result."""

    success: bool
    url: Optional[str] = None


class RequestAudioTranscription(BaseModel):
    """Request model for audio transcription."""

    client_id: str
    audio_file: UploadFile


class ResponseAudioTranscription(BaseModel):
    """Response model for audio transcription."""

    transcription: str
