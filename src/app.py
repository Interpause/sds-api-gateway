"""Main app."""

import asyncio
import logging
from pathlib import Path
from typing import Annotated, Dict
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.api.comfy import GenerationTask
from src.api.groq import groq_create_client, groq_describe_image, groq_transcribe_audio
from src.structs import (
    RequestAudioTranscription,
    RequestGenerate3D,
    RequestGenerationEvents,
    RequestGenerationResult,
    RequestGenerationStatus,
    ResponseAudioTranscription,
    ResponseGenerate3D,
    ResponseGenerationEvents,
    ResponseGenerationResult,
    ResponseGenerationStatus,
)

load_dotenv()


__all__ = ["create_app"]

log = logging.getLogger("app")

HOST_URL = "http://nixrobo.home.arpa:3000"

# TODO:
# - Setup websocket/SSE/streaming response version of events so client don't have to poll server for updates.
# - Implement the generation of 3D content using an Action model like ROS2 Actions. (cant stop tasks atm)
# - How to disable comfyui saving outputs, or delete them?
# - An actual user/security system: https://fastapi.tiangolo.com/tutorial/security/get-current-user/
#   - So how do we secure a cloud bucket with user permissions?


def create_app():
    """App factory.

    Creating the app within a function prevents mishaps if using multiprocessing.
    """
    app = FastAPI()
    groq = groq_create_client()
    # Map of client to map of task id to GenerationTask.
    workflow_tasks: Dict[str, Dict[str, GenerationTask]] = {}

    app.mount("/static", StaticFiles(directory="public"), name="static")

    @app.post("/3d_obj/add_task")
    async def add_obj_task(
        req: Annotated[RequestGenerate3D, Form(..., media_type="multipart/form-data")],
    ) -> ResponseGenerate3D:
        """Endpoint to generate 3D model from user's sketch and description."""
        image = await asyncio.to_thread(Image.open, req.image.file)
        client_id = req.client_id

        prompt = await groq_describe_image(groq, image, req.prompt)

        tasks = workflow_tasks.setdefault(client_id, {})
        task_id = uuid4().hex
        task = GenerationTask(image, prompt.strip())
        tasks[task_id] = task

        task.start()
        log.info(f"Started task {task_id} for client {client_id}.")

        return ResponseGenerate3D(task_id=task_id)

    @app.post("/3d_obj/get_status")
    async def check_obj_status(
        req: Annotated[RequestGenerationStatus, Form()],
    ) -> ResponseGenerationStatus:
        """Endpoint to check the status of a workflow."""
        client_id = req.client_id
        task_id = req.task_id

        tasks = workflow_tasks.get(client_id, {})
        task = tasks.get(task_id)

        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")

        return ResponseGenerationStatus(status=task.status)

    @app.post("/3d_obj/get_events")
    async def get_obj_events(
        req: Annotated[RequestGenerationEvents, Form()],
    ) -> ResponseGenerationEvents:
        """Endpoint to get workflow events."""
        client_id = req.client_id
        task_id = req.task_id
        n_received = req.n_received

        tasks = workflow_tasks.get(client_id, {})
        task = tasks.get(task_id)

        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")

        return ResponseGenerationEvents(
            events=task.event_log[n_received:], n_received=len(task.event_log)
        )

    # NOTE: As a side effect, results are only saved by the gateway if the user
    # requests it. ComfyUI still saves it tho but the filename gets lost.
    @app.post("/3d_obj/get_result")
    async def get_obj_result(
        req: Annotated[RequestGenerationResult, Form()],
    ) -> ResponseGenerationResult:
        """Endpoint to get the result of a workflow."""
        client_id = req.client_id
        task_id = req.task_id

        tasks = workflow_tasks.get(client_id, {})
        task = tasks.get(task_id)

        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")

        # if task.status != "COMPLETED":
        #     raise HTTPException(
        #         status_code=400, detail="Task is not completed yet.")

        raw_file = await task.result()
        if raw_file is None:
            return ResponseGenerationResult(success=False)

        out_path = Path("public") / "generated" / client_id / "obj" / f"{task_id}.glb"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # TODO: It shouldnt store this locally, rather to some cloud bucket...
        with open(out_path, "wb") as f:
            f.write(raw_file)

        url = f"{HOST_URL}/static/generated/{client_id}/obj/{task_id}.glb"
        return ResponseGenerationResult(success=True, url=url)

    @app.post("/audio/transcribe")
    async def transcribe_audio(
        req: Annotated[
            RequestAudioTranscription, Form(..., media_type="multipart/form-data")
        ],
    ) -> ResponseAudioTranscription:
        """Endpoint to transcribe audio."""
        transcription = await groq_transcribe_audio(groq, req.audio_file)
        return ResponseAudioTranscription(transcription=transcription)

    return app
