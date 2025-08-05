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

from src.api.comfy.hdri import create_hdri_task
from src.api.comfy.obj import create_3d_task
from src.api.comfy.shared import GenerationTask
from src.api.llm import ai_create_client, ai_describe_image
from src.structs import (
    RequestGenerate3D,
    RequestGenerateHDRI,
    RequestGenerationEvents,
    RequestGenerationResult,
    RequestGenerationStatus,
    ResponseGenerateTask,
    ResponseGenerationEvents,
    ResponseGenerationResult,
    ResponseGenerationStatus,
)
from src.utils import png_to_jpg

load_dotenv()


__all__ = ["create_app"]

log = logging.getLogger("app")

HOST_URL = "https://recovr.interpause.dev"

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
    ai_client = ai_create_client()
    # Map of client to map of task id to GenerationTask.
    workflow_tasks: Dict[str, Dict[str, GenerationTask]] = {}

    app.mount("/static", StaticFiles(directory="public"), name="static")

    @app.post("/3d_obj/add_task")
    async def add_obj_task(
        req: Annotated[RequestGenerate3D, Form(..., media_type="multipart/form-data")],
    ) -> ResponseGenerateTask:
        """Endpoint to generate 3D model from user's sketch and description."""
        image = Image.open(req.image.file)
        await asyncio.to_thread(image.load)  # Ensure the image is loaded in the thread
        client_id = req.client_id

        prompt = await ai_describe_image(ai_client, image, req.prompt)

        tasks = workflow_tasks.setdefault(client_id, {})
        task_id = uuid4().hex
        task = create_3d_task(image, prompt.strip())
        tasks[task_id] = task

        task.start()
        log.info(f"Started obj task {task_id} for client {client_id}.")

        return ResponseGenerateTask(task_id=task_id)

    @app.post("/3d_obj/get_status")
    async def check_obj_status(
        req: Annotated[RequestGenerationStatus, Form()],
    ) -> ResponseGenerationStatus:
        """Endpoint to check the status of obj workflow."""
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
        """Endpoint to get obj workflow events."""
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
        """Endpoint to get the result of obj workflow."""
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

    @app.post("/hdri/add_task")
    async def add_hdri_task(
        req: Annotated[
            RequestGenerateHDRI, Form(..., media_type="multipart/form-data")
        ],
    ) -> ResponseGenerateTask:
        """Endpoint to generate hdri from user's description."""
        client_id = req.client_id

        # TODO: function to expand the prompt for clip and t5.
        # t5_prompt, clip_prompt = await groq_expand_for_clip(groq, req.prompt)
        t5_prompt = req.prompt
        clip_prompt = t5_prompt

        tasks = workflow_tasks.setdefault(client_id, {})
        task_id = uuid4().hex
        task = create_hdri_task(t5_prompt.strip(), clip_prompt.strip())
        tasks[task_id] = task

        task.start()
        log.info(f"Started hdri task {task_id} for client {client_id}.")

        return ResponseGenerateTask(task_id=task_id)

    # TODO: Literally the same implementation, how to unify?
    @app.post("/hdri/get_status")
    async def check_hdri_status(
        req: Annotated[RequestGenerationStatus, Form()],
    ) -> ResponseGenerationStatus:
        """Endpoint to check the status of hdri workflow."""
        client_id = req.client_id
        task_id = req.task_id

        tasks = workflow_tasks.get(client_id, {})
        task = tasks.get(task_id)

        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")

        return ResponseGenerationStatus(status=task.status)

    @app.post("/hdri/get_events")
    async def get_hdri_events(
        req: Annotated[RequestGenerationEvents, Form()],
    ) -> ResponseGenerationEvents:
        """Endpoint to get hdri workflow events."""
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
    @app.post("/hdri/get_result")
    async def get_hdri_result(
        req: Annotated[RequestGenerationResult, Form()],
    ) -> ResponseGenerationResult:
        """Endpoint to get the result of hdri workflow."""
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
        raw_file = await asyncio.to_thread(png_to_jpg, raw_file)

        out_path = Path("public") / "generated" / client_id / "hdri" / f"{task_id}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # TODO: It shouldnt store this locally, rather to some cloud bucket...
        with open(out_path, "wb") as f:
            f.write(raw_file)

        url = f"{HOST_URL}/static/generated/{client_id}/hdri/{task_id}.jpg"
        return ResponseGenerationResult(success=True, url=url)

    return app
