"""For functions related to ComfyUI."""
# Some parts referenced from: https://www.viewcomfy.com/blog/building-a-production-ready-comfyui-api

import asyncio
import io
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import imagehash
import requests
from PIL import Image
from pydantic import BaseModel, ValidationError
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

import src.prompts
from src.api.comfy_msg_structs import ComfyUIMessageAdapter

PROMPT_DIR = Path(src.prompts.__file__).parent.absolute()
PROMPT_NAME = "sketch23d_api_faster.json"
SERVER_ADDRESS = "http://nixrobo.home.arpa:8187"
WS_ADDRESS = "ws://nixrobo.home.arpa:8187"
CLIENT_ID = "literally_placeholder"

# TODO: This is very workflow dependent atm.
COMFY_OUTPUT_GLB_NODE = "154"
COMFY_INPUT_IMG_NODE = "196"
COMFY_INPUT_TEXT_NODE = "192"

log = logging.getLogger("app.api.comfy")


class ImageMetadata(BaseModel):
    """Metadata for uploaded image (Based off API response)."""

    name: str
    subfolder: str
    type: str


class PromptQueueResponse(BaseModel):
    """Response after submitting workflow (Based off API response)."""

    prompt_id: str
    number: int
    node_errors: dict


class HistoryResponse(BaseModel):
    """Response for workflow history (Based off API response)."""

    prompt: list
    outputs: dict
    status: dict
    meta: dict


async def track_progress(prompt_id: str):
    """Generator that interprets ComfyUI's progress reports."""
    async for ws in connect(f"{WS_ADDRESS}/ws?clientId={CLIENT_ID}"):
        try:
            async for raw in ws:
                msg: dict = json.loads(raw)
                try:
                    m = ComfyUIMessageAdapter.validate_python(msg, strict=True)
                # Don't reconnect on unknown msg type, just continue.
                except ValidationError as e:
                    log.error(f"Unknown message type: {msg}")
                    yield f"Unknown msg: {msg}"
                    continue

                # Filter for only msg related to current prompt, but don't filter
                # for general status messages.
                if hasattr(m.data, "prompt_id") and m.data.prompt_id != prompt_id:  # type: ignore
                    continue

                # Below is based on the specific ComfyUI commit inside comfy_msg_structs.py.
                if m.type == "status":
                    yield f"Queue Remaining: {m.data.status.exec_info.queue_remaining - 1}"
                elif m.type == "progress":
                    yield f"Progress ({m.data.node}): {m.data.value}/{m.data.max}"
                elif m.type == "executing":
                    yield f"Executing Node: {m.data.node}"
                elif m.type == "execution_cached":
                    yield f"Cached Nodes: {', '.join(m.data.nodes)}"
                # TODO: Possible to yield intermediate outputs here.
                elif m.type == "executed":
                    yield f"Executed Node: {m.data.node}"
                elif m.type == "execution_start":
                    yield f"Started: {m.data.prompt_id}"
                elif m.type == "execution_success":
                    yield f"Complete: {m.data.prompt_id}"
                    yield True
                    return

            # Combined with continue on unknown msg type above, this should catch
            # any case where the generation ends without a success message.
            raise RuntimeError("Connection closed without generation completion?")
        except ConnectionClosed:
            log.warning("WebSocket connection closed, retrying...")
            continue
        # If json invalid, maybe connection issue so reconnect.
        except json.JSONDecodeError as e:
            log.error(f"Error decoding JSON message: {e}")
            continue
        except Exception as e:
            log.error(f"Unexpected error (treat as fatal): {e}")
            yield f"Error: {e}"
            yield False
            return


def queue_prompt(image_metadata: ImageMetadata, sketch_description: str):
    """Submit a prompt to ComfyUI's queue."""
    with open(PROMPT_DIR / PROMPT_NAME, "r") as file:
        prompt = json.load(file)

    prompt[COMFY_INPUT_IMG_NODE]["inputs"]["image"] = image_metadata.name
    prompt[COMFY_INPUT_TEXT_NODE]["inputs"]["text"] = sketch_description

    data = {"prompt": prompt, "client_id": CLIENT_ID}
    headers = {"Content-Type": "application/json"}
    resp = requests.post(f"{SERVER_ADDRESS}/prompt", json=data, headers=headers)
    obj = PromptQueueResponse.model_validate_json(resp.content)
    return obj


def get_history(prompt_id: str):
    """Get workflow history (and hence results) for given prompt ID."""
    resp = requests.get(f"{SERVER_ADDRESS}/history/{prompt_id}")
    obj = resp.json()
    obj = HistoryResponse.model_validate(obj[prompt_id])
    return obj


def get_file(filename: str, subfolder: str, folder_type: str):
    """Get a file from ComfyUI's storage."""
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    resp = requests.get(f"{SERVER_ADDRESS}/view", params=params)
    return resp.content


def upload_image(image: Image.Image):
    """Upload an image to ComfyUI's storage."""
    buf = io.BytesIO()
    image.save(buf, format="webp", quality=100)
    buf.seek(0)
    filename = f"{imagehash.phash(image, hash_size=16)}.webp"

    files = {"image": (filename, buf, "image/webp")}
    data = {"type": "input", "overwrite": "false"}
    url = f"{SERVER_ADDRESS}/upload/image"
    resp = requests.post(url, files=files, data=data)

    obj = ImageMetadata.model_validate_json(resp.content)
    return obj


async def generate_3d_prompt(image: Image.Image, sketch_description: str):
    """Generator that uses ComfyUI to generate 3D object from user's sketch and description."""
    img_meta = upload_image(image)
    prompt_meta = queue_prompt(img_meta, sketch_description)

    async for status in track_progress(prompt_meta.prompt_id):
        if isinstance(status, str):
            yield False, status
        elif isinstance(status, bool):
            if status:
                break
            else:
                yield False, "An error occurred during generation."
                return

    await asyncio.sleep(1)
    hist_data = get_history(prompt_meta.prompt_id)
    filename = hist_data.outputs[COMFY_OUTPUT_GLB_NODE]["result"][0]
    raw_file = get_file(filename, "3D", "output")
    yield True, raw_file


# TODO: pretty limited; if we can parse the workflow events more accurately, this
# can be more useful.
TaskStatus = Literal["NOT_STARTED", "IN_PROGRESS", "COMPLETED", "FAILED"]


class GenerationTask:
    """Class to manage a generation task."""

    def __init__(self, image: Image.Image, description: str):
        """Initialize."""
        self.image = image
        self.description = description
        self.event_log = []
        self.status: TaskStatus = "NOT_STARTED"
        self.task = None
        self.timestamp = None
        self.duration = None
        self.error = None

    def start(self):
        """Start the generation task."""
        self.task = asyncio.create_task(self._process())

    async def _process(self):
        """Process task."""
        self.status = "IN_PROGRESS"
        self.timestamp = datetime.now()

        raw_file = None
        start_time = time.monotonic()

        try:
            gen = generate_3d_prompt(self.image, self.description)
            async for done, msg in gen:
                if not done:
                    self.event_log.append(msg)
                else:
                    assert isinstance(msg, bytes), "Expected raw file data."
                    raw_file = msg
                    break
            # Loop completes without yielding done=True, indicating an error.
            else:
                raise RuntimeError("Comfyui stopped without completion.")

        except Exception as e:
            self.error = e
            self.status = "FAILED"
            self.event_log.append(f"Error: {e}")
            self.duration = time.monotonic() - start_time
            return None

        self.status = "COMPLETED"
        self.duration = time.monotonic() - start_time
        return raw_file

    async def result(self):
        """Await the result of the generation task."""
        if self.task is None:
            raise RuntimeError("Task not started. Call start() first.")
        return await self.task
