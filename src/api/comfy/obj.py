"""For object generation."""

import asyncio
import json
import logging

import requests
from PIL import Image

from src.api.comfy.shared import (
    CLIENT_ID,
    PROMPT_DIR,
    SERVER_ADDRESS,
    GenerationTask,
    ImageMetadata,
    PromptQueueResponse,
    get_file,
    get_history,
    track_progress,
    upload_image,
)

# NOTE: This is very workflow dependent.
PROMPT_NAME = "sketch23d_api_faster.json"
COMFY_OUTPUT_GLB_NODE = "154"
COMFY_INPUT_IMG_NODE = "196"
COMFY_INPUT_TEXT_NODE = "192"

log = logging.getLogger("app.api.comfy.obj")


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


def create_3d_task(image: Image.Image, sketch_description: str):
    """Create a task for generating a 3D object."""
    return GenerationTask(generate_3d_prompt(image, sketch_description))
