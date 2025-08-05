"""For hdri generation."""

import asyncio
import json
import logging

import requests

from src.api.comfy.shared import (
    CLIENT_ID,
    PROMPT_DIR,
    SERVER_ADDRESS,
    GenerationTask,
    PromptQueueResponse,
    get_file,
    get_history,
    track_progress,
)

# NOTE: This is very workflow dependent.
PROMPT_NAME = "panorama2.json"
COMFY_OUTPUT_IMG_NODE = "173"
COMFY_INPUT_TEXT_NODE = "27"

log = logging.getLogger("app.api.comfy.hdri")


# Clip description should be comma-separated, t5 description should be natural prose.
def queue_prompt(t5_description: str, clip_description: str):
    """Submit a prompt to ComfyUI's queue."""
    with open(PROMPT_DIR / PROMPT_NAME, "r") as file:
        prompt = json.load(file)

    prompt[COMFY_INPUT_TEXT_NODE]["inputs"]["clip_l"] = clip_description
    prompt[COMFY_INPUT_TEXT_NODE]["inputs"]["t5xxl"] = t5_description

    data = {"prompt": prompt, "client_id": CLIENT_ID}
    headers = {"Content-Type": "application/json"}
    resp = requests.post(f"{SERVER_ADDRESS}/prompt", json=data, headers=headers)
    obj = PromptQueueResponse.model_validate_json(resp.content)
    return obj


async def generate_hdri_prompt(t5_description: str, clip_description: str):
    """Generator that uses ComfyUI to generate HDRI from user's description."""
    prompt_meta = queue_prompt(t5_description, clip_description)

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
    imginfo = hist_data.outputs[COMFY_OUTPUT_IMG_NODE]["images"][0]
    raw_file = get_file(imginfo["filename"], imginfo["subfolder"], imginfo["type"])
    yield True, raw_file


def create_hdri_task(t5_description: str, clip_description: str):
    """Create a task for generating a 3D object."""
    return GenerationTask(generate_hdri_prompt(t5_description, clip_description))
