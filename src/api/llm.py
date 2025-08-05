"""For functions related to AI API."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os

from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel

log = logging.getLogger("app.api.ai")

# Placeholder model - update with actual Google GenAI model
AI_MODEL = "gemini-2.5-flash-lite"

EMERGENCY_BYPASS = False


def ai_create_client():
    """Initialize AI client."""
    api_key = os.getenv("GOOGLE_GENAI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_GENAI_API_KEY environment variable is not set.")

    return genai.Client(api_key=api_key)


class SketchPrompt(BaseModel):
    """Model for sketch prompt."""

    prompt: str


PROMPT_EXAMPLES = [
    "3D product render style, futuristic Pine Green vehicle, finely detailed, purism, ue 5, a computer rendering, minimalism, octane render, 4k",
    "3D product render, futuristic chair, finely detailed, purism, ue 5, a computer rendering, minimalism, octane render, 4k",
    "3D product render, futuristic helmet, finely detailed, purism, ue 5, a computer rendering, minimalism, octane render, 4k",
    "3D Product render, futuristic ((ceramic)) bottle, finely detailed, purism, ue 5, a computer rendering, minimalism, octane render, 4k",
    "3D product render, futuristic kettle, finely detailed, purism, ue 5, a computer rendering, minimalism, octane render, 4k",
    "3D Product render style, futuristic lamp, finely detailed, purism, ue 5, a computer rendering, minimalism, octane render, 4k",
]


async def ai_describe_image(
    client: genai.Client, image: Image.Image, description: str | None = None
):
    """Generate prompt based off user's sketch and description."""
    if EMERGENCY_BYPASS:
        return f"3D product render, futuristic {description}, finely detailed, purism, ue 5, a computer rendering, minimalism, octane render, 4k"

    buf = io.BytesIO()
    await asyncio.to_thread(image.save, buf, format="webp", quality=70)
    img_bytes = buf.getvalue()
    buf.close()

    prompt = f"""\
Example prompts for an AI art model specialized in Product Design renders:
{"\n".join(f" - {example}" for example in PROMPT_EXAMPLES)}\
"""

    if description is not None and description.strip() != "":
        prompt += f"\n\nTake into consideration the user's description of their sketch:\n{description}"

    prompt += f"""

Given the user's sketch image, your task is to output a prompt for the AI art model. Give your output in JSON format with the following schema:
{json.dumps(SketchPrompt.model_json_schema())}\
"""

    resp = await client.aio.models.generate_content(
        model=AI_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(data=img_bytes, mime_type="image/webp"),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SketchPrompt,
        ),
    )

    log.info(f"PROMPT:\n{prompt}")
    log.info(f"RESPONSE:\n{resp.text}")

    obj: SketchPrompt = resp.parsed  # type: ignore
    return obj.prompt
