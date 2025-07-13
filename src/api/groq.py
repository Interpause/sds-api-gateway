"""For functions related to GroqCloud API."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os

import groq
from groq.types.chat.completion_create_params import (
    ResponseFormatResponseFormatJsonSchema,
    ResponseFormatResponseFormatJsonSchemaJsonSchema,
)
from PIL import Image
from pydantic import BaseModel

log = logging.getLogger("app.api.groq")

# https://console.groq.com/docs/speech-to-text
STT_MODEL = "distil-whisper-large-v3-en"
# https://console.groq.com/docs/vision
VLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def groq_create_client():
    """Initialize groq client."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    return groq.AsyncClient(api_key=api_key)


async def groq_transcribe_audio(client: groq.AsyncClient, file):
    """Transcribe audio using the groq client."""
    resp = await client.audio.transcriptions.create(
        model=STT_MODEL,
        file=file,
        language="en",
        response_format="verbose_json",
    )

    return resp.text


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


async def groq_describe_image(
    client: groq.AsyncClient, image: Image.Image, description: str | None = None
):
    """Generate prompt based off user's sketch and description."""
    buf = io.BytesIO()
    await asyncio.to_thread(image.save, buf, format="webp", quality=70)
    img_bytes = buf.getvalue()
    buf.close()

    b64_img = base64.b64encode(img_bytes).decode("utf-8")

    prompt = f"""\
Example prompts for an AI art model specialized in Product Design renders:
{"\n".join(f" - {example}" for example in PROMPT_EXAMPLES)}

Given the user's sketch image, your task is to output a prompt for the AI art model. Give your output in JSON format with the following schema:
{json.dumps(SketchPrompt.model_json_schema())}\
"""

    if description is not None:
        prompt += f"\n\nTake into very slight consideration the user's description of their sketch:\n{description}"

    resp = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/webp;base64,{b64_img}"},
                    },
                ],
            }
        ],
        response_format=ResponseFormatResponseFormatJsonSchema(
            type="json_schema",
            json_schema=ResponseFormatResponseFormatJsonSchemaJsonSchema(
                description="Prompt for the AI art model.",
                name="SketchPrompt",
                schema=SketchPrompt.model_json_schema(),
                strict=False,
            ),
        ),
        model=VLM_MODEL,
    )

    log.info(f"PROMPT:\n{prompt}")
    log.info(f"RESPONSE:\n{resp.choices[0].message}")

    obj = SketchPrompt.model_validate_json(resp.choices[0].message.content or "")
    return obj.prompt
