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


class HdriPrompts(BaseModel):
    """Model for HDRI prompts."""

    t5_prompt: str
    clip_prompt: str


OBJ_GEN_PROMPT_EXAMPLES = [
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
{"\n".join(f" - {example}" for example in OBJ_GEN_PROMPT_EXAMPLES)}\
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


HDRI_GEN_T5_PROMPT_EXAMPLES = [
    "Equirectangular 360 degree panorama. A dense, ancient forest is bathed in the soft glow of dawn. The trees, towering giants with twisted trunks and vast canopies, are covered in a thick layer of emerald-green moss. A small, winding path made of smooth river stones winds its way through the underbrush, disappearing into the shadows. Sunlight filters through the leaves, casting dappled patterns on the forest floor. A gentle babbling brook can be heard nearby, with the sound of birdsong echoing through the trees.",
    "Equirectangular 360 degree panorama. A vast, snow-covered mountain range stretches endlessly under a crystal-clear azure sky. The peaks are jagged and majestic, catching the golden light of late afternoon. In the foreground, a pristine alpine lake reflects the surrounding mountains like a perfect mirror. The air is crisp and pure, with wisps of clouds drifting between the highest summits. A few hardy pine trees dot the rocky landscape, their dark green needles contrasting beautifully with the white snow.",
    "Equirectangular 360 degree panorama. A serene tropical beach with powdery white sand extends in all directions. Crystal-clear turquoise waters gently lap against the shore, while tall coconut palms sway gracefully in the warm breeze. The sky is painted in brilliant shades of orange and pink as the sun sets on the horizon. Seabirds glide overhead, and the distant sound of waves creates a peaceful atmosphere. A few pieces of driftwood lie scattered on the sand, polished smooth by the ocean.",
]

HDRI_GEN_CLIP_PROMPT_EXAMPLES = [
    "equirectangular 360 degree panorama, dense ancient forest, at dawn, trees with twisted trunks, covered with moss, winding path of rounded stones, sunlight through canopy of leaves, small river",
    "equirectangular 360 degree panorama, snow-covered mountain range, azure sky, jagged peaks, golden afternoon light, alpine lake, mirror reflection, wisps of clouds, pine trees, rocky landscape",
    "equirectangular 360 degree panorama, tropical beach, white sand, turquoise waters, coconut palms, warm breeze, sunset, orange and pink sky, seabirds, ocean waves, driftwood",
]


# NOTE: Model page says to avoid saying "spherical projection" since that tends to result in non-equirectangular spherical images.
async def ai_expand_prompt(client: genai.Client, prompt: str):
    """Expand the prompt for T5 and CLIP models."""
    if EMERGENCY_BYPASS:
        return (
            f"equirectangular 360 degree panorama {prompt}",
            f"equirectangular 360 degree panorama {prompt}",
        )

    system_prompt = f"""\
Example T5 prompts for HDRI environment generation (natural prose style):
{"\n".join(f" - {example}" for example in HDRI_GEN_T5_PROMPT_EXAMPLES)}

Example CLIP prompts for HDRI environment generation (comma-separated tag style):
{"\n".join(f" - {example}" for example in HDRI_GEN_CLIP_PROMPT_EXAMPLES)}

Your task is to expand the user's prompt into two different versions for HDRI environment generation:

1. T5 prompt: A detailed, descriptive prompt written in natural, flowing prose that vividly describes the environment
2. CLIP prompt: A detailed, tag-based prompt written as comma-separated keywords/tags that describe the same environment

Both prompts MUST start with "Equirectangular 360 degree panorama" (T5) or "equirectangular 360 degree panorama" (CLIP). Avoid using "spherical projection" as that tends to result in non-equirectangular images. Both prompts should describe the same scene/environment but in their respective styles.

User's prompt: {prompt}

Generate both expanded prompts in JSON format with the following schema:
{json.dumps(HdriPrompts.model_json_schema())}\
"""

    resp = await client.aio.models.generate_content(
        model=AI_MODEL,
        contents=[system_prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=HdriPrompts,
        ),
    )

    log.info(f"PROMPT:\n{system_prompt}")
    log.info(f"RESPONSE:\n{resp.text}")

    obj: HdriPrompts = resp.parsed  # type: ignore
    return obj.t5_prompt, obj.clip_prompt
