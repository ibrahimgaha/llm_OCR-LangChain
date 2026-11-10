import base64
import logging
import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from PIL import Image, UnidentifiedImageError
from requests.exceptions import Timeout

MODEL_NAME = "mistralai/mistral-large-4-1t-a52b"
MAX_IMAGE_MB = 5
logger = logging.getLogger(__name__)

OCR_INSTRUCTION = (
    "Extract all readable text from this image. Preserve the original language, "
    "spelling, and line breaks as much as possible. Return only the extracted text. "
    "Do not summarize, translate, or invent missing words."
)
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            [
                {"type": "text", "text": OCR_INSTRUCTION},
                {"type": "image_url", "image_url": {"url": "{image_url}"}},
            ],
        )
    ]
)


def image_to_data_url(image_bytes: bytes) -> str:
    """Validate an upload and encode its original bytes for NVIDIA."""
    if len(image_bytes) > MAX_IMAGE_MB * 1024 * 1024:
        raise ValueError(f"Choose an image smaller than {MAX_IMAGE_MB} MB.")

    try:
        with Image.open(BytesIO(image_bytes), formats=["PNG", "JPEG"]) as image:
            if image.width * image.height > 25_000_000:
                raise ValueError("Choose an image with at most 25 megapixels.")
            mime_type = Image.MIME[image.format]
            image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Upload a valid PNG or JPEG image.") from error

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_text(image_url: str) -> str:
    """Send a validated image data URL to NVIDIA and return its transcription."""
    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set NVIDIA_API_KEY in your .env file and try again.")

    try:
        model = ChatNVIDIA(
            model=MODEL_NAME,
            api_key=api_key,
            temperature=0,
            max_completion_tokens=4096,
            timeout=90,
        )
        response = (PROMPT | model).invoke({"image_url": image_url})
    except (Timeout, TimeoutError):
        raise RuntimeError("The NVIDIA request timed out. Please try again.") from None
    except Exception as error:
        logger.warning("NVIDIA OCR request failed (%s)", type(error).__name__)
        raise RuntimeError(
            "The NVIDIA request failed. Check your connection, API key, and model access."
        ) from None

    finish_reason = response.response_metadata.get("finish_reason")
    if finish_reason == "length":
        raise RuntimeError("NVIDIA reached the output limit. Try a crop with less text.")

    text = response.text
    if not text.strip():
        raise RuntimeError("NVIDIA returned no extracted text. Try a clearer image with visible text.")

    return text
