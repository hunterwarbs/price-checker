#!/usr/bin/env python3
"""
OCR text extraction using Fireworks AI Qwen2.5-VL (serverless VLM).
Replaces local EasyOCR/Tesseract with remote vision-language OCR.
"""

import base64
import logging
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .config import (
    FIREWORKS_API_KEY,
    FIREWORKS_BASE_URL,
    FIREWORKS_VL_MODEL,
)

logger = logging.getLogger(__name__)


def _read_image_as_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class OCRAnalyzer:
    def __init__(self) -> None:
        if not FIREWORKS_API_KEY:
            raise RuntimeError("FIREWORKS_API_KEY is not set in environment")
        self.client = OpenAI(base_url=FIREWORKS_BASE_URL, api_key=FIREWORKS_API_KEY)
        self.model = FIREWORKS_VL_MODEL
        logger.info(f"Fireworks OCR ready (model={self.model})")

    def extract_text_from_image(self, image_path: str, fast_mode: bool = True) -> str:
        """Extract text from an image via Fireworks VLM.

        fast_mode retained for signature compatibility; serverless OCR is single-shot.
        """
        try:
            path = Path(image_path)
            if not path.exists():
                logger.error(f"Image not found: {image_path}")
                return ""

            b64 = _read_image_as_base64(path)

            # Use OpenAI-compatible Responses API with Fireworks base_url
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all visible text from this image, preserving reading order as best as possible."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                        ],
                    }
                ],
                max_tokens=1500,
                temperature=0.0,
            )

            text = response.choices[0].message.content or ""
            cleaned = self._clean_extracted_text(text)
            logger.debug(f"Fireworks OCR extracted {len(cleaned)} characters")
            return cleaned

        except Exception as e:
            logger.error(f"Error extracting text via Fireworks OCR: {e}")
            return ""

    def _clean_extracted_text(self, text: str) -> str:
        if not text:
            return ""
        # Basic normalization; keep it minimal because the VLM returns structured text
        return text.strip()
    
 