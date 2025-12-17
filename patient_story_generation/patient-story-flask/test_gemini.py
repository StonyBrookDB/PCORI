import logging
import os
import sys

import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY in your environment")

    genai.configure(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    try:
        resp = genai.GenerativeModel(model).generate_content("Say hello")
        logger.info("Model: %s", model)
        logger.info("Response text:")
        logger.info("%s", resp.text if hasattr(resp, "text") else resp)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error calling Gemini: %r", exc)


if __name__ == "__main__":
    main()
