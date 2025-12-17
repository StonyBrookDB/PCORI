import logging
import os
import requests


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in your environment")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gpt-4.1",  # adjust if you have quota on a different model
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 20,
    }

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=30,
    )

    logger.info("Status: %s", resp.status_code)
    try:
        logger.info("%s", resp.json())
    except Exception:
        logger.info("%s", resp.text)


if __name__ == "__main__":
    main()
