"""
Simple key test: sends a tiny “Hi” prompt and logs the reply.
Supports:
- Standard OpenAI (uses OPENAI_API_KEY)
- Azure OpenAI (uses AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION)
"""

import logging
import os
import sys
import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def test_openai() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4.1"),
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 20,
    }
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=30,
    )
    logger.info("Status: %s", r.status_code)
    try:
        logger.info("%s", r.json())
    except Exception:
        logger.info("%s", r.text)


def test_azure() -> None:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    if not all([endpoint, api_key, deployment]):
        sys.exit("Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT")

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions"
    params = {"api-version": api_version}
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
    }
    data = {
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 20,
    }
    r = requests.post(url, params=params, headers=headers, json=data, timeout=30)
    logger.info("Status: %s", r.status_code)
    try:
        logger.info("%s", r.json())
    except Exception:
        logger.info("%s", r.text)


if __name__ == "__main__":
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        test_azure()
    else:
        test_openai()
