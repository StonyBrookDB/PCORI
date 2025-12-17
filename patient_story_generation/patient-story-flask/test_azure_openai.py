import logging
import os
import sys

try:
    from openai import AzureOpenAI
except ImportError:
    sys.exit("Install the OpenAI SDK first: pip install openai>=1.11.0")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    if not all([endpoint, api_key, deployment]):
        sys.exit("Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT")

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )

    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=20,
        )
        logger.info("Status: 200")
        logger.info("%s", resp.choices[0].message.content)
    except Exception as exc:  # noqa: BLE001
        logger.error("Azure OpenAI call failed: %r", exc)


if __name__ == "__main__":
    main()
