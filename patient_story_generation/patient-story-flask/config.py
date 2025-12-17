import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    flask_env: str = os.getenv("FLASK_ENV", "development")
    port: int = int(os.getenv("PORT", "5000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    primary_llm_provider: str = os.getenv("PRIMARY_LLM_PROVIDER", "gemini").lower()
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1")
    openai_api_base: str = os.getenv("OPENAI_API_BASE", "")  # Azure/OpenAI endpoint; leave blank for api.openai.com
    openai_api_version: str = os.getenv("OPENAI_API_VERSION", "")  # Required for Azure; ignored for api.openai.com
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

    debug_log_payload: bool = os.getenv("DEBUG_LOG_PAYLOAD", "false").lower() == "true"
    debug_log_payload_max_chars: int = int(os.getenv("DEBUG_LOG_PAYLOAD_MAX_CHARS", "20000"))

    use_db_cache: bool = os.getenv("USE_DB_CACHE", "false").lower() == "true"
    db_host: str = os.getenv("DB_HOST", "127.0.0.1")
    db_port: int = int(os.getenv("DB_PORT", "3306"))
    db_user: str = os.getenv("DB_USER", "root")
    db_password: str = os.getenv("DB_PASSWORD", "")
    db_name: str = os.getenv("DB_NAME", "pcori_dashboard")
    patient_table: str = os.getenv("PATIENT_TABLE", "patient_data")
    story_cache_table: str = os.getenv("STORY_CACHE_TABLE", "story_cache")

    disclaimer: str = (
        "This AI-generated summary is for educational support only and is not a substitute for professional medical judgment."
    )

    @classmethod
    def validate(cls) -> None:
        cfg = cls()
        if cfg.primary_llm_provider == "gemini" and not cfg.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for gemini provider")
        if cfg.primary_llm_provider == "openai" and not cfg.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for openai provider")
        if cfg.primary_llm_provider == "anthropic" and not cfg.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for anthropic provider")

    @classmethod
    def print_config(cls) -> str:
        cfg = cls()
        lines = [
            "============================================================",
            "CURRENT CONFIGURATION",
            "============================================================",
            f"Primary Provider: {cfg.primary_llm_provider}",
            f"Gemini Model: {cfg.gemini_model}",
            f"OpenAI Model: {cfg.openai_model}",
            f"Anthropic Model: {cfg.anthropic_model}",
            f"Log Level: {cfg.log_level}",
            f"DB Cache: {'enabled' if cfg.use_db_cache else 'disabled'}",
            f"DB Host: {cfg.db_host}:{cfg.db_port}",
            "============================================================",
        ]
        return "\n".join(lines)
