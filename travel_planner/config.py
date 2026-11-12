"""Settings, read from the .env file (or normal environment variables)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Free-tier friendly default. "Flash-Lite" models have a much bigger free
# daily allowance than the bigger "Flash" models.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Used automatically (in this order) when the chosen model is out of quota
# or not available to your key.
BUILTIN_FALLBACKS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

# Shown in the app's model picker.
MODEL_CHOICES = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
]


def _secret(name: str) -> str | None:
    """Read a value from Streamlit's secrets (used when deployed on Streamlit Community Cloud)."""
    try:
        import streamlit as st

        value = st.secrets.get(name)
        return None if value is None else str(value)
    except Exception:  # no secrets file, or not running inside Streamlit
        return None


def _get(name: str) -> str | None:
    """Environment variable / .env first, then Streamlit secrets."""
    return os.getenv(name) or _secret(name)


def _int(value: str | None, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _clean(value: str | None) -> str:
    """Remove spaces and accidental quotes around a value."""
    if value is None:
        return ""
    return value.strip().strip('"').strip("'").strip()


@dataclass
class Settings:
    api_key: str = ""
    model: str = DEFAULT_MODEL
    fallback_models: list[str] = field(default_factory=list)
    default_currency: str = "USD"
    save_to_supabase: bool = False
    supabase_url: str = ""
    supabase_key: str = ""
    # Public-demo protections (see README "Share it online")
    public_mode: bool = False      # hides saved trips, stores nothing on the server
    daily_trip_limit: int = 0      # max trips per day for everyone together (0 = no limit)
    session_trip_limit: int = 0    # max trips per visitor session (0 = no limit)

    @property
    def model_chain(self) -> list[str]:
        """Primary model first, then fallbacks, no duplicates."""
        chain: list[str] = []
        for name in [self.model, *self.fallback_models]:
            if name and name not in chain:
                chain.append(name)
        return chain


def get_settings(model_override: str | None = None) -> Settings:
    """Read settings fresh from .env / environment / Streamlit secrets every time."""
    load_dotenv(ENV_FILE, override=False)

    model = _clean(model_override) or _clean(_get("GEMINI_MODEL")) or DEFAULT_MODEL

    extra = _clean(_get("GEMINI_FALLBACK_MODELS"))
    fallbacks = [m.strip() for m in extra.split(",") if m.strip()] if extra else []
    for name in BUILTIN_FALLBACKS:
        if name not in fallbacks:
            fallbacks.append(name)

    public = _clean(_get("PUBLIC_MODE")).lower() in {"1", "true", "yes"}
    return Settings(
        api_key=_clean(_get("GEMINI_API_KEY")) or _clean(_get("GOOGLE_API_KEY")),
        model=model,
        fallback_models=fallbacks,
        default_currency=(_clean(_get("DEFAULT_CURRENCY")) or "USD").upper()[:3],
        save_to_supabase=_clean(_get("SAVE_TO_SUPABASE")).lower() in {"1", "true", "yes"},
        supabase_url=_clean(_get("SUPABASE_URL")),
        supabase_key=_clean(_get("SUPABASE_KEY")),
        public_mode=public,
        daily_trip_limit=_int(_get("DAILY_TRIP_LIMIT"), 40 if public else 0),
        session_trip_limit=_int(_get("SESSION_TRIP_LIMIT"), 5 if public else 0),
    )


def mask_key(key: str) -> str:
    """Show only the start of a secret, e.g. 'AQ.A... (53 characters)'."""
    if not key:
        return "(missing)"
    return f"{key[:4]}... ({len(key)} characters)"
