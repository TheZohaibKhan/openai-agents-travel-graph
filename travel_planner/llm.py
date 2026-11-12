"""Talking to Gemini.

`GeminiLLM.generate()` asks Gemini for a answer that must match a Pydantic
schema (structured JSON). It handles the things that went wrong in the old
version: temporary overloads, per-minute limits, daily free-tier limits
(automatically switching to the next model), bad keys, and broken JSON.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------
# Friendly errors (each one carries a message a non-programmer can act on)
# --------------------------------------------------------------------------


class PlannerError(Exception):
    """Base class. `user_message` is safe and helpful to show on screen."""

    def __init__(self, user_message: str, technical: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.technical = technical


class MissingKeyError(PlannerError):
    pass


class ApiKeyError(PlannerError):
    pass


class QuotaExceededError(PlannerError):
    pass


class BusyError(PlannerError):
    pass


class NetworkProblemError(PlannerError):
    pass


class BadResponseError(PlannerError):
    pass


class BadRequestError(PlannerError):
    pass


MSG_MISSING_KEY = (
    "No Gemini API key was found. Open the file named .env in the project folder and add a line "
    "like GEMINI_API_KEY=your_key_here (no quotes, no spaces). You can create a free key at "
    "https://aistudio.google.com/apikey"
)
MSG_BAD_KEY = (
    "Google rejected the API key. Create a NEW key at https://aistudio.google.com/apikey, put it in the "
    ".env file as GEMINI_API_KEY=... (no quotes, no spaces), save the file and try again. "
    "If the error mentions ACCESS_TOKEN_TYPE_UNSUPPORTED, also run:  pip install -U google-genai"
)
MSG_QUOTA = (
    "Your free Gemini quota is used up for every model I tried (the free tier only allows a small number "
    "of requests per day). Wait until it resets (it resets once a day), or enable billing for your Google "
    "AI Studio project, or choose a different model in the sidebar."
)
MSG_BUSY = (
    "Google's Gemini servers are overloaded right now (this happens to every model at busy times, it is not "
    "your fault). I already tried the backup models too. Please wait a minute or two and try again."
)
MSG_NETWORK = "I could not reach Google. Check your internet connection (or VPN/firewall) and try again."
MSG_BAD_JSON = (
    "Gemini returned an answer I could not read after several tries. Please press the button again; "
    "if it keeps happening, pick a different model in the sidebar."
)
MSG_REGION = (
    "Google says the Gemini API is not available from your current location. Try a different network "
    "or check Google's list of supported regions."
)


# --------------------------------------------------------------------------
# Classifying raw errors from the Google library
# --------------------------------------------------------------------------

_NETWORK_NAMES = {
    "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout",
    "TimeoutException", "RemoteProtocolError", "ReadError", "ConnectionError",
    "ConnectionResetError", "TimeoutError", "gaierror",
}


def _error_text(exc: Exception) -> str:
    parts = [str(exc)]
    for attr in ("message", "status", "details"):
        value = getattr(exc, attr, None)
        if value:
            parts.append(str(value))
    return " ".join(parts)


def classify_error(exc: Exception) -> tuple[str, float | None]:
    """Return (kind, retry_after_seconds).

    kind is one of: auth, region, daily_quota, rate, model_missing, busy,
    bad_request, network, unknown.
    """
    text = _error_text(exc)
    upper = text.upper()
    code = getattr(exc, "code", None)
    if not isinstance(code, int):
        code = None

    if "LOCATION IS NOT SUPPORTED" in upper:
        return "region", None

    if (
        code in (401, 403)
        or "API KEY NOT VALID" in upper
        or "API_KEY_INVALID" in upper
        or "API KEY EXPIRED" in upper
        or "ACCESS_TOKEN_TYPE_UNSUPPORTED" in upper
        or "PERMISSION_DENIED" in upper
    ):
        return "auth", None

    if code == 429 or "RESOURCE_EXHAUSTED" in upper:
        # A daily limit (or "limit: 0" = this model is not free for you) will
        # not fix itself by waiting a few seconds, so we switch model instead.
        if "PERDAY" in upper.replace(" ", "").replace("_", "") or "PER DAY" in upper or "LIMIT: 0" in upper:
            return "daily_quota", None
        match = re.search(r"retry in ([\d.]+)\s*s", text, re.IGNORECASE) or re.search(
            r"retryDelay['\"]?\s*[:=]\s*['\"]?([\d.]+)s", text
        )
        return "rate", float(match.group(1)) if match else None

    if code == 404 or "IS NOT FOUND" in upper or "NOT_FOUND" in upper:
        return "model_missing", None

    if code in (500, 502, 503, 504) or "UNAVAILABLE" in upper or "OVERLOADED" in upper or "DEADLINE_EXCEEDED" in upper:
        return "busy", None

    if code == 400:
        return "bad_request", None

    if type(exc).__name__ in _NETWORK_NAMES or isinstance(exc, (ConnectionError, TimeoutError)):
        return "network", None

    return "unknown", None


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


# --------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------


class GeminiLLM:
    """Structured-output wrapper around the official google-genai library."""

    MAX_RATE_WAITS = 2  # how many times we wait out a per-minute limit
    MAX_BUSY_TRIES = 3
    MAX_JSON_TRIES = 3

    def __init__(
        self,
        api_key: str,
        models: list[str],
        timeout_seconds: int = 120,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not api_key:
            raise MissingKeyError(MSG_MISSING_KEY)
        if not models:
            raise PlannerError("No Gemini model is configured.")

        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_seconds * 1000),
        )
        self.models = list(models)
        self._skip: set[str] = set()  # models whose quota is used up this run
        self._sleep = sleep
        self.models_used: list[str] = []

    # -- public ----------------------------------------------------------

    def generate(self, schema: type[T], system: str, prompt: str, temperature: float = 0.6) -> T:
        """Ask Gemini and return a validated `schema` object."""
        kinds: set[str] = set()
        technical = ""
        for model in self.models:
            if model in self._skip:
                continue
            try:
                result = self._generate_with_model(model, schema, system, prompt, temperature)
            except _SwitchModel as switch:
                kinds.add(switch.kind)
                technical = switch.technical or technical
                self._skip.add(model)
                continue
            if model not in self.models_used:
                self.models_used.append(model)
            return result

        if "busy" in kinds:
            raise BusyError(MSG_BUSY, technical=technical)
        if kinds == {"model_missing"}:
            raise BadRequestError(
                "None of the configured Gemini models are available for your key. "
                "Pick another model in the sidebar or set GEMINI_MODEL in the .env file.",
                technical=technical,
            )
        raise QuotaExceededError(MSG_QUOTA, technical=technical)

    # -- internals ---------------------------------------------------------

    def _call(self, model: str, schema: type[T], system: str, prompt: str, temperature: float) -> str:
        types = self._types
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )
        response = self._client.models.generate_content(model=model, contents=prompt, config=config)
        return response.text or ""

    def _generate_with_model(self, model, schema, system, prompt, temperature):
        rate_waits = 0
        busy_tries = 0
        json_tries = 0

        while True:
            try:
                text = self._call(model, schema, system, prompt, temperature)
                if not text.strip():
                    raise ValueError("Gemini returned an empty answer.")
                return schema.model_validate_json(_strip_code_fences(text))

            except (ValidationError, ValueError) as exc:
                json_tries += 1
                if json_tries >= self.MAX_JSON_TRIES:
                    raise BadResponseError(MSG_BAD_JSON, technical=str(exc)[:500]) from exc
                temperature = max(0.2, temperature - 0.2)  # be more careful next time
                continue

            except PlannerError:
                raise

            except Exception as exc:  # errors from the Google library / network
                kind, retry_after = classify_error(exc)
                technical = _error_text(exc)[:600]

                if kind == "auth":
                    raise ApiKeyError(MSG_BAD_KEY, technical=technical) from exc
                if kind == "region":
                    raise PlannerError(MSG_REGION, technical=technical) from exc
                if kind in ("daily_quota", "model_missing"):
                    raise _SwitchModel(kind, technical) from exc
                if kind == "rate":
                    rate_waits += 1
                    if rate_waits > self.MAX_RATE_WAITS:
                        raise _SwitchModel("daily_quota", technical) from exc
                    self._sleep(min(max(retry_after or 20.0, 1.0) + 1.0, 70.0))
                    continue
                if kind in ("busy", "network"):
                    busy_tries += 1
                    if busy_tries >= self.MAX_BUSY_TRIES:
                        if kind == "network":
                            raise NetworkProblemError(MSG_NETWORK, technical=technical) from exc
                        # An overloaded model: give up on it and try the next one.
                        raise _SwitchModel("busy", technical) from exc
                    self._sleep(2.0 * (2 ** (busy_tries - 1)))
                    continue
                if kind == "bad_request":
                    raise BadRequestError(
                        "Gemini rejected the request. Details: " + technical[:300], technical=technical
                    ) from exc
                raise PlannerError(
                    "Something unexpected went wrong while talking to Gemini: " + technical[:300],
                    technical=technical,
                ) from exc


class _SwitchModel(Exception):
    """Internal signal: give up on this model and try the next one."""

    def __init__(self, kind: str, technical: str = ""):
        super().__init__(kind)
        self.kind = kind
        self.technical = technical


def build_llm(settings) -> GeminiLLM:
    """Create the real Gemini client from Settings."""
    return GeminiLLM(api_key=settings.api_key, models=settings.model_chain)
