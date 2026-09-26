"""
Gemini compatibility client.

Provides an OpenAI-style chat.completions interface backed by
Google's Gemini API so the existing travel agents can use Gemini
without rewriting every agent individually.
"""

import asyncio
import os
import random
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class GeminiCompatClient:
    """Small OpenAI-compatible wrapper around the Gemini API."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing from the environment."
            )

        self.client = genai.Client(api_key=api_key)
        self.chat = _ChatNamespace(self.client)


class _ChatNamespace:
    """Provides client.chat.completions.create(...)."""

    def __init__(self, client: genai.Client):
        self.completions = _CompletionsNamespace(client)


class _CompletionsNamespace:
    """Implements the OpenAI-style completions interface."""

    def __init__(self, client: genai.Client):
        self.client = client

    async def create(self, **kwargs: Any):
        """
        Create a Gemini response while exposing an OpenAI-compatible
        response structure to the existing travel planner agents.
        """

        messages = kwargs.get("messages", [])

        # The existing agents may still pass OpenAI model names.
        # We intentionally use the Gemini model configured in .env.
        model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.6-flash",
        )

        # ---------------------------------------------------------
        # Convert OpenAI-style messages to Gemini contents
        # ---------------------------------------------------------

        system_instruction = None
        conversation = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            # Handle OpenAI-style multipart content.
            if isinstance(content, list):
                content = "\n".join(
                    str(item.get("text", item))
                    if isinstance(item, dict)
                    else str(item)
                    for item in content
                )

            content = str(content)

            if role == "system":

                if system_instruction:
                    system_instruction += "\n\n" + content
                else:
                    system_instruction = content

            elif role == "assistant":

                conversation.append(
                    types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(
                                text=content
                            )
                        ],
                    )
                )

            else:

                conversation.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(
                                text=content
                            )
                        ],
                    )
                )

        # Gemini requires content.
        if not conversation:
            conversation = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text="Please respond to the request."
                        )
                    ],
                )
            ]

        # ---------------------------------------------------------
        # Generation configuration
        # ---------------------------------------------------------

        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens")

        config_kwargs = {
            "temperature": temperature,
        }

        if max_tokens:
            config_kwargs["max_output_tokens"] = max_tokens

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        # Preserve JSON response behavior used by existing agents.
        if kwargs.get("response_format"):
            config_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(
            **config_kwargs
        )

        # ---------------------------------------------------------
        # Gemini API request with retry / exponential backoff
        # ---------------------------------------------------------

        max_retries = 4
        response = None

        for attempt in range(max_retries):

            try:

                response = await self.client.aio.models.generate_content(
                    model=model,
                    contents=conversation,
                    config=config,
                )

                # Successful request.
                break

            except Exception as exc:

                error_text = str(exc)
                error_upper = error_text.upper()

                # -------------------------------------------------
                # IMPORTANT:
                # Daily free-tier quota exhaustion cannot be fixed
                # by retrying. Propagate immediately.
                # -------------------------------------------------

                daily_quota_exhausted = (
                    "GENERATE_REQUESTS_PER_DAY_PER_PROJECT-FREETIER"
                    in error_upper
                    or "GENERATE_REQUESTS_PER_DAY" in error_upper
                    or "GENERATE_CONTENT_FREE_TIER_REQUESTS"
                    in error_upper
                    or "QUOTA EXCEEDED" in error_upper
                    or "QUOTAVALUE" in error_upper
                )

                if daily_quota_exhausted:
                    raise

                # -------------------------------------------------
                # Retry only genuinely transient errors.
                # -------------------------------------------------

                retryable_errors = (
                    "400" not in error_text
                    and (
                        "408" in error_text
                        or "429" in error_text
                        or "500" in error_text
                        or "503" in error_text
                        or "504" in error_text
                        or "INTERNAL" in error_upper
                        or "UNAVAILABLE" in error_upper
                        or "RESOURCE_EXHAUSTED" in error_upper
                        or "DEADLINE_EXCEEDED" in error_upper
                    )
                )

                # If this isn't a temporary server/rate-limit problem,
                # immediately propagate the original exception.
                if not retryable_errors:
                    raise

                # No retries remaining.
                if attempt >= max_retries - 1:
                    raise

                # Exponential backoff with random jitter.
                #
                # Attempt 1 -> ~1-2 seconds
                # Attempt 2 -> ~2-3 seconds
                # Attempt 3 -> ~4-5 seconds
                #
                # Random jitter prevents multiple requests from retrying
                # at exactly the same time.
                delay = (2 ** attempt) + random.uniform(0.0, 1.0)

                print(
                    "\nGemini temporary API error detected."
                )

                print(
                    f"Error: {error_text[:180]}"
                )

                print(
                    f"Retrying in {delay:.1f} seconds "
                    f"(attempt {attempt + 2}/{max_retries})..."
                )

                await asyncio.sleep(delay)

        # ---------------------------------------------------------
        # Convert Gemini response to OpenAI-compatible response
        # ---------------------------------------------------------

        text = ""

        if response is not None:
            text = response.text or ""

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=text,
                    )
                )
            ],
            model=model,
            gemini_model=model,
        )