"""Run this to find out why something isn't working:   python check_setup.py

It checks your .env file, your API key, and makes one tiny test call to Gemini.
"""

from __future__ import annotations

import sys


def line(ok: bool | None, text: str) -> None:
    icon = {True: "[ OK ]", False: "[FAIL]", None: "[INFO]"}[ok]
    print(f"{icon} {text}")


def main() -> int:
    print("=== AI Travel Planner - setup check ===\n")
    problems = 0

    line(None, f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        line(False, "Python 3.10 or newer is required.")
        return 1

    missing = []
    for module, pip_name in [
        ("streamlit", "streamlit"),
        ("google.genai", "google-genai"),
        ("langgraph", "langgraph"),
        ("pydantic", "pydantic"),
        ("dotenv", "python-dotenv"),
    ]:
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)
    if missing:
        line(False, "Missing libraries: " + ", ".join(missing))
        print("       Fix:  pip install -r requirements.txt")
        return 1
    line(True, "All required libraries are installed.")

    from travel_planner.config import ENV_FILE, get_settings, mask_key
    from travel_planner.llm import GeminiLLM, PlannerError

    line(ENV_FILE.exists(), f".env file {'found' if ENV_FILE.exists() else 'NOT found'} at {ENV_FILE}")
    if not ENV_FILE.exists():
        print("       Fix:  copy .env.example to .env  and put your key in it.")
        problems += 1

    settings = get_settings()
    if not settings.api_key:
        line(False, "GEMINI_API_KEY is missing or empty in .env")
        return 1
    line(True, f"GEMINI_API_KEY found: {mask_key(settings.api_key)}")
    line(None, "Models that will be tried, in order: " + ", ".join(settings.model_chain))

    print("\nMaking one small test call to Gemini (this uses 1 request of your quota)...")
    from pydantic import BaseModel as _BM

    class Ping(_BM):
        reply: str

    try:
        llm = GeminiLLM(settings.api_key, settings.model_chain)
        result = llm.generate(Ping, "Reply with the single word OK in the 'reply' field.", "ping", temperature=0.0)
        line(True, f"Gemini answered using model '{llm.models_used[-1]}': {result.reply!r}")
        print("\nEverything works. Start the app with:  streamlit run app.py")
        return problems
    except PlannerError as exc:
        line(False, exc.user_message)
        if exc.technical:
            print("\nTechnical details:\n  " + exc.technical)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
