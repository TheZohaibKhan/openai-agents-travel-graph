"""Saving plans: always to a local folder, optionally also to Supabase."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from travel_planner.config import PROJECT_ROOT, Settings
from travel_planner.models import FinalPlan

SAVE_DIR = PROJECT_ROOT / "saved_plans"
SUPABASE_TABLE = "trip_plans"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "trip"


def save_plan_local(final: FinalPlan, folder: Path | None = None) -> Path:
    folder = folder or SAVE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = folder / f"{stamp}_{_slug(final.request.destination)}.json"
    path.write_text(final.model_dump_json(indent=2), encoding="utf-8")
    return path


def list_saved_plans(folder: Path | None = None) -> list[Path]:
    folder = folder or SAVE_DIR
    if not folder.exists():
        return []
    return sorted(folder.glob("*.json"), reverse=True)


def load_plan(path: Path) -> FinalPlan:
    return FinalPlan.model_validate_json(path.read_text(encoding="utf-8"))


def save_plan_supabase(final: FinalPlan, settings: Settings, client=None) -> str:
    """Optional. Needs `pip install supabase` and the table from supabase_schema.sql.

    Returns a short status message (never raises, so a Supabase problem can't
    break the trip planner).
    """
    if not (settings.supabase_url and settings.supabase_key):
        return "Supabase is not configured (SUPABASE_URL / SUPABASE_KEY missing)."
    try:
        if client is None:
            from supabase import create_client

            client = create_client(settings.supabase_url, settings.supabase_key)
        client.table(SUPABASE_TABLE).insert(
            {
                "destination": final.request.destination,
                "request": json.loads(final.request.model_dump_json()),
                "plan": json.loads(final.model_dump_json()),
            }
        ).execute()
        return "Saved to Supabase."
    except ImportError:
        return "Supabase library not installed. Run:  pip install supabase"
    except Exception as exc:  # noqa: BLE001 - we want to report, not crash
        return f"Could not save to Supabase: {str(exc)[:200]}"
