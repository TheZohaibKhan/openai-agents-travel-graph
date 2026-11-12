"""Share links: store a finished plan in Supabase and open it later with ?plan=<id>.

Uses Supabase's plain web API (no extra library). Needs the `trip_plans` table from
supabase_schema.sql and a SECRET key (kept only in your server secrets).
Anyone who has the link can view that plan; links are long random ids that cannot be guessed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from urllib.parse import quote

import requests

from travel_planner.config import Settings
from travel_planner.models import FinalPlan

TABLE = "trip_plans"
MAX_BYTES = 200_000
TIMEOUT_SECONDS = 15
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


class ShareError(Exception):
    def __init__(self, user_message: str, technical: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.technical = technical


def sharing_ready(settings: Settings) -> bool:
    return bool(settings.enable_sharing and settings.supabase_url and settings.supabase_key)


def _base(settings: Settings) -> str:
    url = settings.supabase_url.strip().rstrip("/")
    local = url.startswith(("http://127.0.0.1", "http://localhost"))  # only for tests on this computer
    if not (url.startswith("https://") or local):
        raise ShareError("SUPABASE_URL must start with https://")
    return f"{url}/rest/v1/{TABLE}"


def _headers(settings: Settings) -> dict[str, str]:
    # Secret keys go in the 'apikey' header only (they are not login tokens).
    return {"apikey": settings.supabase_key, "Content-Type": "application/json"}


def _explain(response: requests.Response) -> ShareError:
    text = response.text[:300]
    if response.status_code == 404 or "PGRST205" in text or "Could not find the table" in text:
        return ShareError("The sharing table does not exist yet. In Supabase open SQL Editor, paste the contents of "
                          "supabase_schema.sql and click Run.", technical=text)
    if response.status_code in (401, 403):
        return ShareError("Supabase rejected the key. Use your Supabase SECRET key in SUPABASE_KEY.", technical=text)
    return ShareError(f"Supabase answered with error {response.status_code}.", technical=text)


def create_share(final: FinalPlan, settings: Settings, post: Callable | None = None) -> str:
    """Store the plan and return its share id."""
    post = post or requests.post
    row = {
        "destination": final.request.destination,
        "request": json.loads(final.request.model_dump_json()),
        "plan": json.loads(final.model_dump_json()),
    }
    body = json.dumps(row)
    if len(body.encode("utf-8")) > MAX_BYTES:
        raise ShareError("This plan is too large to share.")
    try:
        response = post(_base(settings), data=body.encode("utf-8"),
                        headers={**_headers(settings), "Prefer": "return=representation"}, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise ShareError("I could not reach Supabase. If your project is paused, resume it in the Supabase dashboard.",
                         technical=str(exc)[:300]) from exc
    if response.status_code >= 400:
        raise _explain(response)
    try:
        share_id = str(response.json()[0]["id"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ShareError("Supabase saved the plan but did not return its id.", technical=response.text[:300]) from exc
    if not UUID_RE.match(share_id):
        raise ShareError("Supabase returned an unexpected id.", technical=share_id[:80])
    return share_id


def load_shared(share_id: str, settings: Settings, get: Callable | None = None) -> FinalPlan:
    """Fetch a shared plan. Raises ShareError with a friendly message if it cannot be opened."""
    get = get or requests.get
    share_id = (share_id or "").strip()
    if not UUID_RE.match(share_id):
        raise ShareError("This share link is not valid.")
    if not sharing_ready(settings):
        raise ShareError("Sharing is not switched on for this app, so this link cannot be opened.")
    try:
        response = get(_base(settings), params={"id": f"eq.{share_id}", "select": "plan"},
                       headers=_headers(settings), timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise ShareError("I could not load the shared plan right now. Please try again in a minute.",
                         technical=str(exc)[:300]) from exc
    if response.status_code >= 400:
        raise _explain(response)
    try:
        rows = response.json()
        if not rows:
            raise ShareError("This shared plan was not found. The link may be wrong or the plan was removed.")
        return FinalPlan.model_validate(rows[0]["plan"])
    except ShareError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ShareError("This shared plan could not be read.", technical=str(exc)[:300]) from exc


def share_url(settings: Settings, share_id: str) -> str:
    base = settings.app_url or ""
    return f"{base}/?plan={share_id}" if base else f"?plan={share_id}"


def summary_text(final: FinalPlan, link: str = "") -> str:
    """A short message that is nice to paste into WhatsApp or an email."""
    r, b = final.request, final.budget
    lines = [f"Trip plan: {r.destination} - {r.num_days} days, {r.travelers} traveller(s)"]
    for day in final.plan.days[:7]:
        lines.append(f"Day {day.day}: {day.title}")
    if len(final.plan.days) > 7:
        lines.append(f"...and {len(final.plan.days) - 7} more days")
    lines.append(f"Estimated total: {b.total:,.0f} {r.currency} (about {b.per_person:,.0f} per person)")
    lines.append("(prices are estimates)")
    if link:
        lines.append(f"Full plan: {link}")
    return "\n".join(lines)[:1400]


def whatsapp_link(text: str) -> str:
    return "https://wa.me/?text=" + quote(text)


def mailto_link(subject: str, body: str) -> str:
    return f"mailto:?subject={quote(subject)}&body={quote(body)}"
