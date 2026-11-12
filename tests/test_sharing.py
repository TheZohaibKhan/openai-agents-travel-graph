import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from travel_planner import sharing
from travel_planner.config import Settings
from travel_planner.graph import plan_trip_request
from travel_planner.sharing import ShareError, create_share, load_shared
from tests.fakes import FakeLLM


class FakeSupabase(BaseHTTPRequestHandler):
    rows: dict = {}
    headers_seen: list = []
    mode = "ok"

    def log_message(self, *a):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        FakeSupabase.headers_seen.append(dict(self.headers))
        row = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if FakeSupabase.mode == "notable":
            return self._send(404, {"code": "PGRST205", "message": "Could not find the table 'public.trip_plans'"})
        if FakeSupabase.mode == "badkey":
            return self._send(401, {"message": "Invalid API key"})
        new_id = str(uuid.uuid4())
        FakeSupabase.rows[new_id] = row
        self._send(201, [{"id": new_id, **row}])

    def do_GET(self):
        FakeSupabase.headers_seen.append(dict(self.headers))
        q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        rid = q["id"].removeprefix("eq.")
        row = FakeSupabase.rows.get(rid)
        self._send(200, [{"plan": row["plan"]}] if row else [])


@pytest.fixture()
def supa():
    FakeSupabase.rows, FakeSupabase.headers_seen, FakeSupabase.mode = {}, [], "ok"
    httpd = HTTPServer(("127.0.0.1", 0), FakeSupabase)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    settings = Settings(enable_sharing=True, supabase_url=f"http://127.0.0.1:{httpd.server_port}",
                        supabase_key="sb_secret_test", app_url="https://demo.streamlit.app")
    yield settings
    httpd.shutdown()


def test_share_roundtrip(supa):
    final = plan_trip_request(FakeLLM(), "x")
    share_id = create_share(final, supa)
    assert sharing.UUID_RE.match(share_id)
    assert load_shared(share_id, supa) == final
    sent = FakeSupabase.headers_seen[0]
    assert sent["apikey"] == "sb_secret_test" and "Authorization" not in sent   # secret key goes in apikey only
    assert sent["Prefer"] == "return=representation"
    assert sharing.share_url(supa, share_id) == f"https://demo.streamlit.app/?plan={share_id}"


def test_unknown_or_bad_ids(supa):
    with pytest.raises(ShareError) as info:
        load_shared(str(uuid.uuid4()), supa)
    assert "not found" in info.value.user_message
    n = len(FakeSupabase.headers_seen)
    for bad in ["", "abc", "1; drop table", "eq.x&select=*", "../../etc"]:
        with pytest.raises(ShareError):
            load_shared(bad, supa)
    assert len(FakeSupabase.headers_seen) == n          # junk ids never reach the database


def test_friendly_database_errors(supa):
    final = plan_trip_request(FakeLLM(), "x")
    FakeSupabase.mode = "notable"
    with pytest.raises(ShareError) as info:
        create_share(final, supa)
    assert "supabase_schema.sql" in info.value.user_message
    FakeSupabase.mode = "badkey"
    with pytest.raises(ShareError) as info:
        create_share(final, supa)
    assert "SECRET key" in info.value.user_message


def test_unreachable_supabase():
    s = Settings(enable_sharing=True, supabase_url="http://127.0.0.1:9", supabase_key="k")
    with pytest.raises(ShareError) as info:
        create_share(plan_trip_request(FakeLLM(), "x"), s)
    assert "paused" in info.value.user_message


def test_not_enabled_and_size_limit(supa, monkeypatch):
    assert not sharing.sharing_ready(Settings())
    with pytest.raises(ShareError):
        load_shared(str(uuid.uuid4()), Settings())
    monkeypatch.setattr(sharing, "MAX_BYTES", 100)
    with pytest.raises(ShareError) as info:
        create_share(plan_trip_request(FakeLLM(), "x"), supa)
    assert "too large" in info.value.user_message


def test_summary_and_links():
    final = plan_trip_request(FakeLLM(), "x")
    text = sharing.summary_text(final, "https://demo.streamlit.app/?plan=abc")
    assert "Goa, India" in text and "Day 1: Day 1 title" in text and text.endswith("plan=abc") and len(text) < 1400
    wa = sharing.whatsapp_link(text)
    assert wa.startswith("https://wa.me/?text=") and "\n" not in wa and " " not in wa
    assert sharing.mailto_link("Trip", text).startswith("mailto:?subject=Trip&body=")
