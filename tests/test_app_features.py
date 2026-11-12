"""Page-level tests for live prices and share links."""

import uuid
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import travel_planner.config as config_module
import travel_planner.llm as llm_module
import travel_planner.storage as storage_module
from travel_planner import prices, sharing, usage
from travel_planner.graph import plan_trip_request
from travel_planner.models import LiveFlight, LiveHotel, LivePrices
from tests.fakes import FakeLLM

APP_FILE = str(Path(__file__).resolve().parent.parent / "app.py")
SHARE_ID = str(uuid.uuid4())


def live_sample():
    return LivePrices(currency="INR", depart_date="2027-01-10", return_date="2027-01-13",
                      notes=["Example dates note"],
                      flights=[LiveFlight(airline="IndiGo", price_per_person=5000, stops=0, duration_minutes=75,
                                          departure="2027-01-10 08:25", link="https://www.google.com/travel/flights")],
                      hotels=[LiveHotel(name="Mid Hotel", price_per_night=2000, stars=3, rating=4.2, total_price=6000,
                                        link="https://example.com/h")])


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_module, "SAVE_DIR", tmp_path)
    monkeypatch.setattr(usage, "COUNTER", usage.DailyCounter())
    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(config_module, "_secret", lambda name: None)
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test")
    for name in ("PUBLIC_MODE", "DAILY_TRIP_LIMIT", "SESSION_TRIP_LIMIT", "SERPAPI_API_KEY", "LIVE_PRICES",
                 "ENABLE_SHARING", "SUPABASE_URL", "SUPABASE_KEY", "APP_URL", "SAVE_TO_SUPABASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm_module, "build_llm", lambda settings: FakeLLM())


def start():
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    return at


def plan(at):
    at.text_area[0].set_value("3 days in Goa from Pune").run()
    next(b for b in at.button if "Plan my trip" in b.label).click().run()


# ---- live prices --------------------------------------------------------------------------------

def test_no_price_checkbox_without_a_key():
    at = start()
    assert not any("live flight" in c.label for c in at.checkbox)


def test_live_prices_flow(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
    seen = {}

    def fake_make_fetcher(api_key, daily_limit=0, **kw):
        seen["key"], seen["limit"] = api_key, daily_limit
        return lambda request: live_sample()

    monkeypatch.setattr(prices, "make_fetcher", fake_make_fetcher)
    at = start()
    assert any("live flight" in c.label for c in at.checkbox)
    plan(at)
    assert not at.exception, [e.value for e in at.exception]
    assert seen["key"] == "serp-key"
    assert any("Live prices from Google Flights" in s.value for s in at.success)
    assert any("Example dates note" in i.value for i in at.info)
    assert any("live prices" in c.value.lower() for c in at.caption)           # budget tab note


def test_unchecking_live_prices_skips_the_lookup(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
    monkeypatch.setattr(prices, "make_fetcher", lambda *a, **k: pytest.fail("must not be called"))
    at = start()
    next(c for c in at.checkbox if "live flight" in c.label).uncheck().run()
    plan(at)
    assert not at.exception and any("Your trip to" in h.value for h in at.header)


def test_failed_price_lookup_shows_warning_but_plan_still_appears(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")

    def failing(request):
        raise prices.PriceError("quota gone")

    monkeypatch.setattr(prices, "make_fetcher", lambda *a, **k: failing)
    at = start()
    plan(at)
    assert any("quota gone" in w.value for w in at.warning)
    assert any("Your trip to" in h.value for h in at.header)


# ---- sharing --------------------------------------------------------------------------------------

def enable_sharing(monkeypatch):
    monkeypatch.setenv("ENABLE_SHARING", "true")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "sb_secret_x")
    monkeypatch.setenv("APP_URL", "https://demo.streamlit.app/")


def test_create_share_link(monkeypatch):
    enable_sharing(monkeypatch)
    monkeypatch.setattr(sharing, "create_share", lambda final, settings: SHARE_ID)
    at = start()
    plan(at)
    next(b for b in at.button if "shareable link" in b.label).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(c.value == f"https://demo.streamlit.app/?plan={SHARE_ID}" for c in at.code)


def test_share_limit_per_visit(monkeypatch):
    enable_sharing(monkeypatch)
    monkeypatch.setenv("SHARE_SESSION_LIMIT", "1")
    monkeypatch.setattr(sharing, "create_share", lambda final, settings: SHARE_ID)
    calls = []
    monkeypatch.setattr(sharing, "create_share", lambda final, settings: calls.append(1) or SHARE_ID)
    at = start()
    plan(at)
    next(b for b in at.button if "shareable link" in b.label).click().run()
    at.run()  # the page refreshes on the next interaction
    assert next(b for b in at.button if "shareable link" in b.label).disabled
    assert any("share limit" in c.value for c in at.caption)
    assert len(calls) == 1


def test_share_error_is_friendly(monkeypatch):
    enable_sharing(monkeypatch)

    def boom(final, settings):
        raise sharing.ShareError("The sharing table does not exist yet.", technical="PGRST205")

    monkeypatch.setattr(sharing, "create_share", boom)
    at = start()
    plan(at)
    next(b for b in at.button if "shareable link" in b.label).click().run()
    assert not at.exception and any("sharing table" in e.value for e in at.error)


def test_no_share_button_when_not_configured():
    at = start()
    plan(at)
    assert not any("shareable link" in b.label for b in at.button)   # WhatsApp / email links still exist


def test_open_a_shared_link(monkeypatch):
    enable_sharing(monkeypatch)
    shared = plan_trip_request(FakeLLM(), "x")
    monkeypatch.setattr(sharing, "load_shared", lambda sid, settings: shared if sid == SHARE_ID else pytest.fail(sid))
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.query_params["plan"] = SHARE_ID
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Your trip to Goa" in h.value for h in at.header)
    assert any("shared with you" in i.value for i in at.info)


def test_broken_shared_link_shows_error(monkeypatch):
    enable_sharing(monkeypatch)

    def missing(sid, settings):
        raise sharing.ShareError("This shared plan was not found.")

    monkeypatch.setattr(sharing, "load_shared", missing)
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.query_params["plan"] = SHARE_ID
    at.run()
    assert not at.exception and any("not found" in e.value for e in at.error)
    assert any("Describe your trip" in t.label for t in at.text_area)    # the normal page still works


# ---- full wiring over real HTTP (fake SerpApi + fake Supabase servers, nothing of ours is mocked) ----

def _serve(handler):
    import threading
    from http.server import HTTPServer
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_end_to_end_over_http(monkeypatch):
    from tests.test_prices import FakeSerp
    from tests.test_sharing import FakeSupabase
    FakeSerp.hits, FakeSerp.mode = [], "ok"
    FakeSupabase.rows, FakeSupabase.headers_seen, FakeSupabase.mode = {}, [], "ok"
    serp, supa = _serve(FakeSerp), _serve(FakeSupabase)
    try:
        monkeypatch.setenv("SERPAPI_BASE_URL", f"http://127.0.0.1:{serp.server_port}/search.json")
        monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
        monkeypatch.setenv("PUBLIC_MODE", "true")
        monkeypatch.setenv("LIVE_PRICES_DAILY_LIMIT", "5")
        monkeypatch.setattr(usage, "PRICE_COUNTER", usage.DailyCounter())
        monkeypatch.setattr(prices, "CACHE", prices.PriceCache())
        enable_sharing(monkeypatch)
        monkeypatch.setenv("SUPABASE_URL", f"http://127.0.0.1:{supa.server_port}")

        at = start()
        plan(at)
        assert not at.exception, [e.value for e in at.exception]
        assert any("Live prices from Google Flights" in s.value for s in at.success)     # real fetch + parse worked
        assert any(h["engine"] == "google_flights" and h["api_key"] == "serp-key" for h in FakeSerp.hits)
        assert any("example dates" in i.value for i in at.info)                          # no dates given -> labelled

        next(b for b in at.button if "shareable link" in b.label).click().run()
        link = next(c.value for c in at.code if "?plan=" in c.value)
        share_id = link.split("?plan=")[1]
        assert share_id in FakeSupabase.rows

        friend = AppTest.from_file(APP_FILE, default_timeout=30)                          # someone opens the link
        friend.query_params["plan"] = share_id
        friend.run()
        assert not friend.exception, [e.value for e in friend.exception]
        assert any("Your trip to Goa" in h.value for h in friend.header)
        assert any("shared with you" in i.value for i in friend.info)
        assert any("Live prices from Google Flights" in s.value for s in friend.success)  # live prices travel with the plan
    finally:
        serp.shutdown()
        supa.shutdown()
