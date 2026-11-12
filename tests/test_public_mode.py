from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import travel_planner.config as config_module
import travel_planner.llm as llm_module
import travel_planner.storage as storage_module
from travel_planner import usage
from travel_planner.config import get_settings
from tests.fakes import FakeLLM

APP_FILE = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_module, "SAVE_DIR", tmp_path)
    monkeypatch.setattr(usage, "COUNTER", usage.DailyCounter())
    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **k: None)  # ignore any real .env
    monkeypatch.setattr(config_module, "_secret", lambda name: None)
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test-key")
    for name in ("PUBLIC_MODE", "DAILY_TRIP_LIMIT", "SESSION_TRIP_LIMIT", "SAVE_TO_SUPABASE"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def start(monkeypatch, fake=None):
    monkeypatch.setattr(llm_module, "build_llm", lambda settings: fake or FakeLLM())
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    return at


def plan(at, text="3 days in Goa"):
    at.text_area[0].set_value(text).run()
    next(b for b in at.button if "Plan my trip" in b.label).click().run()


def test_defaults_are_unlimited_for_local_use():
    s = get_settings()
    assert (s.public_mode, s.daily_trip_limit, s.session_trip_limit) == (False, 0, 0)


def test_public_mode_gets_safe_defaults(monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "true")
    s = get_settings()
    assert s.public_mode and s.daily_trip_limit == 40 and s.session_trip_limit == 5
    monkeypatch.setenv("DAILY_TRIP_LIMIT", "7")
    monkeypatch.setenv("SESSION_TRIP_LIMIT", "oops")  # junk falls back to the default
    s = get_settings()
    assert s.daily_trip_limit == 7 and s.session_trip_limit == 5


def test_streamlit_secrets_are_used_as_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setattr(config_module, "_secret", lambda name: "AQ.from-secrets" if name == "GEMINI_API_KEY" else None)
    assert get_settings().api_key == "AQ.from-secrets"


def test_public_mode_stores_nothing_and_hides_saved_trips(monkeypatch, isolate):
    monkeypatch.setenv("PUBLIC_MODE", "true")
    at = start(monkeypatch)
    plan(at)
    assert not at.exception and not at.error
    assert any("Your trip to" in h.value for h in at.header)  # the plan still shows
    assert list(isolate.glob("*.json")) == []                   # nothing saved on the server
    assert not any("Saved trips" in m.value for m in at.sidebar.markdown)
    assert not any("API key" in c.value for c in at.sidebar.caption)


def test_session_limit(monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "true")
    monkeypatch.setenv("SESSION_TRIP_LIMIT", "1")
    fake = FakeLLM()
    at = start(monkeypatch, fake)
    plan(at)
    assert len(fake.calls) == 3
    plan(at)  # second trip in the same visit
    assert len(fake.calls) == 3  # no new AI calls
    assert any("limit of 1" in w.value for w in at.warning)


def test_daily_limit_is_shared_by_everyone(monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "true")
    monkeypatch.setenv("DAILY_TRIP_LIMIT", "1")
    first = start(monkeypatch)
    plan(first)
    assert any("Your trip to" in h.value for h in first.header)
    second_fake = FakeLLM()
    second = start(monkeypatch, second_fake)  # a different visitor
    plan(second)
    assert second_fake.calls == [] and any("quota for today" in w.value for w in second.warning)


def test_failed_trip_does_not_use_up_a_try(monkeypatch):
    from travel_planner.llm import BusyError

    class Broken(FakeLLM):
        def generate(self, *a, **k):
            raise BusyError("busy")

    monkeypatch.setenv("PUBLIC_MODE", "true")
    monkeypatch.setenv("SESSION_TRIP_LIMIT", "1")
    at = start(monkeypatch, Broken())
    plan(at)
    assert any("busy" in e.value for e in at.error)
    monkeypatch.setattr(llm_module, "build_llm", lambda settings: FakeLLM())
    plan(at)  # the retry is still allowed
    assert any("Your trip to" in h.value for h in at.header)


def test_daily_counter_unit():
    c = usage.DailyCounter()
    assert [c.try_take(2, "d1") for _ in range(3)] == [True, True, False]
    assert c.try_take(2, "d2") is True            # a new day resets the count
    assert c.try_take(0, "d2") is True            # 0 = unlimited
    c.give_back()
    assert c.try_take(2, "d2") is True
