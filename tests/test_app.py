"""Drives the real Streamlit page (no browser needed) with a fake AI."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import travel_planner.llm as llm_module
import travel_planner.storage as storage_module
from tests.fakes import FakeLLM, make_request


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_module, "SAVE_DIR", tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test-key")
    monkeypatch.setenv("SAVE_TO_SUPABASE", "false")
    return tmp_path


APP_FILE = str(Path(__file__).resolve().parent.parent / "app.py")


def run_app():
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    return at


def plan_button(at):
    return next(b for b in at.button if "Plan my trip" in b.label)


def test_page_loads_without_errors():
    at = run_app()
    assert not at.exception
    assert any("AI Travel Planner" in m.value for m in at.markdown)  # hero banner
    assert len(at.text_area) == 1


def test_example_buttons_fill_the_text_box():
    at = run_app()
    examples = [b for b in at.button if "Plan my trip" not in b.label]
    assert len(examples) >= 3
    examples[1].click().run()
    assert not at.exception
    assert "Tokyo" in at.text_area[0].value


def test_empty_query_warns():
    at = run_app()
    plan_button(at).click().run()
    assert any("describe your trip" in w.value.lower() for w in at.warning)


def test_full_trip_shows_result_and_saves(monkeypatch, isolate):
    fake = FakeLLM()
    monkeypatch.setattr(llm_module, "build_llm", lambda settings: fake)
    at = run_app()
    at.text_area[0].set_value("3 days in Goa for 2").run()
    plan_button(at).click().run()

    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    assert [h.value for h in at.header if "Your trip to" in h.value] == ["Your trip to Goa, India"]
    assert len(at.tabs) == 5
    assert len(at.expander) >= 1
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Days"] == "3" and metrics["Travellers"] == "2" and "60,000 INR" in metrics["Estimated total"]
    assert len(list(isolate.glob("*.json"))) == 1  # plan was auto-saved


def test_over_budget_shows_warning(monkeypatch):
    from tests.fakes import make_advice
    fake = FakeLLM(advice=make_advice(15000))
    monkeypatch.setattr(llm_module, "build_llm", lambda settings: fake)
    at = run_app()
    at.text_area[0].set_value("x trip").run()
    plan_button(at).click().run()
    assert any("over your budget" in w.value for w in at.warning)


def test_friendly_error_when_key_missing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setattr("travel_planner.config.load_dotenv", lambda *a, **k: None)  # ignore any real .env
    at = run_app()
    at.text_area[0].set_value("3 days in Goa").run()
    plan_button(at).click().run()
    assert not at.exception
    assert any("GEMINI_API_KEY" in e.value for e in at.error)


def test_friendly_error_when_destination_missing(monkeypatch):
    fake = FakeLLM(request=make_request(destination=""))
    monkeypatch.setattr(llm_module, "build_llm", lambda settings: fake)
    at = run_app()
    at.text_area[0].set_value("hello there").run()
    plan_button(at).click().run()
    assert not at.exception
    assert any("could not find a destination" in e.value for e in at.error)


def test_layout_survives_awkward_plans(monkeypatch):
    """Empty sections, many items, no budget, 14 days: the page must not crash."""
    from tests.fakes import make_plan
    plan = make_plan(14)
    plan.flights = []
    plan.stays = plan.stays * 5
    plan.local_transport = []
    plan.packing_tips = ["", "  "]
    plan.destination_info.local_tips = []
    fake = FakeLLM(request=make_request(num_days=14, budget_total=0, assumptions=[], interests=[]), plan=plan)
    monkeypatch.setattr(llm_module, "build_llm", lambda settings: fake)
    at = run_app()
    at.text_area[0].set_value("two weeks somewhere").run()
    plan_button(at).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert {m.label for m in at.metric} >= {"Days", "Per person"}
