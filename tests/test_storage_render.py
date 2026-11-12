from travel_planner.config import Settings, get_settings
from travel_planner.graph import plan_trip_request
from travel_planner.render import plan_to_markdown
from travel_planner.storage import list_saved_plans, load_plan, save_plan_local, save_plan_supabase
from tests.fakes import FakeLLM


def final():
    return plan_trip_request(FakeLLM(), "x")


def test_markdown_has_all_sections():
    md = plan_to_markdown(final())
    for part in ["# Trip to Goa", "Day-by-day itinerary", "### Day 3", "Flights", "Where to stay", "Getting around",
                 "## Budget", "**Total**", "AI estimates"]:
        assert part in md


def test_save_and_load_roundtrip(tmp_path):
    f = final()
    path = save_plan_local(f, tmp_path)
    assert path.exists() and list_saved_plans(tmp_path) == [path]
    assert load_plan(path) == f


class FakeTable:
    def __init__(self, sink): self.sink = sink
    def insert(self, row): self.sink.append(row); return self
    def execute(self): return self


class FakeSupabase:
    def __init__(self): self.rows = []
    def table(self, name): self.name = name; return FakeTable(self.rows)


def test_supabase_ok_and_failure_never_raises():
    fake = FakeSupabase()
    msg = save_plan_supabase(final(), Settings(supabase_url="u", supabase_key="k"), client=fake)
    assert msg == "Saved to Supabase." and fake.name == "trip_plans" and fake.rows[0]["destination"] == "Goa, India"

    class Broken:
        def table(self, name): raise RuntimeError("db down")
    assert "Could not save" in save_plan_supabase(final(), Settings(supabase_url="u", supabase_key="k"), client=Broken())
    assert "not configured" in save_plan_supabase(final(), Settings())


def test_settings_cleaning(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", '  "AQ.abc"  ')
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv("DEFAULT_CURRENCY", "inr")
    s = get_settings()
    assert s.api_key == "AQ.abc" and s.default_currency == "INR"
    assert s.model_chain[0] == "gemini-3.6-flash" and "gemini-3.5-flash-lite" in s.model_chain
    assert len(s.model_chain) == len(set(s.model_chain))
    assert get_settings(model_override="gemini-3.8-flash").model_chain[0] == "gemini-3.8-flash"
