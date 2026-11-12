import pytest

from travel_planner.graph import MAX_DAYS, plan_trip_request
from travel_planner.llm import PlannerError
from tests.fakes import FakeLLM, make_advice, make_plan, make_request


def test_happy_path_and_math():
    llm = FakeLLM()
    steps = []
    final = plan_trip_request(llm, "Plan 3 days in Goa", on_step=steps.append)
    assert [c[0] for c in llm.calls] == ["TripRequest", "TripPlan", "BudgetAdvice"]  # exactly 3 AI calls
    assert steps == ["start", "analyze_request", "plan_trip", "analyze_budget", "finish"]
    assert final.budget.total == 60000.0 and final.budget.per_person == 30000.0
    assert final.budget.within_budget is True and final.budget.difference == 0.0
    assert len(final.plan.days) == 3 and final.models_used == ["fake-model"]


def test_over_budget_flag_and_warning():
    llm = FakeLLM(advice=make_advice(total_each=15000))  # total 90000 vs budget 60000
    final = plan_trip_request(llm, "x")
    assert final.budget.within_budget is False and final.budget.difference == -30000.0
    assert any("over your budget" in w for w in final.warnings)


def test_no_budget_given():
    llm = FakeLLM(request=make_request(budget_total=0))
    final = plan_trip_request(llm, "x")
    assert final.budget.within_budget is None


def test_form_options_override_ai_guesses():
    llm = FakeLLM()
    opts = {"origin": "Mumbai", "travelers": 4, "budget_total": 100000, "currency": "usd",
            "travel_style": "Luxury", "interests": ["hiking"]}
    final = plan_trip_request(llm, "x", opts)
    r = final.request
    assert (r.origin, r.travelers, r.budget_total, r.currency, r.travel_style) == ("Mumbai", 4, 100000.0, "USD", "luxury")
    assert "hiking" in r.interests and "food" in r.interests
    assert final.budget.per_person == 15000.0  # 60000 / 4


def test_dates_set_num_days_and_day_dates():
    llm = FakeLLM(request=make_request(start_date="2026-12-01", end_date="2026-12-04", num_days=9),
                  plan=make_plan(4))
    final = plan_trip_request(llm, "x")
    assert final.request.num_days == 4
    assert [d.date for d in final.plan.days] == ["2026-12-01", "2026-12-02", "2026-12-03", "2026-12-04"]


def test_reversed_dates_ignored():
    llm = FakeLLM(request=make_request(start_date="2026-12-10", end_date="2026-12-01", num_days=3))
    final = plan_trip_request(llm, "x")
    assert final.request.start_date == "" and any("end date was before" in w for w in final.warnings)


def test_day_count_mismatch_warns():
    llm = FakeLLM(request=make_request(num_days=5), plan=make_plan(3))
    final = plan_trip_request(llm, "x")
    assert any("3 day(s) but your trip is 5" in w for w in final.warnings)


def test_long_trip_is_capped():
    llm = FakeLLM(request=make_request(num_days=40), plan=make_plan(MAX_DAYS))
    final = plan_trip_request(llm, "x")
    assert final.request.num_days == MAX_DAYS and any("limited" in w for w in final.warnings)


def test_missing_destination_is_a_friendly_error():
    llm = FakeLLM(request=make_request(destination="  "))
    with pytest.raises(PlannerError) as info:
        plan_trip_request(llm, "hello")
    assert "destination" in info.value.user_message


def test_empty_input():
    with pytest.raises(PlannerError):
        plan_trip_request(FakeLLM(), "   ")


def test_bad_values_are_cleaned():
    llm = FakeLLM(request=make_request(travelers=0, num_days=0, currency="", travel_style="fancy"))
    r = plan_trip_request(llm, "x").request
    assert r.travelers == 1 and r.num_days >= 1 and r.currency == "USD" and r.travel_style == "moderate"


def test_blank_list_entries_are_removed():
    llm = FakeLLM(request=make_request(assumptions=["", "  ", "Real assumption"], interests=["food", ""]))
    r = plan_trip_request(llm, "x").request
    assert r.assumptions == ["Real assumption"] and r.interests == ["food"]
