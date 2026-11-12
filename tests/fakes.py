"""Shared fake data and a fake LLM for tests (no internet needed)."""

from __future__ import annotations

from travel_planner.models import (
    BudgetAdvice,
    BudgetItem,
    DayPlan,
    DestinationInfo,
    FlightOption,
    StayOption,
    TransportOption,
    TripPlan,
    TripRequest,
)


def make_request(**overrides) -> TripRequest:
    data = dict(
        destination="Goa, India", origin="Pune", start_date="", end_date="", num_days=3, travelers=2,
        budget_total=60000.0, currency="INR", travel_style="moderate", interests=["food", "beaches"],
        constraints=[], assumptions=["Assumed travel by train"],
    )
    data.update(overrides)
    return TripRequest(**data)


def make_plan(days: int = 3) -> TripPlan:
    return TripPlan(
        destination_info=DestinationInfo(
            summary="Sunny beaches.", best_time_to_visit="Nov-Feb", weather_during_trip="Warm",
            entry_requirements="None for Indian citizens; verify officially.",
            local_tips=["Rent a scooter"], cultural_notes=["Dress modestly in churches"],
            safety_and_health=["Drink bottled water"],
        ),
        flights=[FlightOption(route="Pune -> Goa", typical_airlines="IndiGo", typical_duration="1h 10m",
                              estimated_price_per_person=6000, notes="Book early")],
        stays=[StayOption(name_or_area="Calangute", category="mid-range hotel", estimated_price_per_night=4000,
                          why_recommended="Central")],
        local_transport=[TransportOption(mode="Scooter", details="Rent daily", estimated_cost=3000)],
        days=[DayPlan(day=i, date="", title=f"Day {i} title", morning="Beach", afternoon="Fort", evening="Dinner",
                      food_suggestion="Fish curry", estimated_cost=4000) for i in range(1, days + 1)],
        packing_tips=["Sunscreen"],
    )


def make_advice(total_each: float = 10000.0) -> BudgetAdvice:
    cats = ["Flights", "Accommodation", "Local transport", "Food", "Activities", "Miscellaneous"]
    return BudgetAdvice(
        items=[BudgetItem(category=c, amount=total_each, notes="est") for c in cats],
        summary="Looks realistic.", money_saving_tips=["Travel off-season"],
    )


class FakeLLM:
    """Returns canned objects based on which schema was requested."""

    def __init__(self, request=None, plan=None, advice=None):
        self.request = request or make_request()
        self.plan = plan or make_plan(self.request.num_days)
        self.advice = advice or make_advice()
        self.models_used = ["fake-model"]
        self.calls: list[tuple[str, str]] = []

    def generate(self, schema, system, prompt, temperature=0.6):
        self.calls.append((schema.__name__, prompt))
        if schema is TripRequest:
            return self.request.model_copy(deep=True)
        if schema is TripPlan:
            return self.plan.model_copy(deep=True)
        return self.advice.model_copy(deep=True)
