"""Data shapes used by the planner.

The "LLM" models below are what we ask Gemini to fill in. They deliberately
have no default values, because Gemini's structured-output feature handles
fully-specified schemas most reliably.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------
# Step 1: understanding the request
# --------------------------------------------------------------------------


class TripRequest(BaseModel):
    destination: str = Field(description="Main destination, e.g. 'Paris, France'. Empty string if none given.")
    origin: str = Field(description="Where the traveller starts from. Empty string if unknown.")
    start_date: str = Field(description="Start date as YYYY-MM-DD, or empty string if unknown.")
    end_date: str = Field(description="End date as YYYY-MM-DD, or empty string if unknown.")
    num_days: int = Field(description="Trip length in days (whole number, at least 1).")
    travelers: int = Field(description="Number of travellers (whole number, at least 1).")
    budget_total: float = Field(description="Total budget for the whole group as a plain number, or 0 if not given.")
    currency: str = Field(description="3-letter currency code such as USD, EUR, INR.")
    travel_style: str = Field(description="One of: budget, moderate, luxury.")
    interests: list[str] = Field(description="Interests such as food, history, beaches, hiking.")
    constraints: list[str] = Field(description="Special needs: kids, wheelchair access, vegetarian, etc.")
    assumptions: list[str] = Field(description="Any assumptions you had to make because info was missing.")
    origin_iata: str = Field(description="3-letter IATA code of the main departure airport (e.g. PNQ), or empty string if unknown.")
    destination_iata: str = Field(description="3-letter IATA code of the main airport for the destination (e.g. GOI), or empty string if unknown.")

    @model_validator(mode="before")
    @classmethod
    def _old_saved_plans(cls, data):
        """Plans saved before these two fields existed must still load."""
        if isinstance(data, dict):
            data.setdefault("origin_iata", "")
            data.setdefault("destination_iata", "")
        return data


# --------------------------------------------------------------------------
# Step 2: the plan itself
# --------------------------------------------------------------------------


class DestinationInfo(BaseModel):
    summary: str = Field(description="2-4 sentence overview of the destination.")
    best_time_to_visit: str
    weather_during_trip: str = Field(description="What weather to expect during the trip dates.")
    entry_requirements: str = Field(description="Visa / passport / entry notes. Say to verify with official sources.")
    local_tips: list[str]
    cultural_notes: list[str]
    safety_and_health: list[str]


class FlightOption(BaseModel):
    route: str = Field(description="e.g. 'Mumbai (BOM) -> Paris (CDG)'")
    typical_airlines: str
    typical_duration: str
    estimated_price_per_person: float = Field(description="Round-trip estimate per person, plain number in the trip currency.")
    notes: str


class StayOption(BaseModel):
    name_or_area: str = Field(description="A neighbourhood or type of place. Do not invent specific hotel prices as facts.")
    category: str = Field(description="hostel / budget hotel / mid-range hotel / luxury / apartment")
    estimated_price_per_night: float = Field(description="Estimated cost per night for the whole group's room(s), plain number.")
    why_recommended: str


class TransportOption(BaseModel):
    mode: str = Field(description="e.g. metro, taxi, rental car, train")
    details: str
    estimated_cost: float = Field(description="Estimated total cost for the whole trip for the whole group, plain number.")


class DayPlan(BaseModel):
    day: int
    date: str = Field(description="YYYY-MM-DD if dates are known, otherwise empty string.")
    title: str
    morning: str
    afternoon: str
    evening: str
    food_suggestion: str
    estimated_cost: float = Field(description="Estimated spend that day (activities + food) for the whole group, plain number.")


class TripPlan(BaseModel):
    destination_info: DestinationInfo
    flights: list[FlightOption]
    stays: list[StayOption]
    local_transport: list[TransportOption]
    days: list[DayPlan]
    packing_tips: list[str]


# --------------------------------------------------------------------------
# Step 3: budget
# --------------------------------------------------------------------------


class BudgetItem(BaseModel):
    category: str = Field(description="Flights, Accommodation, Local transport, Food, Activities, Misc")
    amount: float = Field(description="Estimated amount for the whole group, plain number.")
    notes: str


class BudgetAdvice(BaseModel):
    items: list[BudgetItem]
    summary: str = Field(description="2-3 sentences: is the plan realistic for the budget?")
    money_saving_tips: list[str]


# --------------------------------------------------------------------------
# Final result (assembled by our own code, not by Gemini)
# --------------------------------------------------------------------------


class BudgetSummary(BaseModel):
    items: list[BudgetItem]
    total: float
    per_person: float
    budget_total: float
    within_budget: bool | None = None  # None when the user gave no budget
    difference: float = 0.0  # budget_total - total (negative = over budget)
    summary: str = ""
    money_saving_tips: list[str] = Field(default_factory=list)


class LiveFlight(BaseModel):
    airline: str
    price_per_person: float
    duration_minutes: int = 0
    stops: int = 0
    departure: str = ""
    arrival: str = ""
    link: str = ""


class LiveHotel(BaseModel):
    name: str
    price_per_night: float
    total_price: float = 0.0
    rating: float = 0.0
    stars: int = 0
    link: str = ""


class LivePrices(BaseModel):
    """Real prices looked up on Google Flights / Google Hotels (via SerpApi)."""

    source: str = "Google Flights and Google Hotels (via SerpApi)"
    currency: str = "USD"
    depart_date: str = ""
    return_date: str = ""
    sample_dates: bool = False  # True when the user gave no dates and we picked example dates
    flights: list[LiveFlight] = Field(default_factory=list)
    hotels: list[LiveHotel] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    fetched_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class FinalPlan(BaseModel):
    request: TripRequest
    plan: TripPlan
    budget: BudgetSummary
    live_prices: LivePrices | None = None
    warnings: list[str] = Field(default_factory=list)
    models_used: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
