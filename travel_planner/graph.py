"""The LangGraph workflow.

   analyze_request  ->  plan_trip  ->  analyze_budget  ->  finish

Each step is one call to Gemini. All maths (totals, per-person cost, over/under
budget) is done here in normal Python, not by the AI.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from travel_planner.llm import PlannerError
from travel_planner.models import (
    BudgetAdvice,
    BudgetSummary,
    FinalPlan,
    TripPlan,
    TripRequest,
)
from travel_planner.prompts import ANALYZE_SYSTEM, BUDGET_SYSTEM, PLAN_SYSTEM

MAX_DAYS = 14

STEP_LABELS = {
    "analyze_request": "Understanding your request",
    "plan_trip": "Planning destination, flights, stays and daily itinerary",
    "analyze_budget": "Working out the budget",
    "finish": "Putting the plan together",
}


class LLM(Protocol):
    models_used: list[str]

    def generate(self, schema: type, system: str, prompt: str, temperature: float = 0.6) -> Any: ...


class PlanState(TypedDict, total=False):
    user_request: str
    options: dict[str, Any]
    request: TripRequest
    plan: TripPlan
    advice: BudgetAdvice
    warnings: list[str]
    final: FinalPlan


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat((value or "").strip())
    except ValueError:
        return None


def _options_text(options: dict[str, Any]) -> str:
    """Turn the optional form fields into text for the prompt."""
    lines = []
    labels = {
        "origin": "Starting from",
        "travelers": "Number of travellers",
        "budget_total": "Total budget",
        "currency": "Currency",
        "start_date": "Start date",
        "end_date": "End date",
        "travel_style": "Travel style",
        "interests": "Interests",
    }
    for key, label in labels.items():
        value = options.get(key)
        if value in (None, "", 0, []):
            continue
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def _apply_form_overrides(request: TripRequest, options: dict[str, Any]) -> TripRequest:
    """Anything the user explicitly chose in the form beats the AI's guess."""
    data = request.model_dump()
    if options.get("origin"):
        data["origin"] = str(options["origin"]).strip()
    if options.get("travelers"):
        data["travelers"] = int(options["travelers"])
    if options.get("budget_total"):
        data["budget_total"] = float(options["budget_total"])
    if options.get("currency"):
        data["currency"] = str(options["currency"]).strip().upper()[:3]
    if options.get("start_date"):
        data["start_date"] = str(options["start_date"])
    if options.get("end_date"):
        data["end_date"] = str(options["end_date"])
    if options.get("travel_style"):
        data["travel_style"] = str(options["travel_style"]).lower()
    if options.get("interests"):
        merged = list(dict.fromkeys([*data["interests"], *options["interests"]]))
        data["interests"] = merged
    return TripRequest(**data)


def _clean_request(request: TripRequest, warnings: list[str]) -> TripRequest:
    """Make the numbers sane."""
    data = request.model_dump()

    start, end = _parse_date(data["start_date"]), _parse_date(data["end_date"])
    if start and end and end >= start:
        data["num_days"] = (end - start).days + 1
    elif start and end:
        warnings.append("The end date was before the start date, so I ignored the dates.")
        start = end = None
    # Keep only dates we could actually read.
    data["start_date"] = start.isoformat() if start else ""
    data["end_date"] = end.isoformat() if end else ""

    data["num_days"] = max(1, int(data["num_days"] or 5))
    if data["num_days"] > MAX_DAYS:
        warnings.append(f"Trips are limited to {MAX_DAYS} days here, so I planned the first {MAX_DAYS} days.")
        data["num_days"] = MAX_DAYS
        data["end_date"] = ""

    for key in ("interests", "constraints", "assumptions"):
        data[key] = [str(x).strip() for x in data[key] if str(x).strip()]

    data["travelers"] = max(1, int(data["travelers"] or 1))
    data["budget_total"] = max(0.0, float(data["budget_total"] or 0))
    data["currency"] = (data["currency"] or "USD").strip().upper()[:3] or "USD"
    if data["travel_style"] not in {"budget", "moderate", "luxury"}:
        data["travel_style"] = "moderate"
    return TripRequest(**data)


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------


def build_graph(llm: LLM):
    def analyze_request(state: PlanState) -> dict[str, Any]:
        options = state.get("options", {})
        warnings = list(state.get("warnings", []))

        extra = _options_text(options)
        prompt = f"Traveller's message:\n{state['user_request']}\n"
        if extra:
            prompt += f"\nDetails the traveller selected in the form (treat these as fixed):\n{extra}\n"

        request = llm.generate(
            TripRequest,
            ANALYZE_SYSTEM.format(today=date.today().isoformat()),
            prompt,
            temperature=0.2,
        )
        request = _apply_form_overrides(request, options)
        request = _clean_request(request, warnings)

        if not request.destination.strip():
            raise PlannerError(
                "I could not find a destination in your message. Please say where you want to go, "
                "for example: 'Plan a 5 day trip to Goa for 2 people with a moderate budget'."
            )
        return {"request": request, "warnings": warnings}

    def plan_trip(state: PlanState) -> dict[str, Any]:
        request: TripRequest = state["request"]
        warnings = list(state.get("warnings", []))

        prompt = (
            "Trip requirements (JSON):\n"
            + json.dumps(request.model_dump(), ensure_ascii=False, indent=2)
            + f"\n\nCreate the full plan for exactly {request.num_days} day(s)."
        )
        plan = llm.generate(TripPlan, PLAN_SYSTEM, prompt, temperature=0.7)

        # Keep day numbers tidy and warn if the AI returned the wrong number of days.
        if len(plan.days) != request.num_days:
            warnings.append(
                f"The itinerary has {len(plan.days)} day(s) but your trip is {request.num_days} day(s). "
                "Please double-check it."
            )
        start = _parse_date(request.start_date)
        for index, day in enumerate(plan.days, start=1):
            day.day = index
            if start:
                day.date = date.fromordinal(start.toordinal() + index - 1).isoformat()
        return {"plan": plan, "warnings": warnings}

    def analyze_budget(state: PlanState) -> dict[str, Any]:
        request: TripRequest = state["request"]
        plan: TripPlan = state["plan"]

        compact_plan = {
            "flights": [f.model_dump() for f in plan.flights],
            "stays": [s.model_dump() for s in plan.stays],
            "local_transport": [t.model_dump() for t in plan.local_transport],
            "daily_costs": [{"day": d.day, "estimated_cost": d.estimated_cost} for d in plan.days],
        }
        prompt = (
            "Trip requirements (JSON):\n"
            + json.dumps(request.model_dump(), ensure_ascii=False, indent=2)
            + "\n\nProposed plan estimates (JSON):\n"
            + json.dumps(compact_plan, ensure_ascii=False, indent=2)
        )
        advice = llm.generate(BudgetAdvice, BUDGET_SYSTEM, prompt, temperature=0.3)
        return {"advice": advice}

    def finish(state: PlanState) -> dict[str, Any]:
        request: TripRequest = state["request"]
        plan: TripPlan = state["plan"]
        advice: BudgetAdvice = state["advice"]

        # All arithmetic is done here, never by the AI.
        total = round(sum(max(0.0, item.amount) for item in advice.items), 2)
        per_person = round(total / request.travelers, 2) if request.travelers else total
        within: bool | None = None
        difference = 0.0
        if request.budget_total > 0:
            within = total <= request.budget_total
            difference = round(request.budget_total - total, 2)

        budget = BudgetSummary(
            items=advice.items,
            total=total,
            per_person=per_person,
            budget_total=request.budget_total,
            within_budget=within,
            difference=difference,
            summary=advice.summary,
            money_saving_tips=advice.money_saving_tips,
        )
        warnings = list(state.get("warnings", []))
        if within is False:
            warnings.append(
                f"The estimated total is over your budget by {abs(difference):,.0f} {request.currency}. "
                "See the money-saving tips."
            )
        final = FinalPlan(
            request=request,
            plan=plan,
            budget=budget,
            warnings=warnings,
            models_used=list(getattr(llm, "models_used", [])),
        )
        return {"final": final}

    graph = StateGraph(PlanState)
    graph.add_node("analyze_request", analyze_request)
    graph.add_node("plan_trip", plan_trip)
    graph.add_node("analyze_budget", analyze_budget)
    graph.add_node("finish", finish)
    graph.add_edge(START, "analyze_request")
    graph.add_edge("analyze_request", "plan_trip")
    graph.add_edge("plan_trip", "analyze_budget")
    graph.add_edge("analyze_budget", "finish")
    graph.add_edge("finish", END)
    return graph.compile()


def plan_trip_request(
    llm: LLM,
    user_request: str,
    options: dict[str, Any] | None = None,
    on_step: Callable[[str], None] | None = None,
) -> FinalPlan:
    """Run the whole workflow and return the finished plan.

    `on_step(node_name)` is called when each step starts finishing, so the
    web page can show progress.
    """
    text = (user_request or "").strip()
    if not text:
        raise PlannerError("Please describe your trip first.")

    graph = build_graph(llm)
    state: dict[str, Any] = {"user_request": text, "options": options or {}, "warnings": []}

    if on_step:
        on_step("start")
    for update in graph.stream(state, stream_mode="updates"):
        for node_name, node_update in update.items():
            state.update(node_update or {})
            if on_step:
                on_step(node_name)

    final = state.get("final")
    if final is None:
        raise PlannerError("The planner stopped before finishing. Please try again.")
    return final
