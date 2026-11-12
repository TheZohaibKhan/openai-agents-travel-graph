"""Turn a FinalPlan into readable Markdown (used for display and download)."""

from __future__ import annotations

from travel_planner.models import FinalPlan


def compact(items) -> list[str]:
    """Drop empty / whitespace-only entries from a list of text."""
    return [str(i).strip() for i in (items or []) if str(i).strip()]


def money(amount: float, currency: str) -> str:
    return f"{amount:,.0f} {currency}"


def plan_to_markdown(final: FinalPlan) -> str:
    r, p, b = final.request, final.plan, final.budget
    cur = r.currency
    out: list[str] = []

    out.append(f"# Trip to {r.destination}")
    facts = [f"{r.num_days} day(s)", f"{r.travelers} traveller(s)", f"style: {r.travel_style}"]
    if r.origin:
        facts.append(f"from {r.origin}")
    if r.start_date:
        facts.append(f"starting {r.start_date}")
    out.append(" | ".join(facts))
    out.append("")
    out.append("> All prices are AI estimates, not live prices. Always check flights, hotels and entry rules before booking.")
    out.append("")

    if final.warnings:
        out.append("## Heads up")
        out.extend(f"- {w}" for w in final.warnings)
        out.append("")
    if compact(r.assumptions):
        out.append("## Assumptions I made")
        out.extend(f"- {a}" for a in compact(r.assumptions))
        out.append("")

    d = p.destination_info
    out += ["## About the destination", d.summary, ""]
    out += [f"**Best time to visit:** {d.best_time_to_visit}", ""]
    out += [f"**Weather during your trip:** {d.weather_during_trip}", ""]
    out += [f"**Entry requirements:** {d.entry_requirements}", ""]
    for title, items in (
        ("Local tips", d.local_tips),
        ("Culture", d.cultural_notes),
        ("Safety and health", d.safety_and_health),
    ):
        if compact(items):
            out.append(f"**{title}**")
            out.extend(f"- {i}" for i in compact(items))
            out.append("")

    out.append("## Day-by-day itinerary")
    for day in p.days:
        heading = f"### Day {day.day}: {day.title}"
        if day.date:
            heading += f" ({day.date})"
        out += [
            heading,
            f"- **Morning:** {day.morning}",
            f"- **Afternoon:** {day.afternoon}",
            f"- **Evening:** {day.evening}",
            f"- **Food:** {day.food_suggestion}",
            f"- **Estimated spend:** {money(day.estimated_cost, cur)}",
            "",
        ]

    out.append("## Flights (estimates)")
    for f in p.flights:
        out.append(
            f"- **{f.route}** - {f.typical_airlines}, {f.typical_duration}, about "
            f"{money(f.estimated_price_per_person, cur)} per person. {f.notes}"
        )
    out.append("")

    out.append("## Where to stay (estimates)")
    for s in p.stays:
        out.append(
            f"- **{s.name_or_area}** ({s.category}) - about {money(s.estimated_price_per_night, cur)} per night. "
            f"{s.why_recommended}"
        )
    out.append("")

    out.append("## Getting around")
    for t in p.local_transport:
        out.append(f"- **{t.mode}** - {t.details} (about {money(t.estimated_cost, cur)} total)")
    out.append("")

    out.append("## Budget")
    out.append("| Category | Estimate | Notes |")
    out.append("|---|---:|---|")
    for item in b.items:
        out.append(f"| {item.category} | {money(item.amount, cur)} | {item.notes} |")
    out.append(f"| **Total** | **{money(b.total, cur)}** | about {money(b.per_person, cur)} per person |")
    out.append("")
    if b.budget_total > 0:
        status = "within" if b.within_budget else "over"
        out.append(f"Your budget: {money(b.budget_total, cur)} - the plan is **{status} budget** "
                   f"({'under' if b.within_budget else 'over'} by {money(abs(b.difference), cur)}).")
        out.append("")
    if b.summary:
        out += [b.summary, ""]
    if compact(b.money_saving_tips):
        out.append("**Money-saving tips**")
        out.extend(f"- {t}" for t in compact(b.money_saving_tips))
        out.append("")

    if compact(p.packing_tips):
        out.append("## Packing tips")
        out.extend(f"- {t}" for t in compact(p.packing_tips))
        out.append("")

    out.append(f"*Generated {final.generated_at} with {', '.join(final.models_used) or 'Gemini'}.*")
    return "\n".join(out)
