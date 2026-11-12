"""AI Travel Planner - Streamlit web app.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from travel_planner import usage
from travel_planner.config import MODEL_CHOICES, get_settings, mask_key
from travel_planner.graph import STEP_LABELS, plan_trip_request
from travel_planner.llm import PlannerError, build_llm
from travel_planner.models import FinalPlan
from travel_planner.render import compact, money, plan_to_markdown
from travel_planner.storage import list_saved_plans, load_plan, save_plan_local, save_plan_supabase

st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")

CURRENCIES = ["USD", "EUR", "GBP", "INR", "AED", "SGD", "AUD", "CAD", "JPY", "THB"]
STYLES = ["(let the AI decide)", "budget", "moderate", "luxury"]
INTERESTS = ["Food", "History", "Culture", "Beaches", "Hiking", "Nightlife", "Shopping", "Nature", "Adventure", "Relaxation"]
EXAMPLES = [
    ("🏖️ Goa weekend", "Plan a 4 day trip to Goa for 2 people from Pune with a moderate budget. We love food and beaches."),
    ("🗼 Tokyo adventure", "Plan a 7 day trip to Tokyo for 2 people with a luxury budget. We love food, culture and shopping."),
    ("🏔️ Manali on a budget", "Plan a 5 day budget trip to Manali for 4 friends from Delhi. We love nature and adventure."),
]

# --------------------------------------------------------------------------
# Look and feel (static CSS only; nothing from the AI is ever put into HTML)
# --------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1150px; }

.hero {
  background: linear-gradient(120deg, #0B5E6B 0%, #0E7C86 45%, #3DA5B8 100%);
  border-radius: 24px; padding: 2.6rem 2.4rem; color: #fff; margin-bottom: 1.4rem;
  box-shadow: 0 12px 30px rgba(14, 124, 134, .28); position: relative; overflow: hidden;
}
.hero:after { content: "✈"; position: absolute; right: 2rem; top: .2rem; font-size: 9rem; opacity: .12; transform: rotate(-12deg); }
.hero h1 { color: #fff; font-size: 2.5rem; font-weight: 800; margin: 0 0 .4rem 0; padding: 0; letter-spacing: -.5px; }
.hero p { color: #E6F6F8; font-size: 1.05rem; margin: 0 0 1rem 0; max-width: 46rem; line-height: 1.55; }
.badge { display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
  color: #fff; padding: 4px 13px; border-radius: 999px; font-size: .8rem; font-weight: 600; margin-right: 8px; }

.chip { display: inline-block; background: #E1F2F4; color: #0B5E6B; padding: 5px 14px; border-radius: 999px;
  font-size: .85rem; font-weight: 600; margin: 3px 8px 3px 0; }
.chip.warm { background: #FDEBD8; color: #9A4F0B; }

h2, h3, h4 { color: #12313B; letter-spacing: -.2px; }
[data-testid="stMetric"] { background: #fff; border: 1px solid #DCE8EC; border-radius: 18px; padding: 16px 20px;
  box-shadow: 0 4px 14px rgba(18, 49, 59, .06); }
[data-testid="stMetricValue"] { font-weight: 800; color: #0B5E6B; font-size: clamp(1.25rem, 2.1vw, 2rem); }
/* Never cut long numbers off with "..." - let them wrap onto a second line instead. */
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { white-space: normal !important; overflow: visible !important;
  text-overflow: clip !important; overflow-wrap: anywhere; line-height: 1.15; }
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * { white-space: normal !important; overflow: visible !important; }
[data-testid="stMetricLabel"] p { color: #5C7480; font-weight: 600; text-transform: uppercase; font-size: .72rem; letter-spacing: .6px; }

[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px; }
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]) { border-color: #DCE8EC; }

.stButton > button { border-radius: 14px; font-weight: 600; border: 1px solid #CFE0E5; padding: .55rem 1rem; transition: all .15s ease; }
.stButton > button:hover { border-color: #0E7C86; color: #0E7C86; transform: translateY(-1px); }
.stButton > button[kind="primary"] { background: linear-gradient(120deg, #0E7C86, #12A0AC); color: #fff; border: none;
  font-size: 1.05rem; padding: .75rem 1.6rem; box-shadow: 0 8px 18px rgba(14,124,134,.32); }
.stButton > button[kind="primary"]:hover { color: #fff; filter: brightness(1.07); }

.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 2px solid #DCE8EC; }
.stTabs [data-baseweb="tab"] { font-weight: 700; padding: .6rem 1.1rem; border-radius: 12px 12px 0 0; }
.stTextArea textarea { border-radius: 16px; font-size: 1.02rem; }
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #DCE8EC; }
.small-note { color: #6A808B; font-size: .85rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def chips(values: list[str], warm: bool = False) -> None:
    """Show small pill labels. Values are escaped, so any text is safe."""
    cls = "chip warm" if warm else "chip"
    markup = "".join(f'<span class="{cls}">{html.escape(str(v))}</span>' for v in values if str(v).strip())
    if markup:
        st.markdown(markup, unsafe_allow_html=True)


def bullet_list(items) -> None:
    for item in compact(items):
        st.markdown(f"- {item}")


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

settings = get_settings()

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    choices = list(dict.fromkeys([settings.model, *MODEL_CHOICES]))
    model = st.selectbox(
        "Gemini model",
        choices,
        index=0,
        help="Flash-Lite models have a much bigger free daily allowance. If one runs out or is busy, the next one is tried automatically.",
    )
    if not settings.public_mode:
        st.caption(f"🔑 API key: {mask_key(settings.api_key)}")
    if not settings.api_key:
        st.error("No GEMINI_API_KEY found (in the .env file, or in the app's Secrets when deployed).")

    if not settings.public_mode:
        st.divider()
        st.markdown("### 🗂️ Saved trips")
        saved = list_saved_plans()
        if saved:
            picked = st.selectbox("Open an earlier plan", ["-"] + [p.name for p in saved[:25]])
            if picked != "-" and st.button("Open this plan"):
                try:
                    st.session_state["result"] = load_plan(next(p for p in saved if p.name == picked))
                    st.session_state["save_note"] = ""
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not open that file: {exc}")
        else:
            st.caption("Nothing saved yet. Plans are saved automatically in the 'saved_plans' folder.")
    else:
        st.caption("Demo mode: your plan is not stored on the server. Download it from the Download tab.")

    st.divider()
    st.markdown('<p class="small-note">Prices are AI estimates, not live prices. Always verify before booking.</p>',
                unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Hero + input
# --------------------------------------------------------------------------

st.markdown(
    """
<div class="hero">
  <h1>AI Travel Planner</h1>
  <p>Describe your trip in your own words and get a complete plan: destination guide, flights, stays,
  local transport, a day-by-day itinerary and a budget.</p>
  <span class="badge">Google Gemini</span><span class="badge">LangGraph</span><span class="badge">Free to run</span>
</div>
""",
    unsafe_allow_html=True,
)


def use_example(text: str) -> None:
    st.session_state["query_text"] = text


st.markdown("##### ✨ Try an example")
example_cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(example_cols, EXAMPLES):
    col.button(label, on_click=use_example, args=(text,))

query = st.text_area(
    "Describe your trip",
    key="query_text",
    placeholder="Example: Plan a 5 day trip to Goa for 2 people from Pune with a moderate budget. We love food and beaches.",
    height=130,
)

with st.expander("⚙️ Optional details (these override anything the AI guesses)"):
    c1, c2, c3 = st.columns(3)
    origin = c1.text_input("Travelling from")
    travelers = c2.number_input("Travellers", min_value=1, max_value=20, value=1, step=1)
    style = c3.selectbox("Travel style", STYLES)

    c4, c5 = st.columns(2)
    budget = c4.number_input("Total budget for the whole group (0 = not set)", min_value=0.0, value=0.0, step=100.0)
    currency_default = settings.default_currency if settings.default_currency in CURRENCIES else "USD"
    currency = c5.selectbox("Currency", CURRENCIES, index=CURRENCIES.index(currency_default))

    use_dates = st.checkbox("I know my travel dates")
    start_date = end_date = None
    if use_dates:
        d1, d2 = st.columns(2)
        start_date = d1.date_input("Start date")
        end_date = d2.date_input("End date")

    interests = st.multiselect("Interests", INTERESTS)

go = st.button("🌍 Plan my trip", type="primary")


def collect_options() -> dict:
    return {
        "origin": origin.strip(),
        "travelers": int(travelers) if travelers and int(travelers) > 1 else 0,
        "budget_total": float(budget) if budget else 0,
        "currency": currency if budget else "",
        "travel_style": "" if style.startswith("(") else style,
        "start_date": start_date.isoformat() if start_date else "",
        "end_date": end_date.isoformat() if end_date else "",
        "interests": [i.lower() for i in interests],
    }


def _refund_trip() -> None:
    """A failed trip should not use up the visitor's tries or the shared daily quota."""
    st.session_state["trips_started"] = max(0, st.session_state.get("trips_started", 1) - 1)
    usage.COUNTER.give_back()


if go:
    if not query.strip():
        st.warning("Please describe your trip first.")
    else:
        run_settings = get_settings(model_override=model)
        if run_settings.session_trip_limit and st.session_state.get("trips_started", 0) >= run_settings.session_trip_limit:
            st.warning(f"You have reached the limit of {run_settings.session_trip_limit} trip plans for this visit. "
                       "This is a free demo with a limited AI quota. Please come back later.")
        elif not usage.COUNTER.try_take(run_settings.daily_trip_limit):
            st.warning("This free demo has used up its AI quota for today. Please try again tomorrow.")
        else:
            st.session_state["trips_started"] = st.session_state.get("trips_started", 0) + 1
            try:
                llm = build_llm(run_settings)
                with st.status("Planning your trip... (usually 20-60 seconds)", expanded=True) as status:

                    def on_step(name: str) -> None:
                        if name == "start":
                            st.write("🧭 Starting...")
                        elif name in STEP_LABELS:
                            st.write(f"✅ {STEP_LABELS[name]}")

                    final = plan_trip_request(llm, query, collect_options(), on_step=on_step)
                    status.update(label="Trip plan ready!", state="complete", expanded=False)

                st.session_state["result"] = final
                st.session_state["save_note"] = ""
                if not run_settings.public_mode:
                    try:
                        path = save_plan_local(final)
                        st.session_state["save_note"] = f"Saved to saved_plans/{path.name}"
                    except OSError as exc:
                        st.session_state["save_note"] = f"(Could not save a local copy: {exc})"
                    if run_settings.save_to_supabase:
                        st.session_state["save_note"] += " | " + save_plan_supabase(final, run_settings)

            except PlannerError as exc:
                _refund_trip()
                st.error(exc.user_message)
                if exc.technical:
                    with st.expander("Technical details (for debugging)"):
                        st.code(exc.technical)
            except Exception as exc:  # noqa: BLE001
                _refund_trip()
                st.error("Something unexpected went wrong. Details below.")
                st.exception(exc)


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

result: FinalPlan | None = st.session_state.get("result")

if result:
    r, p, b = result.request, result.plan, result.budget
    cur = r.currency

    st.divider()
    st.header(f"Your trip to {r.destination}")
    facts = [f"📅 {r.num_days} days", f"👥 {r.travelers} traveller{'s' if r.travelers != 1 else ''}",
             f"💼 {r.travel_style.title()} style"]
    if r.origin:
        facts.append(f"📍 From {r.origin}")
    if r.start_date:
        facts.append(f"🗓️ Starts {r.start_date}")
    chips(facts)
    chips(r.interests, warm=True)
    if st.session_state.get("save_note"):
        st.caption(st.session_state["save_note"])

    for warning in result.warnings:
        st.warning(warning)
    if compact(r.assumptions):
        with st.expander("💡 Assumptions the AI made"):
            bullet_list(r.assumptions)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Days", r.num_days)
    m2.metric("Travellers", r.travelers)
    m3.metric("Estimated total", money(b.total, cur))
    if b.budget_total > 0:
        # "+" (under budget, green) or "-" (over budget, red) comes from the sign of the number.
        m4.metric("Your budget", money(b.budget_total, cur), delta=f"{b.difference:+,.0f} {cur} vs estimate")
    else:
        m4.metric("Per person", money(b.per_person, cur))

    st.write("")
    tab_overview, tab_days, tab_travel, tab_budget, tab_download = st.tabs(
        ["🌍 Overview", "🗓️ Itinerary", "✈️ Flights, stays & transport", "💰 Budget", "⬇️ Download"]
    )

    # ---- Overview ---------------------------------------------------------
    with tab_overview:
        d = p.destination_info
        with st.container(border=True):
            st.markdown(f"#### About {r.destination}")
            st.write(d.summary)
        k1, k2, k3 = st.columns(3)
        for col, title, text in (
            (k1, "📅 Best time to visit", d.best_time_to_visit),
            (k2, "🌤️ Weather on your trip", d.weather_during_trip),
            (k3, "🛂 Entry requirements", d.entry_requirements),
        ):
            with col, st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(text)
        g1, g2, g3 = st.columns(3)
        for col, title, items in (
            (g1, "💡 Local tips", d.local_tips),
            (g2, "🎭 Culture", d.cultural_notes),
            (g3, "🛡️ Safety & health", d.safety_and_health),
        ):
            with col, st.container(border=True):
                st.markdown(f"**{title}**")
                bullet_list(items)
        if compact(p.packing_tips):
            with st.container(border=True):
                st.markdown("**🎒 Packing tips**")
                bullet_list(p.packing_tips)

    # ---- Itinerary --------------------------------------------------------
    with tab_days:
        for day in p.days:
            with st.container(border=True):
                head = f"#### Day {day.day} · {day.title}"
                if day.date:
                    head += f"  —  {day.date}"
                st.markdown(head)
                a, bcol, c = st.columns(3)
                a.markdown("**🌅 Morning**")
                a.write(day.morning)
                bcol.markdown("**☀️ Afternoon**")
                bcol.write(day.afternoon)
                c.markdown("**🌙 Evening**")
                c.write(day.evening)
                st.caption(f"🍽️ {day.food_suggestion}   ·   💰 about {money(day.estimated_cost, cur)} for the group")

    # ---- Flights, stays, transport ------------------------------------------
    with tab_travel:
        st.markdown("#### ✈️ Flights (estimates)")
        cols = st.columns(max(1, min(3, len(p.flights))))
        for i, f in enumerate(p.flights):
            with cols[i % len(cols)], st.container(border=True):
                st.markdown(f"**{f.route}**")
                st.caption(f"{f.typical_airlines} · {f.typical_duration}")
                st.markdown(f"### {money(f.estimated_price_per_person, cur)}")
                st.caption("per person, round trip")
                st.write(f.notes)

        st.markdown("#### 🏨 Where to stay (estimates)")
        cols = st.columns(max(1, min(3, len(p.stays))))
        for i, s in enumerate(p.stays):
            with cols[i % len(cols)], st.container(border=True):
                st.markdown(f"**{s.name_or_area}**")
                st.caption(s.category)
                st.markdown(f"### {money(s.estimated_price_per_night, cur)}")
                st.caption("per night, whole group")
                st.write(s.why_recommended)

        st.markdown("#### 🚕 Getting around")
        cols = st.columns(max(1, min(3, len(p.local_transport))))
        for i, t in enumerate(p.local_transport):
            with cols[i % len(cols)], st.container(border=True):
                st.markdown(f"**{t.mode}**")
                st.write(t.details)
                st.caption(f"about {money(t.estimated_cost, cur)} for the whole trip")

    # ---- Budget -----------------------------------------------------------
    with tab_budget:
        left, right = st.columns([3, 2])
        with left:
            st.markdown("#### Where the money goes")
            chart_df = pd.DataFrame({"Category": [i.category for i in b.items],
                                     f"Estimate ({cur})": [round(i.amount) for i in b.items]}).set_index("Category")
            st.bar_chart(chart_df, color="#0E7C86")
        with right:
            st.markdown("#### Summary")
            st.metric("Estimated total", money(b.total, cur))
            st.caption(f"about {money(b.per_person, cur)} per person")
            if b.budget_total > 0:
                st.progress(min(b.total / b.budget_total, 1.0))
                if b.within_budget:
                    st.success(f"Within your budget of {money(b.budget_total, cur)} ({money(b.difference, cur)} to spare).")
                else:
                    st.error(f"Over your budget of {money(b.budget_total, cur)} by {money(abs(b.difference), cur)}.")

        rows = [{"Category": i.category, f"Estimate ({cur})": round(i.amount), "Notes": i.notes} for i in b.items]
        rows.append({"Category": "TOTAL", f"Estimate ({cur})": round(b.total), "Notes": f"about {money(b.per_person, cur)} per person"})
        st.dataframe(rows, hide_index=True)

        if b.summary:
            st.info(b.summary)
        if compact(b.money_saving_tips):
            with st.container(border=True):
                st.markdown("**💸 Money-saving tips**")
                bullet_list(b.money_saving_tips)

        st.markdown("#### Spending day by day")
        daily = pd.DataFrame({"Day": [f"Day {d.day}" for d in p.days],
                              f"Estimate ({cur})": [round(d.estimated_cost) for d in p.days]}).set_index("Day")
        st.bar_chart(daily, color="#F4A261")

    # ---- Download ---------------------------------------------------------
    with tab_download:
        slug = "".join(ch if ch.isalnum() else "-" for ch in r.destination.lower()).strip("-") or "trip"
        st.markdown("#### Take your plan with you")
        st.download_button("⬇️ Download as Markdown (.md)", plan_to_markdown(result),
                           file_name=f"{slug}-trip-plan.md", mime="text/markdown")
        st.download_button("⬇️ Download as JSON (.json)", result.model_dump_json(indent=2),
                           file_name=f"{slug}-trip-plan.json", mime="application/json")
        st.caption(f"Generated {result.generated_at} with {', '.join(result.models_used) or 'Gemini'}.")
