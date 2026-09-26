import asyncio
import streamlit as st

from travel_planner.orchestration.workflow import TravelWorkflow


st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
)


st.title("✈️ AI Travel Planner")
st.write(
    "Plan trips using Google Gemini, LangGraph, and Supabase."
)

st.divider()

query = st.text_area(
    "Describe your trip",
    placeholder=(
        "Example: Plan a 5 day trip to Paris for 2 people "
        "with a moderate budget."
    ),
    height=120,
)

if st.button("🌍 Plan My Trip", type="primary"):
    if not query.strip():
        st.warning("Please enter a travel request.")
    else:
        try:
            with st.spinner("Planning your trip..."):
                workflow = TravelWorkflow()

                result = asyncio.run(
                    workflow.process_query_async(query.strip())
                )

            st.success("Trip planning completed!")

            st.subheader("🗺️ Your Travel Plan")

            if hasattr(result, "travel_plan") and result.travel_plan:
                st.json(
                    result.travel_plan.model_dump(
                        mode="json"
                    )
                )
            elif hasattr(result, "plan") and result.plan:
                st.json(
                    result.plan.model_dump(
                        mode="json"
                    )
                )
            else:
                st.json(
                    result.model_dump(mode="json")
                    if hasattr(result, "model_dump")
                    else result
                )

        except Exception as exc:
            st.error(f"Travel planning failed: {exc}")
            st.exception(exc)