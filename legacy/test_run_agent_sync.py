from travel_planner.orchestration.nodes.base_node import run_agent_sync
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.data.models import TravelQuery


class FakeAgent:

    async def run(self, conversation):

        print("MESSAGES:")

        for message in conversation:
            print(message)

        assert conversation[-1]["role"] == "user"
        assert "Paris" in conversation[-1]["content"]

        return {"test": "ok"}


state = TravelPlanningState(
    query=TravelQuery(
        raw_query="Plan a 5 day trip to Paris for 2 people"
    )
)

state.preferences = {
    "style": "budget"
}

result = run_agent_sync(
    FakeAgent(),
    state,
)

print("RESULT:", result)
print("RUN_AGENT_SYNC TEST: OK")