"""
Parallel search node implementation for the travel planning workflow.

This module executes flight, accommodation, and transportation searches
concurrently using the project's asyncio-based parallel execution system.
"""

import asyncio

from travel_planner.orchestration.parallel import parallel_search_tasks
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


def create_parallel_search_branch():
    """
    Create the parallel search workflow node.

    The previous implementation used LangGraph's internal ParallelBranch
    API, which is not available in the current LangGraph version.

    The project already provides asyncio-based parallel execution through
    parallel_search_tasks(), so this function exposes that functionality
    as a normal LangGraph-compatible node.
    """

    def parallel_search_node(state: TravelPlanningState) -> TravelPlanningState:
        logger.info("Starting parallel search tasks")

        try:
            updated_state = asyncio.run(parallel_search_tasks(state))

            logger.info("Parallel search tasks completed")
            return updated_state

        except Exception as e:
            logger.error(f"Error in parallel search node: {e!s}")

            state.error = f"Parallel search error: {e!s}"

            if state.plan is not None:
                if not state.plan.alerts:
                    state.plan.alerts = []
                state.plan.alerts.append(
                    f"Parallel search error: {e!s}"
                )

            state.update_stage(WorkflowStage.ERROR)
            return state

    parallel_search_node.__name__ = "parallel_search"

    return parallel_search_node


def combine_search_results(state: TravelPlanningState) -> TravelPlanningState:
    """
    Combine the results from the parallel search node.

    Args:
        state: Current travel planning state

    Returns:
        Updated state with combined results from parallel search.
    """
    logger.info("Combining results from parallel search")

    state.update_stage(WorkflowStage.PARALLEL_SEARCH_COMPLETED)

    has_plan = state.plan is not None

    flights = state.plan.flights if has_plan else None
    flight_count = len(flights) if flights else 0

    accom = state.plan.accommodation if has_plan else None
    accom_count = len(accom) if accom else 0

    transport = state.plan.transportation if has_plan else None
    transport_count = len(transport) if transport else 0

    state.conversation_history.append(
        {
            "role": "system",
            "content": (
                f"Completed parallel search: {flight_count} flights, "
                f"{accom_count} accommodations, "
                f"{transport_count} transportation options"
            ),
        }
    )

    return state