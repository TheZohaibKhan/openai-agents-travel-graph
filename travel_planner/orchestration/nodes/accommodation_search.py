from travel_planner.orchestration.nodes.base_node import run_agent_sync
"""
Accommodation search node implementation for the travel planning workflow.

This module defines functions for searching accommodations using the
accommodation search agent, both for individual execution and as part
of parallel processing.
"""

from travel_planner.agents.accommodation import AccommodationAgent
from travel_planner.data.models import NodeFunctionParams
from travel_planner.orchestration.nodes.base_node import create_node_function
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.orchestration.states.workflow_stages import WorkflowStage
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


# Using the factory function for a simpler implementation
accommodation_search = create_node_function(
    NodeFunctionParams(
        agent_class=AccommodationAgent,
        task_name="accommodation_search",
        complete_stage=WorkflowStage.ACCOMMODATION_SEARCHED,
        result_field="accommodation_options",
        plan_field="accommodation",
        message_template="Found {count} accommodation options",
    )
)


def accommodation_search_task(state: TravelPlanningState) -> dict[str, any]:
    """
    Execute accommodation search task in parallel branch.

    Args:
        state: Current travel planning state

    Returns:
        Dictionary with task results
    """
    from travel_planner.orchestration.parallel import ParallelResult, ParallelTask

    try:
        agent = AccommodationAgent()
        result = run_agent_sync(agent, state)

        return {
            "result": ParallelResult(
                task_type=ParallelTask.ACCOMMODATION_SEARCH,
                result=result,
                completed=True,
            )
        }
    except Exception as e:
        logger.error(f"Error in accommodation search task: {e!s}")
        return {
            "result": ParallelResult(
                task_type=ParallelTask.ACCOMMODATION_SEARCH,
                result={},
                error=str(e),
                completed=False,
            )
        }
