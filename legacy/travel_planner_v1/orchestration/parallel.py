"""
Parallel execution capabilities for the travel planner system.

This module executes the specialized travel agents concurrently while keeping
compatibility with the current async BaseAgent.run() interface.
"""

import asyncio
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from travel_planner.agents.base import BaseAgent
from travel_planner.orchestration.states.workflow_stages import (
    PARALLEL_SEARCH_COMPLETED,
)
from travel_planner.utils.logging import get_logger


if TYPE_CHECKING:
    from travel_planner.orchestration.states.planning_state import TravelPlanningState
else:
    TravelPlanningState = Any


T = TypeVar("T")
UpdateFunction = Callable[[T], T]

logger = get_logger(__name__)


class ParallelTask(Enum):
    """Enum representing parallel travel-planning tasks."""

    FLIGHT_SEARCH = "flight_search"
    ACCOMMODATION = "accommodation"
    TRANSPORTATION = "transportation"
    ACTIVITIES = "activities"
    BUDGET = "budget"


class ParallelResult(BaseModel):
    """Model for storing results from parallel task execution."""

    task_type: ParallelTask
    result: dict[str, Any]
    error: str | None = None
    completed: bool = False


def _get_raw_query(state: TravelPlanningState) -> str:
    """Extract the original user query from the workflow state."""

    query = getattr(state, "query", None)

    if query is not None:
        raw_query = getattr(query, "raw_query", None)
        if raw_query:
            return str(raw_query)

    history = getattr(state, "conversation_history", None) or []

    # Search from newest to oldest so the actual user request is preferred.
    for message in reversed(history):
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content")

        if role == "user" and content:
            text = str(content)

            # Ignore generated structured-context messages.
            if not text.startswith("Current travel planning context:"):
                return text

    return "Plan this trip."


def _build_agent_input(state: TravelPlanningState) -> list[dict[str, Any]]:
    """
    Build chat input for agents.

    The original user query is preserved and useful structured information
    from the current state is appended separately.
    """

    query = getattr(state, "query", None)
    preferences = getattr(state, "preferences", None)

    original_query = _get_raw_query(state)

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": original_query,
        }
    ]

    context_parts: list[str] = []

    if query is not None:
        fields = (
            "origin",
            "destination",
            "departure_date",
            "return_date",
            "travelers",
            "budget",
            "trip_duration",
        )

        for field in fields:
            value = getattr(query, field, None)

            if value is not None and value != "":
                context_parts.append(f"{field}: {value}")

    if preferences is not None:
        context_parts.append(f"preferences: {preferences}")

    if context_parts:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Current travel planning context:\n"
                    + "\n".join(context_parts)
                ),
            }
        )

    return messages


async def _run_agent(
    agent: BaseAgent,
    input_data: str | list[dict[str, Any]],
    timeout: int,
) -> Any:
    """
    Run a migrated async agent.

    All specialized agents in the current project expose:
        async def run(input_data, context=None)

    We intentionally do not call process() here.
    """

    return await asyncio.wait_for(
        agent.run(input_data),
        timeout=timeout,
    )


async def execute_in_parallel(
    tasks: list[tuple[BaseAgent, dict[str, Any]]],
    state: TravelPlanningState,
) -> dict[str, Any]:
    """
    Execute multiple agents concurrently.

    Each task dictionary may contain:
        input_data
        timeout

    No agent-specific keyword arguments are passed directly to run().
    """

    async def execute_task(
        agent: BaseAgent,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        timeout = int(params.get("timeout", 60))
        input_data = params.get("input_data")

        if input_data is None:
            input_data = _build_agent_input(state)

        try:
            result = await _run_agent(
                agent=agent,
                input_data=input_data,
                timeout=timeout,
            )

            logger.info(
                f"Parallel task completed successfully: {agent.name}"
            )

            return {
                agent.name: {
                    "result": result,
                    "error": None,
                    "retries": 0,
                }
            }

        except asyncio.TimeoutError:
            error_message = (
                f"{agent.name} timed out after {timeout} seconds"
            )

            logger.error(error_message)

            return {
                agent.name: {
                    "result": None,
                    "error": error_message,
                    "retries": 0,
                }
            }

        except Exception as exc:
            error_message = str(exc)

            logger.error(
                f"Error in parallel task {agent.name}: {error_message}"
            )

            return {
                agent.name: {
                    "result": None,
                    "error": error_message,
                    "retries": 0,
                }
            }

    coroutines = [
        execute_task(agent, params)
        for agent, params in tasks
    ]

    try:
        results = await asyncio.gather(*coroutines)

        combined_results: dict[str, Any] = {}

        for result in results:
            combined_results.update(result)

        return combined_results

    except asyncio.TimeoutError:
        logger.error("Parallel execution timed out")

        return {
            "error": "Execution timeout exceeded"
        }


async def parallel_search_tasks(
    state: TravelPlanningState,
) -> TravelPlanningState:
    """
    Execute flight, accommodation, transportation, and activity agents
    concurrently.
    """

    from travel_planner.agents.accommodation import AccommodationAgent
    from travel_planner.agents.activity_planning import ActivityPlanningAgent
    from travel_planner.agents.flight_search import FlightSearchAgent
    from travel_planner.agents.transportation import TransportationAgent

    logger.info("Setting up parallel search tasks")

    working_state = state.model_copy(deep=True)

    try:
        agents = {
            "flight": FlightSearchAgent(),
            "accommodation": AccommodationAgent(),
            "transportation": TransportationAgent(),
            "activity": ActivityPlanningAgent(),
        }

        agent_input = _build_agent_input(working_state)

        tasks = [
            (
                agents["flight"],
                {
                    "input_data": agent_input,
                    "timeout": 60,
                },
            ),
            (
                agents["accommodation"],
                {
                    "input_data": agent_input,
                    "timeout": 60,
                },
            ),
            (
                agents["transportation"],
                {
                    "input_data": agent_input,
                    "timeout": 45,
                },
            ),
            (
                agents["activity"],
                {
                    "input_data": agent_input,
                    "timeout": 60,
                },
            ),
        ]

        logger.info(
            f"Executing {len(tasks)} tasks in parallel"
        )

        async with asyncio.timeout(180):
            results = await execute_in_parallel(
                tasks,
                working_state,
            )

        # Detect complete failure.
        agent_results = [
            value
            for key, value in results.items()
            if key != "error" and isinstance(value, dict)
        ]

        successful_results = [
            item
            for item in agent_results
            if item.get("result") is not None
            and not item.get("error")
        ]

        if not successful_results and results.get("error"):
            logger.error(
                f"All parallel tasks failed: {results['error']}"
            )

            working_state.error = (
                f"Parallel execution error: {results['error']}"
            )

            _add_plan_alert(
                working_state,
                f"Error in parallel search: {results['error']}",
            )

            working_state.current_stage = "error"

            return working_state

        updated_state = merge_parallel_results(
            working_state,
            results,
        )

        updated_state.current_stage = PARALLEL_SEARCH_COMPLETED

        logger.info(
            "Parallel search tasks completed"
        )

        return updated_state

    except asyncio.TimeoutError:
        logger.error(
            "Parallel search tasks timed out after 3 minutes"
        )

        working_state.error = "Parallel search timed out"

        _add_plan_alert(
            working_state,
            "Search operations timed out. Some results may be incomplete.",
        )

        working_state.current_stage = "error"

        return working_state

    except Exception as exc:
        logger.error(
            f"Unexpected error in parallel search: {exc!s}"
        )

        working_state.error = (
            f"Unexpected error: {exc!s}"
        )

        _add_plan_alert(
            working_state,
            f"Unexpected error in search: {exc!s}",
        )

        working_state.current_stage = "error"

        return working_state


def _add_plan_alert(
    state: TravelPlanningState,
    message: str,
) -> None:
    """Safely add an alert to the travel plan."""

    if state.plan is None:
        from travel_planner.data.models import TravelPlan

        state.plan = TravelPlan()

    if state.plan.alerts is None:
        state.plan.alerts = []

    state.plan.alerts.append(message)


def merge_parallel_results(
    state: TravelPlanningState,
    results: dict[str, Any],
) -> TravelPlanningState:
    """Merge parallel agent results into the workflow state."""

    updated_state = state.model_copy(deep=True)

    updated_state = _ensure_plan_initialized(
        updated_state
    )

    updated_state = _process_flight_results(
        updated_state,
        results,
    )

    updated_state = _process_accommodation_results(
        updated_state,
        results,
    )

    updated_state = _process_transportation_results(
        updated_state,
        results,
    )

    updated_state = _process_activity_results(
        updated_state,
        results,
    )

    updated_state = _process_parallel_errors(
        updated_state,
        results,
    )

    return updated_state


def _ensure_plan_initialized(
    state: TravelPlanningState,
) -> TravelPlanningState:
    """Ensure a TravelPlan exists."""

    if state.plan is None:
        from travel_planner.data.models import TravelPlan

        state.plan = TravelPlan()

    if state.plan.alerts is None:
        state.plan.alerts = []

    return state


def _unwrap_agent_result(value: Any) -> Any:
    """
    Normalize an agent result.

    Agents may return:
      - dict
      - Pydantic model
      - object containing a result-like dictionary
    """

    if value is None:
        return None

    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass

    if hasattr(value, "dict"):
        try:
            return value.dict()
        except Exception:
            pass

    if hasattr(value, "result"):
        nested = getattr(value, "result", None)

        if isinstance(nested, dict):
            return nested

        if hasattr(nested, "model_dump"):
            try:
                return nested.model_dump()
            except Exception:
                pass

    return value


def _process_flight_results(
    state: TravelPlanningState,
    results: dict[str, Any],
) -> TravelPlanningState:
    """Process flight search results."""

    entry = results.get("FlightSearchAgent")

    if not entry or not entry.get("result"):
        return state

    flight_data = _unwrap_agent_result(
        entry["result"]
    )

    if not isinstance(flight_data, dict):
        return state

    if "flights" in flight_data:
        state.plan.flights = flight_data["flights"]

    elif "flight_options" in flight_data:
        state.plan.flights = flight_data["flight_options"]

    return state


def _process_accommodation_results(
    state: TravelPlanningState,
    results: dict[str, Any],
) -> TravelPlanningState:
    """Process accommodation results."""

    entry = results.get("AccommodationAgent")

    if not entry or not entry.get("result"):
        return state

    accommodation_data = _unwrap_agent_result(
        entry["result"]
    )

    if not isinstance(accommodation_data, dict):
        return state

    if "accommodations" in accommodation_data:
        state.plan.accommodation = (
            accommodation_data["accommodations"]
        )

    elif "accommodation_options" in accommodation_data:
        state.plan.accommodation = (
            accommodation_data["accommodation_options"]
        )

    return state


def _process_transportation_results(
    state: TravelPlanningState,
    results: dict[str, Any],
) -> TravelPlanningState:
    """Process transportation results."""

    entry = results.get("TransportationAgent")

    if not entry or not entry.get("result"):
        return state

    transportation_data = _unwrap_agent_result(
        entry["result"]
    )

    if not isinstance(transportation_data, dict):
        return state

    if "transportation" in transportation_data:
        state.plan.transportation = (
            transportation_data["transportation"]
        )

    elif "transportation_options" in transportation_data:
        state.plan.transportation = (
            transportation_data["transportation_options"]
        )

    return state


def _process_activity_results(
    state: TravelPlanningState,
    results: dict[str, Any],
) -> TravelPlanningState:
    """Process activity planning results."""

    entry = results.get("ActivityPlanningAgent")

    if not entry or not entry.get("result"):
        return state

    activity_data = _unwrap_agent_result(
        entry["result"]
    )

    if not isinstance(activity_data, dict):
        return state

    if "activities" in activity_data:
        state.plan.activities = activity_data["activities"]

    elif "daily_itineraries" in activity_data:
        state.plan.activities = (
            activity_data["daily_itineraries"]
        )

    return state


def _process_parallel_errors(
    state: TravelPlanningState,
    results: dict[str, Any],
) -> TravelPlanningState:
    """Add individual parallel-agent errors to plan alerts."""

    errors: list[str] = []

    for agent_name, result in results.items():
        if agent_name == "error":
            continue

        if not isinstance(result, dict):
            continue

        error = result.get("error")

        if error:
            errors.append(
                f"{agent_name}: {error}"
            )

    if errors:
        _ensure_plan_initialized(state)

        state.plan.alerts.extend(errors)

    return state


# ---------------------------------------------------------------------------
# Compatibility helpers for LangGraph branch execution
# ---------------------------------------------------------------------------

def combine_parallel_branch_results(
    state: TravelPlanningState,
    branch_results: dict[str, ParallelResult],
) -> TravelPlanningState:
    """Combine LangGraph parallel branch results."""

    updated_state = state.model_copy(deep=True)

    updated_state = _ensure_plan_initialized(
        updated_state
    )

    if not _validate_branch_results(branch_results):
        logger.warning(
            "No successful results from parallel branch execution"
        )
        return updated_state

    results_by_task = _organize_branch_results(
        branch_results
    )

    updated_state = _process_branch_flight_results(
        updated_state,
        results_by_task,
    )

    updated_state = _process_branch_accommodation_results(
        updated_state,
        results_by_task,
    )

    updated_state = _process_branch_transportation_results(
        updated_state,
        results_by_task,
    )

    updated_state = _process_branch_activity_results(
        updated_state,
        results_by_task,
    )

    updated_state = _process_branch_budget_results(
        updated_state,
        results_by_task,
    )

    updated_state = _process_branch_errors(
        updated_state,
        results_by_task,
    )

    updated_state = _update_workflow_stage(
        updated_state
    )

    return updated_state


def _validate_branch_results(
    branch_results: dict[str, ParallelResult],
) -> bool:
    """Check whether branch results contain usable data."""

    if not branch_results:
        return False

    if branch_results.get("result"):
        return True

    for value in branch_results.values():
        if (
            isinstance(value, ParallelResult)
            and value.completed
            and not value.error
        ):
            return True

    return False


def _organize_branch_results(
    branch_results: dict[str, ParallelResult],
) -> dict[ParallelTask, ParallelResult]:
    """Organize branch results by task type."""

    results_by_task: dict[
        ParallelTask,
        ParallelResult,
    ] = {}

    for task_result in branch_results.values():
        if isinstance(task_result, ParallelResult):
            results_by_task[
                task_result.task_type
            ] = task_result

    return results_by_task


def _process_branch_flight_results(
    state: TravelPlanningState,
    results_by_task: dict[ParallelTask, ParallelResult],
) -> TravelPlanningState:
    """Process flight branch results."""

    result = results_by_task.get(
        ParallelTask.FLIGHT_SEARCH
    )

    if result and result.completed and not result.error:
        state.plan.flights = result.result.get(
            "flight_options",
            result.result.get("flights", []),
        )

    return state


def _process_branch_accommodation_results(
    state: TravelPlanningState,
    results_by_task: dict[ParallelTask, ParallelResult],
) -> TravelPlanningState:
    """Process accommodation branch results."""

    result = results_by_task.get(
        ParallelTask.ACCOMMODATION
    )

    if result and result.completed and not result.error:
        state.plan.accommodation = result.result.get(
            "accommodations",
            result.result.get(
                "accommodation_options",
                [],
            ),
        )

    return state


def _process_branch_transportation_results(
    state: TravelPlanningState,
    results_by_task: dict[ParallelTask, ParallelResult],
) -> TravelPlanningState:
    """Process transportation branch results."""

    result = results_by_task.get(
        ParallelTask.TRANSPORTATION
    )

    if result and result.completed and not result.error:
        state.plan.transportation = result.result.get(
            "transportation_options",
            result.result.get(
                "transportation",
                {},
            ),
        )

    return state


def _process_branch_activity_results(
    state: TravelPlanningState,
    results_by_task: dict[ParallelTask, ParallelResult],
) -> TravelPlanningState:
    """Process activity branch results."""

    result = results_by_task.get(
        ParallelTask.ACTIVITIES
    )

    if result and result.completed and not result.error:
        state.plan.activities = result.result.get(
            "daily_itineraries",
            result.result.get(
                "activities",
                {},
            ),
        )

    return state


def _process_branch_budget_results(
    state: TravelPlanningState,
    results_by_task: dict[ParallelTask, ParallelResult],
) -> TravelPlanningState:
    """Process budget branch results."""

    result = results_by_task.get(
        ParallelTask.BUDGET
    )

    if result and result.completed and not result.error:
        state.plan.budget = result.result.get(
            "report",
            result.result.get(
                "budget",
                {},
            ),
        )

    return state


def _process_branch_errors(
    state: TravelPlanningState,
    results_by_task: dict[ParallelTask, ParallelResult],
) -> TravelPlanningState:
    """Add branch errors to the plan."""

    errors: list[str] = []

    for task_type, task_result in results_by_task.items():
        if task_result.error:
            errors.append(
                f"{task_type.value}: {task_result.error}"
            )

    if errors:
        _ensure_plan_initialized(state)
        state.plan.alerts.extend(errors)

    return state


def _update_workflow_stage(
    state: TravelPlanningState,
) -> TravelPlanningState:
    """Update the workflow stage after parallel processing."""

    state.current_stage = PARALLEL_SEARCH_COMPLETED

    return state