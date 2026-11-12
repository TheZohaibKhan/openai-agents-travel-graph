"""
Base node implementation for the travel planning workflow.

This module defines base functionality for workflow nodes, including
common execution patterns and error handling to reduce code duplication.
"""

from collections.abc import Callable
from typing import Any

from travel_planner.data.models import (
    AgentTaskParams,
    NodeFunctionParams,
    TravelPlan,
)
from travel_planner.orchestration.states.planning_state import (
    TravelPlanningState,
)
from travel_planner.utils.logging import get_logger


logger = get_logger(__name__)


def execute_agent_task(
    state: TravelPlanningState,
    agent: Any,
    task_name: str,
    complete_stage: Any,
    result_formatter: Callable[[Any], str],
    result_processor: Callable[[TravelPlanningState, Any], None] | None = None,
) -> TravelPlanningState:
    """
    Generic function to execute an agent task and update workflow state.

    This function accepts the keyword-based interface used by the
    individual workflow nodes.

    Args:
        state:
            Current travel planning state.

        agent:
            Agent instance responsible for the task.

        task_name:
            Name of the workflow task.

        complete_stage:
            Workflow stage to set after successful completion.

        result_formatter:
            Function used to convert the agent result into a
            conversation-history message.

        result_processor:
            Optional function used to process the agent result and
            update the travel plan.

    Returns:
        Updated travel planning state.
    """

    logger.info(
        f"Executing {task_name} with "
        f"{agent.__class__.__name__}"
    )

    try:

        # ---------------------------------------------------------
        # Execute agent
        # ---------------------------------------------------------

        result = run_agent_sync(
            agent,
            state,
        )

        # ---------------------------------------------------------
        # Initialize travel plan if required
        # ---------------------------------------------------------

        if state.plan is None:
            state.plan = TravelPlan()

        # ---------------------------------------------------------
        # Update workflow stage
        # ---------------------------------------------------------

        state.update_stage(
            complete_stage
        )

        # ---------------------------------------------------------
        # Add formatted result to conversation history
        # ---------------------------------------------------------

        message = result_formatter(
            result
        )

        state.conversation_history.append(
            {
                "role": "system",
                "content": message,
            }
        )

        # ---------------------------------------------------------
        # Process result
        # ---------------------------------------------------------

        if result_processor:
            result_processor(
                state,
                result,
            )

        # ---------------------------------------------------------
        # Store task result
        # ---------------------------------------------------------

        state.add_task_result(
            task_name,
            result,
        )

        logger.info(
            f"Completed {task_name} successfully"
        )

        return state

    except Exception as e:

        logger.error(
            f"Error in {task_name}: {e!s}"
        )

        state.mark_error(
            f"Error during {task_name}: {e!s}"
        )

        # Check whether the workflow should retry.
        if state.should_retry(
            task_name
        ):
            logger.info(
                f"Will retry {task_name} "
                f"(attempt "
                f"{state.retry_count.get(task_name, 0)})"
            )

        return state


def run_agent_sync(
    agent: Any,
    state: TravelPlanningState,
) -> Any:
    """
    Run an asynchronous agent from a synchronous LangGraph node.

    The original user query is always preserved as the latest user
    message.

    Structured workflow context is supplied as a separate system
    message so agents that use _get_latest_user_input() receive
    the actual travel request instead of the generated context.
    """

    import asyncio
    import threading

    # ---------------------------------------------------------
    # Extract the ORIGINAL user query
    # ---------------------------------------------------------

    query = getattr(
        state,
        "query",
        None,
    )

    raw_query = (
        getattr(
            query,
            "raw_query",
            "",
        )
        if query
        else ""
    )

    raw_query = str(
        raw_query or ""
    ).strip()

    if not raw_query:
        raw_query = "Plan this trip."

    # ---------------------------------------------------------
    # Build conversation from existing workflow history
    # ---------------------------------------------------------

    conversation = list(
        getattr(
            state,
            "conversation_history",
            [],
        )
        or []
    )

    # ---------------------------------------------------------
    # Remove previously generated context messages
    #
    # These messages begin with:
    #
    # Current travel planning context:
    #
    # We remove them so the same context is not duplicated
    # every time another LangGraph node executes.
    # ---------------------------------------------------------

    cleaned_conversation: list[dict[str, Any]] = []

    for message in conversation:

        if not isinstance(
            message,
            dict,
        ):
            continue

        content = str(
            message.get(
                "content",
                "",
            )
            or ""
        )

        if content.startswith(
            "Current travel planning context:"
        ):
            continue

        cleaned_conversation.append(
            message
        )

    conversation = cleaned_conversation

    # ---------------------------------------------------------
    # Build structured travel context
    # ---------------------------------------------------------

    context_parts: list[str] = []

    if query is not None:

        for field in (
            "origin",
            "destination",
            "departure_date",
            "return_date",
            "travelers",
            "budget",
        ):

            value = getattr(
                query,
                field,
                None,
            )

            if value is not None and value != "":
                context_parts.append(
                    f"{field}: {value}"
                )

    # ---------------------------------------------------------
    # Add preferences
    # ---------------------------------------------------------

    preferences = getattr(
        state,
        "preferences",
        None,
    )

    if preferences is not None:

        context_parts.append(
            f"preferences: {preferences}"
        )

    # ---------------------------------------------------------
    # Add structured context BEFORE the final user message
    #
    # This is important because several agents determine their
    # input using the latest user message.
    # ---------------------------------------------------------

    if context_parts:

        conversation.append(
            {
                "role": "system",
                "content": (
                    "Current travel planning context:\n"
                    + "\n".join(
                        context_parts
                    )
                ),
            }
        )

    # ---------------------------------------------------------
    # ALWAYS make the original query the final user message
    # ---------------------------------------------------------

    conversation.append(
        {
            "role": "user",
            "content": raw_query,
        }
    )

    # ---------------------------------------------------------
    # Execute asynchronous agent
    # ---------------------------------------------------------

    async def execute():
        return await agent.run(
            conversation
        )

    # ---------------------------------------------------------
    # Case 1:
    # No event loop is currently running
    # ---------------------------------------------------------

    try:

        asyncio.get_running_loop()

    except RuntimeError:

        return asyncio.run(
            execute()
        )

    # ---------------------------------------------------------
    # Case 2:
    # An event loop is already running
    #
    # Run the async agent inside a separate thread with its
    # own event loop.
    # ---------------------------------------------------------

    result: list[Any] = []
    error: list[Exception] = []

    def runner():

        try:

            result.append(
                asyncio.run(
                    execute()
                )
            )

        except Exception as exc:

            error.append(
                exc
            )

    thread = threading.Thread(
        target=runner,
        daemon=True,
    )

    thread.start()
    thread.join()

    # ---------------------------------------------------------
    # Propagate agent errors
    # ---------------------------------------------------------

    if error:
        raise error[0]

    # ---------------------------------------------------------
    # Return agent result
    # ---------------------------------------------------------

    if result:
        return result[0]

    raise RuntimeError(
        "Agent execution completed without returning a result."
    )


def create_node_function(
    params: NodeFunctionParams,
) -> Callable[
    [TravelPlanningState],
    TravelPlanningState,
]:
    """
    Factory function to create workflow node functions.

    Args:
        params:
            Node configuration containing the agent class,
            task name, workflow stage, result fields, and
            message template.

    Returns:
        A LangGraph-compatible node function.
    """

    def result_formatter(
        result: dict[str, Any],
    ) -> str:
        """
        Format the agent result for conversation history.
        """

        if not isinstance(
            result,
            dict,
        ):
            return (
                f"{params.task_name} completed."
            )

        data = result.get(
            params.result_field,
            [],
        )

        if isinstance(
            data,
            list,
        ):
            count = len(data)

        elif data:
            count = 1

        else:
            count = 0

        return params.message_template.format(
            count=count
        )

    def result_processor(
        state: TravelPlanningState,
        result: dict[str, Any],
    ) -> None:
        """
        Process agent results and update the travel plan.
        """

        if (
            state.plan
            and isinstance(
                result,
                dict,
            )
            and params.result_field in result
        ):

            setattr(
                state.plan,
                params.plan_field,
                result[
                    params.result_field
                ],
            )

    def node_function(
        state: TravelPlanningState,
    ) -> TravelPlanningState:
        """
        Execute the configured agent task.
        """

        agent = params.agent_class()

        return execute_agent_task(
            state=state,
            agent=agent,
            task_name=params.task_name,
            complete_stage=params.complete_stage,
            result_formatter=result_formatter,
            result_processor=result_processor,
        )

    # ---------------------------------------------------------
    # Preserve useful function metadata
    # ---------------------------------------------------------

    node_function.__name__ = (
        params.task_name
    )

    node_function.__doc__ = (
        f"Execute {params.task_name} using "
        f"{params.agent_class.__name__}."
    )

    return node_function