"""
Workflow orchestration for the travel planner system.

This module implements the high-level workflow orchestration for the travel planner,
integrating the LangGraph state graph with agent interactions and event handling.
"""

import traceback
from datetime import datetime
from typing import Any

from langgraph.errors import GraphInterrupt, InvalidUpdateError

from travel_planner.data.models import TravelPlan, TravelQuery, UserPreferences
from travel_planner.orchestration.core.agent_registry import register_default_agents
from travel_planner.orchestration.core.graph_builder import create_planning_graph
from travel_planner.orchestration.serialization.checkpoint import save_state_checkpoint
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.utils.logging import get_logger


logger = get_logger(__name__)


class TravelWorkflow:
    """
    Coordinates the entire travel planning workflow.

    This class is responsible for:
    1. Initializing all the agents
    2. Creating and managing the state graph
    3. Processing user queries and generating travel plans
    4. Handling interruptions and state updates
    """

    def __init__(self):
        """Initialize the travel planning workflow."""

        # Register all default agents
        register_default_agents()

        # Initialize the state graph
        self.graph = create_planning_graph()

    def process_query(
        self,
        query: str,
        preferences: UserPreferences | None = None,
    ) -> TravelPlan:
        """
        Process a travel query and generate a complete travel plan.

        This method provides a synchronous interface for testing. In production,
        the async version can be used for better performance.

        Args:
            query: User's travel query
            preferences: Optional user preferences

        Returns:
            Complete travel plan
        """

        logger.info(f"Processing travel query: {query}")

        # Create initial state
        initial_state = TravelPlanningState(
            query=TravelQuery(raw_query=query),
            preferences=preferences or UserPreferences(),
            conversation_history=[
                {
                    "role": "user",
                    "content": query,
                }
            ],
        )

        # Execute the graph with the initial state
        try:
            # Execute the full state graph workflow
            final_state = self._execute_graph(initial_state)

            return final_state.plan

        except InvalidUpdateError as e:
            # Handle validation errors
            logger.error(f"Validation error in workflow: {e!s}")

            initial_state.error = f"Validation error: {e!s}"

            initial_state.conversation_history.append(
                {
                    "role": "system",
                    "content": (
                        "Error: The travel query couldn't be processed due to "
                        f"validation issues. {e!s}"
                    ),
                }
            )

            return self._create_error_plan(
                e,
                "validation_error",
            )

        except GraphInterrupt as e:
            # Handle interruptions
            logger.info(f"Workflow interrupted: {e!s}")

            return self._handle_interruption(
                initial_state,
                e,
            )

        except Exception as e:
            # Handle any other unexpected errors
            logger.error(
                f"Unexpected error in travel planning workflow: {e!s}"
            )
            logger.error(traceback.format_exc())

            initial_state.error = f"Unexpected error: {e!s}"

            return self._create_error_plan(
                e,
                "unexpected_error",
            )

    async def process_query_async(
        self,
        query: str,
        preferences: UserPreferences | None = None,
    ) -> TravelPlan:
        """
        Process a travel query and generate a complete travel plan asynchronously.

        Args:
            query: User's travel query
            preferences: Optional user preferences

        Returns:
            Complete travel plan
        """

        logger.info(
            f"Processing travel query asynchronously: {query}"
        )

        # Create initial state
        initial_state = TravelPlanningState(
            query=TravelQuery(raw_query=query),
            preferences=preferences or UserPreferences(),
            conversation_history=[
                {
                    "role": "user",
                    "content": query,
                }
            ],
        )

        # Execute the graph with the initial state
        try:
            # Execute the full state graph workflow asynchronously
            final_state = await self._execute_graph_async(
                initial_state
            )

            return final_state.plan

        except InvalidUpdateError as e:
            # Handle validation errors
            logger.error(
                f"Validation error in async workflow: {e!s}"
            )

            initial_state.error = f"Validation error: {e!s}"

            initial_state.conversation_history.append(
                {
                    "role": "system",
                    "content": (
                        "Error: The travel query couldn't be processed due to "
                        f"validation issues. {e!s}"
                    ),
                }
            )

            return self._create_error_plan(
                e,
                "validation_error",
            )

        except GraphInterrupt as e:
            # Handle interruptions
            logger.info(
                f"Async workflow interrupted: {e!s}"
            )

            return self._handle_interruption(
                initial_state,
                e,
            )

        except Exception as e:
            # Handle any other unexpected errors
            logger.error(
                "Unexpected error in async travel planning workflow: "
                f"{e!s}"
            )
            logger.error(traceback.format_exc())

            initial_state.error = f"Unexpected error: {e!s}"

            return self._create_error_plan(
                e,
                "unexpected_error",
            )

    def _normalize_graph_result(
        self,
        result: Any,
        fallback_state: TravelPlanningState,
    ) -> TravelPlanningState:
        """
        Convert a LangGraph result into TravelPlanningState.

        LangGraph commonly returns a dictionary-like AddableValuesDict
        rather than the original Pydantic state object.
        """

        # Already the correct state type
        if isinstance(
            result,
            TravelPlanningState,
        ):
            return result

        # Normal LangGraph result
        if isinstance(
            result,
            dict,
        ):
            result_dict = dict(result)

            try:
                # Preferred conversion path
                return TravelPlanningState.model_validate(
                    result_dict
                )

            except Exception as exc:
                logger.warning(
                    "Could not directly convert LangGraph result "
                    f"to TravelPlanningState: {exc!s}"
                )

                # Fallback: update the original state
                normalized_state = fallback_state

                for key, value in result_dict.items():

                    if not hasattr(
                        normalized_state,
                        key,
                    ):
                        continue

                    try:
                        setattr(
                            normalized_state,
                            key,
                            value,
                        )

                    except Exception as field_exc:
                        logger.debug(
                            "Could not copy workflow field "
                            f"{key}: {field_exc!s}"
                        )

                return normalized_state

        # Unexpected result type
        logger.warning(
            "Unexpected workflow result type: "
            f"{type(result).__name__}"
        )

        return fallback_state

    def _execute_graph(
        self,
        initial_state: TravelPlanningState,
    ) -> TravelPlanningState:
        """
        Execute the state graph with the given initial state.

        This method executes the graph workflow synchronously using
        LangGraph's StateGraph.
        """

        logger.info(
            "Starting workflow graph execution"
        )

        try:
            # Execute graph
            result = self.graph.invoke(
                initial_state
            )

            # IMPORTANT:
            # LangGraph returns a dict-like AddableValuesDict.
            # Convert it to TravelPlanningState.
            final_state = self._normalize_graph_result(
                result,
                initial_state,
            )

            logger.info(
                "Workflow execution completed successfully"
            )

            return final_state

        except Exception as e:
            logger.error(
                f"Error executing graph: {e!s}"
            )
            raise

    async def _execute_graph_async(
        self,
        initial_state: TravelPlanningState,
    ) -> TravelPlanningState:
        """
        Execute the state graph asynchronously.

        LangGraph's ainvoke() may return an AddableValuesDict/dict-like
        object, so the result is explicitly converted back into
        TravelPlanningState.
        """

        logger.info(
            "Starting async workflow graph execution"
        )

        try:
            # Execute graph asynchronously
            result = await self.graph.ainvoke(
                initial_state
            )

            # IMPORTANT:
            # LangGraph returns a dict-like AddableValuesDict.
            # Convert it to TravelPlanningState.
            final_state = self._normalize_graph_result(
                result,
                initial_state,
            )

            logger.info(
                "Async workflow execution completed successfully"
            )

            return final_state

        except Exception as e:
            logger.error(
                f"Error executing async graph: {e!s}"
            )
            raise

    def _create_error_plan(
        self,
        error: Exception,
        error_type: str,
    ) -> TravelPlan:
        """
        Create a minimal travel plan with error information.
        """

        error_plan = TravelPlan()

        error_plan.metadata = {
            "error": str(error),
            "error_type": error_type,
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
        }

        error_plan.alerts = [
            f"Error: {error!s}"
        ]

        return error_plan

    def _handle_interruption(
        self,
        state: TravelPlanningState,
        interrupt_error: GraphInterrupt,
    ) -> TravelPlan:
        """
        Handle workflow interruption by creating a partial travel plan.
        """

        # Create a plan with whatever information we have so far
        partial_plan = (
            state.plan
            or TravelPlan()
        )

        # Add interruption metadata
        partial_plan.metadata = (
            partial_plan.metadata
            or {}
        )

        partial_plan.metadata.update(
            {
                "interrupted": True,
                "interruption_reason": str(
                    interrupt_error
                ),
                "timestamp": datetime.now().isoformat(),
                "current_stage": str(
                    state.current_stage
                ),
                "resumable": True,
                "checkpoint_id": (
                    state.state_checkpoint_id
                    or f"auto_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                ),
            }
        )

        # Add an alert about the interruption
        if not partial_plan.alerts:
            partial_plan.alerts = []

        partial_plan.alerts.append(
            "Note: This plan is incomplete due to an interruption: "
            f"{interrupt_error!s}"
        )

        # Store interrupted state
        if not state.state_checkpoint_id:
            state.state_checkpoint_id = (
                partial_plan.metadata["checkpoint_id"]
            )

            self._store_interrupted_state(
                state
            )

        return partial_plan

    def _store_interrupted_state(
        self,
        state: TravelPlanningState,
    ) -> None:
        """
        Store an interrupted state for later resumption.
        """

        checkpoint_id = save_state_checkpoint(
            state
        )

        logger.info(
            f"Stored interrupted state with checkpoint ID: "
            f"{checkpoint_id}"
        )

    def resume_workflow(
        self,
        checkpoint_id: str,
        updates: dict[str, Any] | None = None,
    ) -> TravelPlan:
        """
        Resume an interrupted workflow from a checkpoint.
        """

        from travel_planner.orchestration.serialization.checkpoint import (
            load_state_checkpoint,
        )

        logger.info(
            f"Resuming workflow from checkpoint: "
            f"{checkpoint_id}"
        )

        try:
            # Load the state from the checkpoint
            state = load_state_checkpoint(
                checkpoint_id
            )

            # Apply any updates
            if updates:
                for key, value in updates.items():

                    if hasattr(
                        state,
                        key,
                    ):
                        setattr(
                            state,
                            key,
                            value,
                        )

            # Mark as no longer interrupted
            state.interrupted = False
            state.interruption_reason = None

            # Add note to conversation history
            state.conversation_history.append(
                {
                    "role": "system",
                    "content": (
                        "Resuming workflow from stage: "
                        f"{state.current_stage}"
                    ),
                }
            )

            # Execute graph
            resumed_state = self._execute_graph(
                state
            )

            logger.info(
                "Successfully resumed and completed workflow"
            )

            return resumed_state.plan

        except Exception as e:
            logger.error(
                f"Error resuming workflow: {e!s}"
            )
            logger.error(
                traceback.format_exc()
            )

            error_plan = TravelPlan()

            error_plan.metadata = {
                "error": str(e),
                "error_type": "resume_error",
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "checkpoint_id": checkpoint_id,
            }

            error_plan.alerts = [
                f"Error resuming workflow: {e!s}"
            ]

            return error_plan

    async def resume_workflow_async(
        self,
        checkpoint_id: str,
        updates: dict[str, Any] | None = None,
    ) -> TravelPlan:
        """
        Resume an interrupted workflow from a checkpoint asynchronously.
        """

        from travel_planner.orchestration.serialization.checkpoint import (
            load_state_checkpoint,
        )

        logger.info(
            "Resuming workflow asynchronously from checkpoint: "
            f"{checkpoint_id}"
        )

        try:
            # Load state
            state = load_state_checkpoint(
                checkpoint_id
            )

            # Apply updates
            if updates:
                for key, value in updates.items():

                    if hasattr(
                        state,
                        key,
                    ):
                        setattr(
                            state,
                            key,
                            value,
                        )

            # Mark as no longer interrupted
            state.interrupted = False
            state.interruption_reason = None

            # Add note to conversation history
            state.conversation_history.append(
                {
                    "role": "system",
                    "content": (
                        "Resuming workflow from stage: "
                        f"{state.current_stage}"
                    ),
                }
            )

            # Execute graph asynchronously
            resumed_state = await self._execute_graph_async(
                state
            )

            logger.info(
                "Successfully resumed and completed workflow asynchronously"
            )

            return resumed_state.plan

        except Exception as e:
            logger.error(
                f"Error resuming workflow asynchronously: {e!s}"
            )
            logger.error(
                traceback.format_exc()
            )

            error_plan = TravelPlan()

            error_plan.metadata = {
                "error": str(e),
                "error_type": "resume_error",
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "checkpoint_id": checkpoint_id,
            }

            error_plan.alerts = [
                f"Error resuming workflow asynchronously: {e!s}"
            ]

            return error_plan