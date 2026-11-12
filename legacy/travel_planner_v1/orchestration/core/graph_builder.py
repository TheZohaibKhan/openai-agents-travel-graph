"""
Graph builder for the travel planner workflow.

This module defines the workflow state graph using LangGraph.
It connects the travel-planning nodes with conditional routing,
error handling, and human-intervention handling.
"""

from langgraph.graph import END, START, StateGraph

from travel_planner.orchestration.nodes.activity_planning import activity_planning
from travel_planner.orchestration.nodes.budget_management import budget_management
from travel_planner.orchestration.nodes.destination_research import destination_research
from travel_planner.orchestration.nodes.final_plan import generate_final_plan
from travel_planner.orchestration.nodes.parallel_search import (
    combine_search_results,
    create_parallel_search_branch,
)
from travel_planner.orchestration.nodes.query_analysis import query_analysis
from travel_planner.orchestration.routing.conditions import (
    continue_after_intervention,
    error_recoverable,
    has_error,
    needs_human_intervention,
    query_research_needed,
    recover_to_stage,
)
from travel_planner.orchestration.routing.error_recovery import (
    handle_error,
    handle_interruption,
)
from travel_planner.orchestration.states.planning_state import TravelPlanningState
from travel_planner.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------


def route_after_analyze_query(state: TravelPlanningState) -> str:
    """
    Decide what should happen after query analysis.
    """
    if has_error(state) == "true":
        return "handle_error"

    if needs_human_intervention(state) == "true":
        return "handle_interruption"

    return query_research_needed(state)


def route_after_research_destination(state: TravelPlanningState) -> str:
    """
    Decide what should happen after destination research.
    """
    if has_error(state) == "true":
        return "handle_error"

    if needs_human_intervention(state) == "true":
        return "handle_interruption"

    return "parallel_search"


def route_after_parallel_search(state: TravelPlanningState) -> str:
    """
    Decide what should happen after parallel search.
    """
    if has_error(state) == "true":
        return "handle_error"

    if needs_human_intervention(state) == "true":
        return "handle_interruption"

    return "combine_search_results"


def route_after_combine_search(state: TravelPlanningState) -> str:
    """
    Decide what should happen after combining search results.
    """
    if has_error(state) == "true":
        return "handle_error"

    if needs_human_intervention(state) == "true":
        return "handle_interruption"

    return "plan_activities"


def route_after_plan_activities(state: TravelPlanningState) -> str:
    """
    Decide what should happen after activity planning.
    """
    if has_error(state) == "true":
        return "handle_error"

    if needs_human_intervention(state) == "true":
        return "handle_interruption"

    return "manage_budget"


def route_after_manage_budget(state: TravelPlanningState) -> str:
    """
    Decide what should happen after budget management.
    """
    if has_error(state) == "true":
        return "handle_error"

    if needs_human_intervention(state) == "true":
        return "handle_interruption"

    return "generate_final_plan"


def route_after_final_plan(state: TravelPlanningState) -> str:
    """
    Decide what should happen after final plan generation.
    """
    if has_error(state) == "true":
        return "handle_error"

    if needs_human_intervention(state) == "true":
        return "handle_interruption"

    return "END"


def route_after_error(state: TravelPlanningState) -> str:
    """
    Decide whether an error can be recovered from.

    If recovery is possible, recover_to_stage() returns the stage
    to continue from. Otherwise the workflow terminates.
    """
    if error_recoverable(state) == "true":
        return recover_to_stage(state)

    return "END"


def route_after_interruption(state: TravelPlanningState) -> str:
    """
    Decide where the workflow should continue after interruption handling.
    """
    return continue_after_intervention(state)


# ---------------------------------------------------------------------------
# Graph creation
# ---------------------------------------------------------------------------


def create_planning_graph():
    """
    Create and compile the travel planning workflow.

    Returns:
        Compiled LangGraph workflow.
    """

    logger.info("Creating planning graph")

    workflow = StateGraph(TravelPlanningState)

    # ------------------------------------------------------------------
    # Core workflow nodes
    # ------------------------------------------------------------------

    workflow.add_node("analyze_query", query_analysis)

    workflow.add_node(
        "research_destination",
        destination_research,
    )

    workflow.add_node(
        "parallel_search",
        create_parallel_search_branch(),
    )

    workflow.add_node(
        "combine_search_results",
        combine_search_results,
    )

    workflow.add_node(
        "plan_activities",
        activity_planning,
    )

    workflow.add_node(
        "manage_budget",
        budget_management,
    )

    workflow.add_node(
        "generate_final_plan",
        generate_final_plan,
    )

    # ------------------------------------------------------------------
    # Error and interruption nodes
    # ------------------------------------------------------------------

    workflow.add_node(
        "handle_error",
        handle_error,
    )

    workflow.add_node(
        "handle_interruption",
        handle_interruption,
    )

    # ------------------------------------------------------------------
    # Workflow start
    # ------------------------------------------------------------------

    workflow.add_edge(
        START,
        "analyze_query",
    )

    # ------------------------------------------------------------------
    # Query analysis routing
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "analyze_query",
        route_after_analyze_query,
        {
            "research_destination": "research_destination",
            "parallel_search": "parallel_search",
            "handle_error": "handle_error",
            "handle_interruption": "handle_interruption",
        },
    )

    # ------------------------------------------------------------------
    # Destination research routing
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "research_destination",
        route_after_research_destination,
        {
            "parallel_search": "parallel_search",
            "handle_error": "handle_error",
            "handle_interruption": "handle_interruption",
        },
    )

    # ------------------------------------------------------------------
    # Parallel search routing
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "parallel_search",
        route_after_parallel_search,
        {
            "combine_search_results": "combine_search_results",
            "handle_error": "handle_error",
            "handle_interruption": "handle_interruption",
        },
    )

    # ------------------------------------------------------------------
    # Search result combination routing
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "combine_search_results",
        route_after_combine_search,
        {
            "plan_activities": "plan_activities",
            "handle_error": "handle_error",
            "handle_interruption": "handle_interruption",
        },
    )

    # ------------------------------------------------------------------
    # Activity planning routing
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "plan_activities",
        route_after_plan_activities,
        {
            "manage_budget": "manage_budget",
            "handle_error": "handle_error",
            "handle_interruption": "handle_interruption",
        },
    )

    # ------------------------------------------------------------------
    # Budget management routing
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "manage_budget",
        route_after_manage_budget,
        {
            "generate_final_plan": "generate_final_plan",
            "handle_error": "handle_error",
            "handle_interruption": "handle_interruption",
        },
    )

    # ------------------------------------------------------------------
    # Final plan routing
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "generate_final_plan",
        route_after_final_plan,
        {
            "END": END,
            "handle_error": "handle_error",
            "handle_interruption": "handle_interruption",
        },
    )

    # ------------------------------------------------------------------
    # Error recovery routing
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "handle_error",
        route_after_error,
        {
            "END": END,
            "analyze_query": "analyze_query",
            "research_destination": "research_destination",
            "parallel_search": "parallel_search",
            "combine_search_results": "combine_search_results",
            "plan_activities": "plan_activities",
            "manage_budget": "manage_budget",
            "generate_final_plan": "generate_final_plan",
            "handle_interruption": "handle_interruption",
        },
    )

    # ------------------------------------------------------------------
    # Human intervention routing
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "handle_interruption",
        route_after_interruption,
        {
            "analyze_query": "analyze_query",
            "research_destination": "research_destination",
            "parallel_search": "parallel_search",
            "combine_search_results": "combine_search_results",
            "plan_activities": "plan_activities",
            "manage_budget": "manage_budget",
            "generate_final_plan": "generate_final_plan",
            "END": END,
        },
    )

    logger.info("Planning graph created and compiled")

    return workflow.compile()