"""
Main entry point for the Travel Planner application.

This module serves as the application entry point, initializing the necessary
components and providing a CLI interface for interacting with the travel
planning system.
"""

import argparse
import asyncio
import json
import os
import sys
import traceback
from datetime import date, datetime
from typing import Any, TextIO

from travel_planner.agents.accommodation import AccommodationAgent
from travel_planner.agents.activity_planning import ActivityPlanningAgent
from travel_planner.agents.budget_management import BudgetManagementAgent
from travel_planner.agents.destination_research import DestinationResearchAgent
from travel_planner.agents.flight_search import FlightSearchAgent
from travel_planner.agents.research_tools import DestinationResearchTools
from travel_planner.agents.transportation import TransportationAgent
from travel_planner.config import TravelPlannerConfig, initialize_config
from travel_planner.data.models import (
    Accommodation,
    DailyItinerary,
    Flight,
    TravelPlan,
    TravelQuery,
)
from travel_planner.data.setup import initialize_database
from travel_planner.data.supabase import SupabaseClient
from travel_planner.orchestration.state_graph import TravelPlanningState
from travel_planner.orchestration.workflow import TravelWorkflow
from travel_planner.utils.logging import get_logger, setup_logging
from travel_planner.utils.rate_limiting import initialize_rate_limiting


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAVE_COMMAND_LENGTH = 5


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def setup_argparse() -> argparse.ArgumentParser:
    """
    Set up the argument parser for the CLI.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(
        description="AI Travel Planning System powered by Gemini"
    )

    # -----------------------------------------------------------------------
    # System configuration
    # -----------------------------------------------------------------------

    system_group = parser.add_argument_group(
        "System Configuration"
    )

    system_group.add_argument(
        "--log-level",
        type=str,
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ],
        default="INFO",
        help="Set the logging level",
    )

    system_group.add_argument(
        "--log-file",
        type=str,
        help="Path to write log file (optional)",
    )

    system_group.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file",
    )

    system_group.add_argument(
        "--headless",
        action="store_true",
        help="Run browser automation in headless mode",
    )

    system_group.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching for browser automation and API calls",
    )

    system_group.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize Supabase database tables if they don't exist",
    )

    system_group.add_argument(
        "--disable-rate-limits",
        action="store_true",
        help="Disable API rate limiting",
    )

    system_group.add_argument(
        "--rate-limit-config",
        type=str,
        help="Path to custom rate limit configuration file",
    )

    # -----------------------------------------------------------------------
    # Query mode
    # -----------------------------------------------------------------------

    query_group = parser.add_argument_group(
        "Query Mode"
    )

    query_group.add_argument(
        "--query",
        type=str,
        help="Initial travel query to start planning",
    )

    query_group.add_argument(
        "--origin",
        type=str,
        help="Origin location (e.g., city or airport code)",
    )

    query_group.add_argument(
        "--destination",
        type=str,
        help="Destination location",
    )

    query_group.add_argument(
        "--departure-date",
        type=str,
        help="Departure date (YYYY-MM-DD)",
    )

    query_group.add_argument(
        "--return-date",
        type=str,
        help="Return date (YYYY-MM-DD)",
    )

    query_group.add_argument(
        "--travelers",
        type=int,
        default=1,
        help="Number of travelers",
    )

    query_group.add_argument(
        "--budget",
        type=str,
        help="Budget range (e.g., '1000-2000')",
    )

    query_group.add_argument(
        "--preferences-file",
        type=str,
        help="Path to JSON file with detailed user preferences",
    )

    # -----------------------------------------------------------------------
    # Output options
    # -----------------------------------------------------------------------

    output_group = parser.add_argument_group(
        "Output Options"
    )

    output_group.add_argument(
        "--save-to",
        type=str,
        help="Save the travel plan to specified file path",
    )

    output_group.add_argument(
        "--format",
        type=str,
        choices=[
            "json",
            "text",
            "html",
            "pdf",
        ],
        default="json",
        help="Output format for saved travel plans",
    )

    output_group.add_argument(
        "--save-to-db",
        action="store_true",
        help="Save the travel plan to the Supabase database",
    )

    return parser


# ---------------------------------------------------------------------------
# LangGraph result conversion
# ---------------------------------------------------------------------------

def normalize_workflow_result(
    graph_result: Any,
    fallback_state: TravelPlanningState,
) -> TravelPlanningState:
    """
    Convert the result returned by LangGraph into TravelPlanningState.

    Current LangGraph versions return a dictionary-like AddableValuesDict
    from graph execution. The rest of this application expects
    TravelPlanningState attribute access.

    Args:
        graph_result:
            Result returned by the LangGraph workflow.

        fallback_state:
            Original TravelPlanningState used to start execution.

    Returns:
        TravelPlanningState instance.
    """

    # Already normalized.
    if isinstance(
        graph_result,
        TravelPlanningState,
    ):
        return graph_result

    # LangGraph normally returns a dict-like object.
    if isinstance(
        graph_result,
        dict,
    ):

        result_dict = dict(
            graph_result
        )

        try:
            # Pydantic validation is the cleanest conversion path.
            return TravelPlanningState.model_validate(
                result_dict
            )

        except Exception as exc:

            logger.warning(
                "Could not directly convert LangGraph result "
                f"to TravelPlanningState: {exc!s}"
            )

            # ----------------------------------------------------------------
            # Fallback:
            # copy known fields into the original state.
            # ----------------------------------------------------------------

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

    # Unexpected result type.
    logger.warning(
        "Unexpected workflow result type: "
        f"{type(graph_result).__name__}"
    )

    return fallback_state


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

async def run_interactive_mode(
    args: argparse.Namespace,
) -> None:
    """
    Run the travel planner in interactive mode.

    Args:
        args: Command-line arguments.
    """

    logger.info(
        "Starting interactive travel planning session"
    )

    workflow = create_travel_workflow(
        args
    )

    print(
        "\n=== AI Travel Planning System ==="
    )

    print(
        "Welcome! Describe your travel plans and preferences."
    )

    print(
        "Type 'exit' or 'quit' to end the session."
    )

    print(
        "Type 'help' for available commands.\n"
    )

    state = TravelPlanningState()

    travel_plan = None

    while True:

        user_input = input(
            "\nYou: "
        ).strip()

        # ---------------------------------------------------------------
        # Exit
        # ---------------------------------------------------------------

        if user_input.lower() in [
            "exit",
            "quit",
            "q",
            "bye",
        ]:

            print(
                "\nThank you for using the Travel Planner. Goodbye!"
            )

            break

        # ---------------------------------------------------------------
        # Help
        # ---------------------------------------------------------------

        if user_input.lower() == "help":

            display_help()

            continue

        # ---------------------------------------------------------------
        # Save
        # ---------------------------------------------------------------

        if user_input.lower().startswith(
            "save"
        ):

            if travel_plan:

                path = (
                    user_input[
                        SAVE_COMMAND_LENGTH:
                    ].strip()
                    if len(user_input)
                    > SAVE_COMMAND_LENGTH
                    else None
                )

                await save_travel_plan(
                    travel_plan,
                    path,
                    args.format,
                )

                print(
                    "\nTravel Planner: Travel plan saved successfully."
                )

            else:

                print(
                    "\nTravel Planner: No travel plan available to save."
                )

            continue

        try:

            # -----------------------------------------------------------
            # Parse user query
            # -----------------------------------------------------------

            query = TravelQuery(
                raw_query=user_input
            )

            state.query = query

            # -----------------------------------------------------------
            # Execute workflow
            # -----------------------------------------------------------

            print(
                "\nTravel Planner: Processing your request. "
                "This may take a moment..."
            )

            graph_result = (
                await workflow.execute(
                    state
                )
            )

            state = normalize_workflow_result(
                graph_result,
                state,
            )

            # -----------------------------------------------------------
            # Store generated plan
            # -----------------------------------------------------------

            if (
                state
                and state.travel_plan
            ):

                travel_plan = (
                    state.travel_plan
                )

            # -----------------------------------------------------------
            # Display
            # -----------------------------------------------------------

            if (
                state
                and state.travel_plan
            ):

                display_travel_plan(
                    state.travel_plan
                )

            elif (
                state
                and state.error
            ):

                print(
                    "\nTravel Planner: "
                    f"I encountered an issue: {state.error}"
                )

            else:

                print(
                    "\nTravel Planner: "
                    "I couldn't complete your travel planning request."
                )

        except Exception as exc:

            logger.error(
                "Error in interactive session: "
                f"{exc!s}\n"
                f"{traceback.format_exc()}"
            )

            print(
                "\nTravel Planner: "
                f"I'm sorry, I encountered an error: {exc!s}"
            )


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

def display_help() -> None:
    """
    Display available commands.
    """

    print(
        "\nAvailable commands:"
    )

    print(
        "  help                 - Display this help message"
    )

    print(
        "  save [path]          - Save the current travel plan to a file"
    )

    print(
        "  exit, quit, q, bye   - Exit the application"
    )

    print(
        "\nTravel query examples:"
    )

    print(
        "  I want to visit Tokyo for a week in October"
    )

    print(
        "  Plan a budget trip from New York to London "
        "from June 10-17 for 2 people"
    )

    print(
        "  Find family-friendly activities in Paris "
        "for a 3-day weekend"
    )


# ---------------------------------------------------------------------------
# Travel plan display
# ---------------------------------------------------------------------------

def display_travel_plan(
    plan: TravelPlan,
) -> None:
    """
    Display a travel plan in a readable format.

    Args:
        plan: Travel plan to display.
    """

    print(
        f"\n=== Travel Plan to {plan.destination} ===\n"
    )

    # -----------------------------------------------------------------------
    # Flights
    # -----------------------------------------------------------------------

    if plan.flights:

        print(
            "Flights:"
        )

        for i, flight in enumerate(
            plan.flights
        ):

            print(
                f"  {i + 1}. "
                f"{flight.airline}: "
                f"{flight.departure_location} "
                f"to "
                f"{flight.arrival_location}"
            )

            print(
                f"     "
                f"{flight.departure_time.strftime('%Y-%m-%d %H:%M')} "
                f"- "
                f"{flight.arrival_time.strftime('%Y-%m-%d %H:%M')}"
            )

            print(
                f"     Price: ${flight.price:.2f}"
            )

        print()

    # -----------------------------------------------------------------------
    # Accommodation
    # -----------------------------------------------------------------------

    if plan.accommodations:

        print(
            "Accommodations:"
        )

        for i, acc in enumerate(
            plan.accommodations
        ):

            print(
                f"  {i + 1}. "
                f"{acc.name} "
                f"({acc.type})"
            )

            print(
                f"     "
                f"{acc.check_in_date.strftime('%Y-%m-%d')} "
                f"to "
                f"{acc.check_out_date.strftime('%Y-%m-%d')}"
            )

            print(
                f"     Price: "
                f"${acc.price_per_night:.2f} per night "
                f"(Total: ${acc.total_price:.2f})"
            )

        print()

    # -----------------------------------------------------------------------
    # Daily itinerary
    # -----------------------------------------------------------------------

    if plan.daily_itinerary:

        print(
            "Daily Itinerary:"
        )

        for i, day in enumerate(
            plan.daily_itinerary
        ):

            print(
                f"  Day {i + 1} "
                f"({day.date.strftime('%Y-%m-%d')}):"
            )

            for j, activity in enumerate(
                day.activities
            ):

                print(
                    f"     {j + 1}. "
                    f"{activity.name} "
                    f"({activity.time_start.strftime('%H:%M')} "
                    f"- "
                    f"{activity.time_end.strftime('%H:%M')})"
                )

                print(
                    f"        {activity.description}"
                )

                if activity.price > 0:

                    print(
                        f"        Price: "
                        f"${activity.price:.2f}"
                    )

            print()

    # -----------------------------------------------------------------------
    # Budget
    # -----------------------------------------------------------------------

    if plan.budget_summary:

        print(
            "Budget Summary:"
        )

        print(
            "  Total Estimated Cost: "
            f"${plan.budget_summary.total_cost:.2f}"
        )

        if plan.budget_summary.breakdown:

            print(
                "  Breakdown:"
            )

            for category, amount in (
                plan.budget_summary.breakdown.items()
            ):

                print(
                    f"     {category}: "
                    f"${amount:.2f}"
                )

        print()


# ---------------------------------------------------------------------------
# Save travel plan
# ---------------------------------------------------------------------------

async def save_travel_plan(
    plan: TravelPlan,
    file_path: str | None = None,
    format_type: str = "json",
) -> None:
    """
    Save a travel plan to a file.
    """

    file_path = _prepare_file_path(
        plan,
        file_path,
        format_type,
    )

    if format_type == "json":

        _save_as_json(
            plan,
            file_path,
        )

    elif format_type == "text":

        _save_as_text(
            plan,
            file_path,
        )

    elif format_type == "html":

        _save_as_html(
            plan,
            file_path,
        )

    elif format_type == "pdf":

        _handle_pdf_save(
            file_path
        )

    logger.info(
        f"Travel plan saved to {file_path}"
    )


def _prepare_file_path(
    plan: TravelPlan,
    file_path: str | None,
    format_type: str,
) -> str:
    """
    Prepare the file path for saving.
    """

    if not file_path:

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        destination = str(
            plan.destination
        ).replace(
            " ",
            "_",
        )

        file_path = (
            f"travel_plan_"
            f"{destination}_"
            f"{timestamp}."
            f"{format_type}"
        )

    directory = os.path.dirname(
        file_path
    )

    if directory and not os.path.exists(
        directory
    ):

        os.makedirs(
            directory
        )

    return file_path


def _save_as_json(
    plan: TravelPlan,
    file_path: str,
) -> None:
    """
    Save travel plan as JSON.
    """

    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            plan.model_dump(
                mode="json"
            ),
            file,
            indent=2,
        )


def _handle_pdf_save(
    file_path: str,
) -> None:
    """
    Handle PDF save request.
    """

    logger.error(
        "PDF export not implemented yet"
    )

    raise NotImplementedError(
        "PDF export not implemented yet"
    )


def _save_as_text(
    plan: TravelPlan,
    file_path: str,
) -> None:
    """
    Save travel plan as plain text.
    """

    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as file:

        _write_text_header(
            file,
            plan,
        )

        _write_text_overview(
            file,
            plan,
        )

        _write_text_flights(
            file,
            plan,
        )

        _write_text_accommodations(
            file,
            plan,
        )

        _write_text_activities(
            file,
            plan,
        )

        _write_text_budget(
            file,
            plan,
        )

        _write_text_recommendations(
            file,
            plan,
        )

        _write_text_alerts(
            file,
            plan,
        )


# ---------------------------------------------------------------------------
# Text export helpers
# ---------------------------------------------------------------------------

def _write_text_header(
    file: TextIO,
    plan: TravelPlan,
) -> None:

    destination_name = (
        plan.destination.get(
            "name",
            "Unknown",
        )
        if isinstance(
            plan.destination,
            dict,
        )
        else plan.destination
    )

    file.write(
        f"TRAVEL PLAN TO "
        f"{str(destination_name).upper()}\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )


def _write_text_overview(
    file: TextIO,
    plan: TravelPlan,
) -> None:

    if not plan.overview:
        return

    file.write(
        "OVERVIEW\n"
    )

    file.write(
        "-" * 8 + "\n"
    )

    file.write(
        f"{plan.overview}\n\n"
    )


def _write_text_flights(
    file: TextIO,
    plan: TravelPlan,
) -> None:

    if not plan.flights:
        return

    file.write(
        "FLIGHTS\n"
    )

    file.write(
        "-" * 7 + "\n"
    )

    for i, flight in enumerate(
        plan.flights
    ):

        file.write(
            f"{i + 1}. "
            f"{flight.airline}: "
            f"{flight.flight_number}\n"
        )

        file.write(
            f"   From: "
            f"{flight.departure_airport} "
            f"- To: "
            f"{flight.arrival_airport}\n"
        )

        file.write(
            f"   Departure: "
            f"{flight.departure_time.strftime('%Y-%m-%d %H:%M')}\n"
        )

        file.write(
            f"   Arrival: "
            f"{flight.arrival_time.strftime('%Y-%m-%d %H:%M')}\n"
        )

        file.write(
            f"   Class: "
            f"{flight.travel_class.value}\n"
        )

        file.write(
            f"   Price: "
            f"{flight.currency} "
            f"{flight.price:.2f}\n"
        )

        _write_text_flight_layovers(
            file,
            flight,
        )

        file.write(
            f"   Duration: "
            f"{flight.duration_minutes} minutes\n"
        )

        if flight.booking_link:

            file.write(
                f"   Booking: "
                f"{flight.booking_link}\n"
            )

        file.write("\n")


def _write_text_flight_layovers(
    file: TextIO,
    flight: Flight,
) -> None:

    if not flight.layovers:
        return

    file.write(
        f"   Layovers: "
        f"{len(flight.layovers)}\n"
    )

    for j, layover in enumerate(
        flight.layovers
    ):

        file.write(
            f"      {j + 1}. "
            f"{layover.get('airport', 'Unknown')} "
            f"- Duration: "
            f"{layover.get('duration_minutes', 0)} min\n"
        )


def _write_text_accommodations(
    file: TextIO,
    plan: TravelPlan,
) -> None:

    # The project currently uses both `accommodation`
    # and `accommodations` in different places.
    accommodations = getattr(
        plan,
        "accommodation",
        None,
    )

    if accommodations is None:

        accommodations = getattr(
            plan,
            "accommodations",
            None,
        )

    if not accommodations:
        return

    file.write(
        "ACCOMMODATIONS\n"
    )

    file.write(
        "-" * 14 + "\n"
    )

    for i, acc in enumerate(
        accommodations
    ):

        acc_type = (
            acc.type.value
            if hasattr(
                acc.type,
                "value",
            )
            else acc.type
        )

        file.write(
            f"{i + 1}. "
            f"{acc.name} "
            f"({acc_type})\n"
        )

        file.write(
            f"   Address: "
            f"{acc.address}\n"
        )

        if acc.rating:

            file.write(
                f"   Rating: "
                f"{acc.rating}/5\n"
            )

        file.write(
            f"   Check-in: "
            f"{acc.check_in_time} "
            f"- Check-out: "
            f"{acc.check_out_time}\n"
        )

        file.write(
            f"   Price per night: "
            f"{acc.currency} "
            f"{acc.price_per_night:.2f}\n"
        )

        file.write(
            f"   Total price: "
            f"{acc.currency} "
            f"{acc.total_price:.2f}\n"
        )

        if acc.amenities:

            file.write(
                f"   Amenities: "
                f"{', '.join(acc.amenities)}\n"
            )

        if acc.booking_link:

            file.write(
                f"   Booking: "
                f"{acc.booking_link}\n"
            )

        file.write("\n")


def _write_text_activities(
    file: TextIO,
    plan: TravelPlan,
) -> None:

    if not plan.activities:
        return

    file.write(
        "DAILY ITINERARY\n"
    )

    file.write(
        "-" * 15 + "\n"
    )

    for _day_key, day in (
        plan.activities.items()
    ):

        file.write(
            f"Day {day.day_number} "
            f"- "
            f"{day.date.strftime('%Y-%m-%d')}\n"
        )

        _write_text_day_weather(
            file,
            day,
        )

        _write_text_day_activities(
            file,
            day,
        )

        _write_text_day_transportation(
            file,
            day,
        )

        if day.notes:

            file.write(
                f"   Notes: "
                f"{day.notes}\n"
            )

        file.write("\n")


def _write_text_day_weather(
    file: TextIO,
    day: DailyItinerary,
) -> None:

    if not day.weather_forecast:
        return

    weather = (
        day.weather_forecast
    )

    file.write(
        f"   Weather: "
        f"{weather.get('description', 'N/A')}, "
        f"{weather.get('temperature', 'N/A')}°C\n"
    )


def _write_text_day_activities(
    file: TextIO,
    day: DailyItinerary,
) -> None:

    if not day.activities:
        return

    for i, activity in enumerate(
        day.activities
    ):

        duration_hours = (
            activity.duration_minutes // 60
        )

        duration_mins = (
            activity.duration_minutes % 60
        )

        duration = (
            f"{duration_hours}h "
            f"{duration_mins}m"
            if duration_hours > 0
            else f"{duration_mins}m"
        )

        activity_type = (
            activity.type.value
            if hasattr(
                activity.type,
                "value",
            )
            else activity.type
        )

        file.write(
            f"   {i + 1}. "
            f"{activity.name} "
            f"({activity_type})\n"
        )

        file.write(
            f"      "
            f"{activity.description}\n"
        )

        file.write(
            f"      Location: "
            f"{activity.location}\n"
        )

        file.write(
            f"      Duration: "
            f"{duration}\n"
        )

        if activity.cost:

            file.write(
                f"      Cost: "
                f"{activity.currency} "
                f"{activity.cost:.2f}\n"
            )

        if activity.booking_required:

            booking_info = (
                f" - {activity.booking_link}"
                if activity.booking_link
                else ""
            )

            file.write(
                f"      Booking required"
                f"{booking_info}\n"
            )


def _write_text_day_transportation(
    file: TextIO,
    day: DailyItinerary,
) -> None:

    if not day.transportation:
        return

    file.write(
        "   Transportation:\n"
    )

    for i, transport in enumerate(
        day.transportation
    ):

        transport_type = (
            transport.type.value
            if hasattr(
                transport.type,
                "value",
            )
            else transport.type
        )

        file.write(
            f"      {i + 1}. "
            f"{transport_type}: "
            f"{transport.description}\n"
        )

        if transport.cost:

            file.write(
                f"         Cost: "
                f"{transport.currency} "
                f"{transport.cost:.2f}\n"
            )


def _write_text_budget(
    file: TextIO,
    plan: TravelPlan,
) -> None:

    if not plan.budget:
        return

    file.write(
        "BUDGET SUMMARY\n"
    )

    file.write(
        "-" * 14 + "\n"
    )

    file.write(
        f"Total budget: "
        f"{plan.budget.currency} "
        f"{plan.budget.total_budget:.2f}\n"
    )

    file.write(
        f"Spent: "
        f"{plan.budget.currency} "
        f"{plan.budget.spent:.2f}\n"
    )

    file.write(
        f"Remaining: "
        f"{plan.budget.currency} "
        f"{plan.budget.remaining:.2f}\n"
    )

    _write_text_budget_breakdown(
        file,
        plan,
    )

    _write_text_saving_recommendations(
        file,
        plan,
    )

    file.write("\n")


def _write_text_budget_breakdown(
    file: TextIO,
    plan: TravelPlan,
) -> None:

    if (
        not plan.budget
        or not plan.budget.breakdown
    ):
        return

    file.write(
        "Breakdown:\n"
    )

    for category, amount in (
        plan.budget.breakdown.items()
    ):

        file.write(
            f"   {category}: "
            f"{plan.budget.currency} "
            f"{amount:.2f}\n"
        )


def _write_text_saving_recommendations(
    file: TextIO,
    plan: TravelPlan,
) -> None:

    if (
        not plan.budget
        or not plan.budget.saving_recommendations
    ):
        return

    file.write(
        "Saving Recommendations:\n"
    )

    for i, recommendation in enumerate(
        plan.budget.saving_recommendations
    ):

        file.write(
            f"   {i + 1}. "
            f"{recommendation}\n"
        )


def _write_text_recommendations(
    file: TextIO,
    plan: TravelPlan,
) -> None:

    if not plan.recommendations:
        return

    file.write(
        "RECOMMENDATIONS\n"
    )

    file.write(
        "-" * 16 + "\n"
    )

    for i, recommendation in enumerate(
        plan.recommendations
    ):

        file.write(
            f"{i + 1}. "
            f"{recommendation}\n"
        )

    file.write("\n")


def _write_text_alerts(
    file: TextIO,
    plan: TravelPlan,
) -> None:

    if not plan.alerts:
        return

    file.write(
        "IMPORTANT ALERTS\n"
    )

    file.write(
        "-" * 16 + "\n"
    )

    for i, alert in enumerate(
        plan.alerts
    ):

        file.write(
            f"{i + 1}. "
            f"{alert}\n"
        )


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------

def _save_as_html(
    plan: TravelPlan,
    file_path: str,
) -> None:

    destination_name = (
        _get_destination_name(
            plan
        )
    )

    html_parts = [
        _generate_html_header(
            destination_name
        ),
        _generate_html_overview(
            plan
        ),
        _generate_html_flights(
            plan
        ),
        _generate_html_accommodations(
            plan
        ),
        _generate_html_activities(
            plan
        ),
        _generate_html_budget(
            plan
        ),
        _generate_html_recommendations(
            plan
        ),
        _generate_html_alerts(
            plan
        ),
        _generate_html_footer(),
    ]

    html_content = "".join(
        html_parts
    )

    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            html_content
        )


def _get_destination_name(
    plan: TravelPlan,
) -> str:

    return (
        plan.destination.get(
            "name",
            "Unknown",
        )
        if isinstance(
            plan.destination,
            dict,
        )
        else str(
            plan.destination
        )
    )


def _generate_html_header(
    destination_name: str,
) -> str:

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>
Travel Plan to {destination_name}
</title>

<style>

body {{
    font-family: Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 1000px;
    margin: 0 auto;
    padding: 20px;
}}

h1,
h2,
h3 {{
    color: #2c3e50;
}}

h1 {{
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
}}

h2 {{
    border-bottom: 1px solid #bdc3c7;
    padding-bottom: 5px;
    margin-top: 30px;
}}

.card {{
    background: #f9f9f9;
    border-radius: 5px;
    padding: 15px;
    margin-bottom: 20px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}}

.card h3 {{
    margin-top: 0;
    color: #3498db;
}}

.flight,
.accommodation,
.activity,
.transportation {{
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 1px dashed #ddd;
}}

.day {{
    margin-bottom: 30px;
    padding: 15px;
    background: #f5f5f5;
    border-radius: 5px;
}}

.day-header {{
    background: #3498db;
    color: white;
    padding: 10px;
    margin: -15px -15px 15px -15px;
    border-radius: 5px 5px 0 0;
}}

.price {{
    font-weight: bold;
    color: #e74c3c;
}}

.alert {{
    background-color: #f8d7da;
    color: #721c24;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}}

.recommendation {{
    background-color: #d4edda;
    color: #155724;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 20px;
}}

table,
th,
td {{
    border: 1px solid #ddd;
}}

th,
td {{
    padding: 12px;
    text-align: left;
}}

th {{
    background-color: #f2f2f2;
}}

</style>

</head>

<body>

<h1>
Travel Plan to {destination_name}
</h1>
"""


def _generate_html_overview(
    plan: TravelPlan,
) -> str:

    if not plan.overview:
        return ""

    return f"""
<div class="card">

<h2>
Overview
</h2>

<p>
{plan.overview}
</p>

</div>
"""


def _generate_html_flights(
    plan: TravelPlan,
) -> str:

    if not plan.flights:
        return ""

    html = """
<h2>
Flights
</h2>
"""

    for flight in plan.flights:

        html += _generate_html_flight_card(
            flight
        )

    return html


def _generate_html_flight_card(
    flight: Flight,
) -> str:

    html = f"""
<div class="card flight">

<h3>
{flight.airline} - Flight {flight.flight_number}
</h3>

<p>
<strong>From:</strong>
{flight.departure_airport}
&rarr;
<strong>To:</strong>
{flight.arrival_airport}
</p>

<p>
<strong>Departure:</strong>
{flight.departure_time.strftime("%Y-%m-%d %H:%M")}
</p>

<p>
<strong>Arrival:</strong>
{flight.arrival_time.strftime("%Y-%m-%d %H:%M")}
</p>

<p>
<strong>Class:</strong>
{flight.travel_class.value}
</p>

<p>
<strong>Duration:</strong>
{flight.duration_minutes // 60}h
{flight.duration_minutes % 60}m
</p>
"""

    if flight.layovers:

        html += _generate_html_flight_layovers(
            flight
        )

    html += f"""
<p class="price">

<strong>
Price:
</strong>

{flight.currency}
{flight.price:.2f}

</p>
"""

    if flight.booking_link:

        html += f"""
<p>

<a
href="{flight.booking_link}"
target="_blank"
>
Booking Link
</a>

</p>
"""

    html += """
</div>
"""

    return html


def _generate_html_flight_layovers(
    flight: Flight,
) -> str:

    html = f"""
<p>
<strong>
Layovers:
</strong>
{len(flight.layovers)}
</p>

<ul>
"""

    for layover in flight.layovers:

        html += f"""
<li>

{layover.get("airport", "Unknown")}

-

Duration:
{layover.get("duration_minutes", 0)}
minutes

</li>
"""

    html += """
</ul>
"""

    return html


def _generate_html_accommodations(
    plan: TravelPlan,
) -> str:

    accommodations = getattr(
        plan,
        "accommodation",
        None,
    )

    if accommodations is None:

        accommodations = getattr(
            plan,
            "accommodations",
            None,
        )

    if not accommodations:
        return ""

    html = """
<h2>
Accommodations
</h2>
"""

    for acc in accommodations:

        html += _generate_html_accommodation_card(
            acc
        )

    return html


def _generate_html_accommodation_card(
    acc: Accommodation,
) -> str:

    acc_type = (
        acc.type.value
        if hasattr(
            acc.type,
            "value",
        )
        else acc.type
    )

    html = f"""
<div class="card accommodation">

<h3>
{acc.name}
({acc_type})
</h3>

<p>
<strong>
Address:
</strong>

{acc.address}
</p>
"""

    if acc.rating:

        html += f"""
<p>
<strong>
Rating:
</strong>
{acc.rating}/5
</p>
"""

    html += f"""
<p>

<strong>
Check-in:
</strong>

{acc.check_in_time}

-

<strong>
Check-out:
</strong>

{acc.check_out_time}

</p>

<p class="price">

<strong>
Price per night:
</strong>

{acc.currency}
{acc.price_per_night:.2f}

</p>

<p class="price">

<strong>
Total price:
</strong>

{acc.currency}
{acc.total_price:.2f}

</p>
"""

    if acc.amenities:

        html += f"""
<p>

<strong>
Amenities:
</strong>

{", ".join(acc.amenities)}

</p>
"""

    if acc.booking_link:

        html += f"""
<p>

<a
href="{acc.booking_link}"
target="_blank"
>
Booking Link
</a>

</p>
"""

    html += """
</div>
"""

    return html


def _generate_html_activities(
    plan: TravelPlan,
) -> str:

    if not plan.activities:
        return ""

    html = """
<h2>
Daily Itinerary
</h2>
"""

    sorted_days = sorted(
        plan.activities.items(),
        key=lambda item: item[1].day_number,
    )

    for _day_key, day in sorted_days:

        html += _generate_html_day_card(
            day
        )

    return html


def _generate_html_day_card(
    day: DailyItinerary,
) -> str:

    html = f"""
<div class="day">

<div class="day-header">

<h3>

Day {day.day_number}
-
{day.date.strftime("%Y-%m-%d")}

</h3>
"""

    if day.weather_forecast:

        weather = (
            day.weather_forecast
        )

        html += f"""
<p>

<strong>
Weather:
</strong>

{weather.get("description", "N/A")},

{weather.get("temperature", "N/A")}°C

</p>
"""

    html += """
</div>
"""

    html += _generate_html_day_activities(
        day
    )

    html += _generate_html_day_transportation(
        day
    )

    if day.notes:

        html += f"""
<div class="notes">

<h4>
Notes
</h4>

<p>
{day.notes}
</p>

</div>
"""

    html += """
</div>
"""

    return html


def _generate_html_day_activities(
    day: DailyItinerary,
) -> str:

    if not day.activities:
        return ""

    html = """
<h4>
Activities
</h4>
"""

    for activity in day.activities:

        duration_hours = (
            activity.duration_minutes // 60
        )

        duration_mins = (
            activity.duration_minutes % 60
        )

        duration = (
            f"{duration_hours}h "
            f"{duration_mins}m"
            if duration_hours > 0
            else f"{duration_mins}m"
        )

        activity_type = (
            activity.type.value
            if hasattr(
                activity.type,
                "value",
            )
            else activity.type
        )

        html += f"""
<div class="activity card">

<h5>
{activity.name}
({activity_type})
</h5>

<p>
{activity.description}
</p>

<p>

<strong>
Location:
</strong>

{activity.location}

</p>

<p>

<strong>
Duration:
</strong>

{duration}

</p>
"""

        if activity.cost:

            html += f"""
<p class="price">

<strong>
Cost:
</strong>

{activity.currency}
{activity.cost:.2f}

</p>
"""

        if activity.booking_required:

            html += """
<p>
<strong>
Booking required
</strong>
</p>
"""

            if activity.booking_link:

                html += f"""
<p>

<a
href="{activity.booking_link}"
target="_blank"
>
Booking Link
</a>

</p>
"""

        html += """
</div>
"""

    return html


def _generate_html_day_transportation(
    day: DailyItinerary,
) -> str:

    if not day.transportation:
        return ""

    html = """
<h4>
Transportation
</h4>
"""

    for transport in day.transportation:

        transport_type = (
            transport.type.value
            if hasattr(
                transport.type,
                "value",
            )
            else transport.type
        )

        html += f"""
<div class="transportation card">

<h5>
{transport_type}
</h5>

<p>
{transport.description}
</p>
"""

        if transport.cost:

            html += f"""
<p class="price">

<strong>
Cost:
</strong>

{transport.currency}
{transport.cost:.2f}

</p>
"""

        if transport.duration_minutes:

            dur_hours = (
                transport.duration_minutes // 60
            )

            dur_mins = (
                transport.duration_minutes % 60
            )

            dur_str = (
                f"{dur_hours}h "
                f"{dur_mins}m"
                if dur_hours > 0
                else f"{dur_mins}m"
            )

            html += f"""
<p>

<strong>
Duration:
</strong>

{dur_str}

</p>
"""

        html += """
</div>
"""

    return html


def _generate_html_budget(
    plan: TravelPlan,
) -> str:

    if not plan.budget:
        return ""

    html = f"""
<h2>
Budget Summary
</h2>

<div class="card">

<p>

<strong>
Total Budget:
</strong>

{plan.budget.currency}
{plan.budget.total_budget:.2f}

</p>

<p>

<strong>
Spent:
</strong>

{plan.budget.currency}
{plan.budget.spent:.2f}

</p>

<p>

<strong>
Remaining:
</strong>

{plan.budget.currency}
{plan.budget.remaining:.2f}

</p>
"""

    html += _generate_html_budget_breakdown(
        plan
    )

    html += _generate_html_saving_recommendations(
        plan
    )

    html += """
</div>
"""

    return html


def _generate_html_budget_breakdown(
    plan: TravelPlan,
) -> str:

    if (
        not plan.budget
        or not plan.budget.breakdown
    ):
        return ""

    html = """
<h3>
Budget Breakdown
</h3>

<table>

<tr>

<th>
Category
</th>

<th>
Amount
</th>

</tr>
"""

    for category, amount in (
        plan.budget.breakdown.items()
    ):

        html += f"""
<tr>

<td>
{category}
</td>

<td>
{plan.budget.currency}
{amount:.2f}
</td>

</tr>
"""

    html += """
</table>
"""

    return html


def _generate_html_saving_recommendations(
    plan: TravelPlan,
) -> str:

    if (
        not plan.budget
        or not plan.budget.saving_recommendations
    ):
        return ""

    html = """
<h3>
Saving Recommendations
</h3>

<ul>
"""

    for recommendation in (
        plan.budget.saving_recommendations
    ):

        html += f"""
<li>
{recommendation}
</li>
"""

    html += """
</ul>
"""

    return html


def _generate_html_recommendations(
    plan: TravelPlan,
) -> str:

    if not plan.recommendations:
        return ""

    html = """
<h2>
Recommendations
</h2>
"""

    for recommendation in (
        plan.recommendations
    ):

        html += f"""
<div class="recommendation">

{recommendation}

</div>
"""

    return html


def _generate_html_alerts(
    plan: TravelPlan,
) -> str:

    if not plan.alerts:
        return ""

    html = """
<h2>
Important Alerts
</h2>
"""

    for alert in plan.alerts:

        html += f"""
<div class="alert">

{alert}

</div>
"""

    return html


def _generate_html_footer() -> str:

    return f"""
<footer>

<p>

<small>

Generated by AI Travel Planning System on

{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

</small>

</p>

</footer>

</body>

</html>
"""


# ---------------------------------------------------------------------------
# Query mode
# ---------------------------------------------------------------------------

async def run_query_mode(
    args: argparse.Namespace,
) -> TravelPlanningState:
    """
    Run the travel planner with a single query.

    LangGraph currently returns an AddableValuesDict/dict-like result.
    This function normalizes it back to TravelPlanningState.
    """

    query_text = args.query or ""

    logger.info(
        f"Starting travel planning for query: {query_text}"
    )

    # -----------------------------------------------------------------------
    # Build query
    # -----------------------------------------------------------------------

    query = build_travel_query_from_args(
        args
    )

    # -----------------------------------------------------------------------
    # Initial state
    # -----------------------------------------------------------------------

    state = TravelPlanningState(
        query=query
    )

    # -----------------------------------------------------------------------
    # Workflow
    # -----------------------------------------------------------------------

    workflow = create_travel_workflow(
        args
    )

    logger.info(
        "Executing travel planning workflow"
    )

    # -----------------------------------------------------------------------
    # Execute graph
    # -----------------------------------------------------------------------

    graph_result = (
        await workflow._execute_graph_async(
            state
        )
    )

    # -----------------------------------------------------------------------
    # IMPORTANT FIX:
    #
    # LangGraph returns AddableValuesDict.
    # Convert it into TravelPlanningState.
    # -----------------------------------------------------------------------

    final_state = normalize_workflow_result(
        graph_result,
        state,
    )

    # -----------------------------------------------------------------------
    # Save to file
    # -----------------------------------------------------------------------

    if (
        args.save_to
        and final_state
        and final_state.travel_plan
    ):

        await save_travel_plan(
            final_state.travel_plan,
            args.save_to,
            args.format,
        )

        logger.info(
            f"Travel plan saved to {args.save_to}"
        )

    # -----------------------------------------------------------------------
    # Save to database
    # -----------------------------------------------------------------------

    if (
        args.save_to_db
        and final_state
        and final_state.travel_plan
    ):

        await save_to_database(
            final_state.travel_plan
        )

        logger.info(
            "Travel plan saved to database"
        )

    return final_state


# ---------------------------------------------------------------------------
# Build TravelQuery
# ---------------------------------------------------------------------------

def build_travel_query_from_args(
    args: argparse.Namespace,
) -> TravelQuery:
    """
    Build a TravelQuery from CLI arguments.

    Args:
        args: Command-line arguments.

    Returns:
        Constructed TravelQuery object.
    """

    query_params: dict[str, Any] = {
        "raw_query": args.query or ""
    }

    # -----------------------------------------------------------------------
    # Origin
    # -----------------------------------------------------------------------

    if args.origin:

        query_params[
            "origin"
        ] = args.origin

    # -----------------------------------------------------------------------
    # Destination
    # -----------------------------------------------------------------------

    if args.destination:

        query_params[
            "destination"
        ] = args.destination

    # -----------------------------------------------------------------------
    # Departure date
    # -----------------------------------------------------------------------

    if args.departure_date:

        try:

            query_params[
                "departure_date"
            ] = date.fromisoformat(
                args.departure_date
            )

        except ValueError:

            logger.warning(
                f"Invalid departure date format: "
                f"{args.departure_date}. "
                f"Expected YYYY-MM-DD."
            )

    # -----------------------------------------------------------------------
    # Return date
    # -----------------------------------------------------------------------

    if args.return_date:

        try:

            query_params[
                "return_date"
            ] = date.fromisoformat(
                args.return_date
            )

        except ValueError:

            logger.warning(
                f"Invalid return date format: "
                f"{args.return_date}. "
                f"Expected YYYY-MM-DD."
            )

    # -----------------------------------------------------------------------
    # Travelers
    # -----------------------------------------------------------------------

    if args.travelers:

        query_params[
            "travelers"
        ] = args.travelers

    # -----------------------------------------------------------------------
    # Budget
    # -----------------------------------------------------------------------

    if args.budget:

        try:

            min_val, max_val = map(
                float,
                args.budget.split("-"),
            )

            query_params[
                "budget_range"
            ] = {
                "min": min_val,
                "max": max_val,
            }

        except ValueError:

            logger.warning(
                f"Invalid budget format: "
                f"{args.budget}. "
                f"Expected format like '1000-2000'."
            )

    # -----------------------------------------------------------------------
    # Preferences
    # -----------------------------------------------------------------------

    if (
        args.preferences_file
        and os.path.exists(
            args.preferences_file
        )
    ):

        try:

            with open(
                args.preferences_file,
                "r",
                encoding="utf-8",
            ) as file:

                preferences = json.load(
                    file
                )

            query_params[
                "requirements"
            ] = preferences

        except Exception as exc:

            logger.warning(
                "Error loading preferences file: "
                f"{exc!s}"
            )

    return TravelQuery(
        **query_params
    )


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

async def save_to_database(
    travel_plan: TravelPlan,
) -> None:
    """
    Save a travel plan to Supabase.
    """

    try:

        supabase_client = (
            SupabaseClient()
        )

        travel_plan_data = (
            travel_plan.model_dump(
                mode="json"
            )
        )

        result = (
            await supabase_client
            .from_("travel_plans")
            .insert(
                travel_plan_data
            )
            .execute()
        )

        logger.info(
            "Travel plan saved to database "
            f"with ID: "
            f"{result.data[0]['id'] if result.data else 'unknown'}"
        )

    except Exception as exc:

        logger.error(
            f"Error saving to database: {exc!s}"
        )

        raise


# ---------------------------------------------------------------------------
# Workflow creation
# ---------------------------------------------------------------------------

def create_travel_workflow(
    args: argparse.Namespace,
) -> TravelWorkflow:
    """
    Create the configured travel planning workflow.
    """

    return TravelWorkflow()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code.
    """

    try:

        args = (
            _parse_arguments_and_setup_basic_logging()
        )

        system_config = (
            await _initialize_system_configuration(
                args
            )
        )

        if not system_config:

            return 1

        return await _run_selected_mode(
            args
        )

    except TravelPlannerConfig.ConfigurationError as exc:

        return _handle_configuration_error(
            exc
        )

    except KeyboardInterrupt:

        return _handle_keyboard_interrupt()

    except FileNotFoundError as exc:

        return _handle_file_not_found_error(
            exc
        )

    except Exception as exc:

        return _handle_general_exception(
            exc
        )


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _parse_arguments_and_setup_basic_logging(
) -> argparse.Namespace:

    parser = setup_argparse()

    args = parser.parse_args()

    setup_logging(
        log_level=args.log_level or "INFO",
        log_file=args.log_file,
    )

    logger.info(
        "Starting AI Travel Planning System"
    )

    return args


async def _initialize_system_configuration(
    args: argparse.Namespace,
) -> TravelPlannerConfig | None:
    """
    Initialize system configuration.
    """

    system_config = initialize_config(
        custom_config_path=args.config,
        validate=True,
        raise_on_error=False,
    )

    setup_logging(
        log_level=(
            args.log_level
            or system_config.system.log_level
        ),
        log_file=args.log_file,
    )

    _initialize_rate_limiting(
        args
    )

    # -----------------------------------------------------------------------
    # Gemini / Supabase validation
    # -----------------------------------------------------------------------

    if not system_config.api.validate():

        _display_missing_api_keys_error()

        return None

    # -----------------------------------------------------------------------
    # Database
    # -----------------------------------------------------------------------

    if (
        args.init_db
        and not await _initialize_database_if_requested(
            system_config
        )
    ):

        return None

    return system_config


def _initialize_rate_limiting(
    args: argparse.Namespace,
) -> None:
    """
    Initialize external API rate limiting.
    """

    if args.disable_rate_limits:

        logger.warning(
            "API rate limiting is disabled."
        )

        return

    logger.info(
        "Initializing API rate limiting..."
    )

    if (
        args.rate_limit_config
        and os.path.exists(
            args.rate_limit_config
        )
    ):

        _load_custom_rate_limits(
            args.rate_limit_config
        )

    else:

        initialize_rate_limiting()


def _load_custom_rate_limits(
    config_path: str,
) -> None:

    try:

        from travel_planner.utils.rate_limiting import (
            update_rate_limits_from_config,
        )

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as file:

            rate_limit_config = json.load(
                file
            )

        update_rate_limits_from_config(
            rate_limit_config
        )

        logger.info(
            f"Loaded custom rate limits from "
            f"{config_path}"
        )

    except Exception as exc:

        logger.error(
            f"Error loading rate limit configuration: "
            f"{exc!s}"
        )

        print(
            "\nError loading rate limit configuration: "
            f"{exc!s}"
        )


# ---------------------------------------------------------------------------
# Missing API key message
# ---------------------------------------------------------------------------

def _display_missing_api_keys_error() -> None:
    """
    Display missing configuration information.

    The project has been migrated from OpenAI to Gemini.
    """

    print(
        "\nERROR: Missing required API configuration."
    )

    print(
        "\nPlease configure your .env file with:"
    )

    print(
        "  - GEMINI_API_KEY"
    )

    print(
        "  - SUPABASE_URL"
    )

    print(
        "  - SUPABASE_KEY"
    )

    print(
        "\nYou can set these in the .env file "
        "in the project root directory."
    )

    print(
        "See the README.md for setup instructions.\n"
    )


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------

async def _initialize_database_if_requested(
    system_config: TravelPlannerConfig,
) -> bool:

    logger.info(
        "Initializing Supabase database..."
    )

    try:

        db_success = await initialize_database(
            system_config
        )

        if not db_success:

            logger.error(
                "Failed to initialize Supabase database"
            )

            print(
                "\nERROR: Failed to initialize Supabase database."
            )

            return False

        logger.info(
            "Supabase database initialized successfully"
        )

        return True

    except Exception as exc:

        logger.error(
            f"Error initializing Supabase database: {exc!s}"
        )

        print(
            "\nERROR: Failed to initialize Supabase database: "
            f"{exc!s}"
        )

        return False


# ---------------------------------------------------------------------------
# Selected mode
# ---------------------------------------------------------------------------

async def _run_selected_mode(
    args: argparse.Namespace,
) -> int:
    """
    Run query or interactive mode.
    """

    query_mode = any(
        [
            args.query,
            args.origin,
            args.destination,
            args.departure_date,
            args.return_date,
            args.budget,
            args.preferences_file,
        ]
    )

    if query_mode:

        return await _run_query_mode_with_display(
            args
        )

    await run_interactive_mode(
        args
    )

    return 0


# ---------------------------------------------------------------------------
# Query mode display
# ---------------------------------------------------------------------------

async def _run_query_mode_with_display(
    args: argparse.Namespace,
) -> int:
    """
    Run query mode and display the generated travel plan.

    The returned object from run_query_mode() is always normalized
    to TravelPlanningState.
    """

    final_state = await run_query_mode(
        args
    )

    # -----------------------------------------------------------------------
    # Travel plan generated
    # -----------------------------------------------------------------------

    if (
        final_state
        and final_state.travel_plan
    ):

        display_travel_plan(
            final_state.travel_plan
        )

        return 0

    # -----------------------------------------------------------------------
    # Workflow error
    # -----------------------------------------------------------------------

    if (
        final_state
        and final_state.error
    ):

        print(
            f"Error: {final_state.error}"
        )

        return 1

    # -----------------------------------------------------------------------
    # Nothing generated
    # -----------------------------------------------------------------------

    print(
        "No travel plan was generated."
    )

    return 1


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

def _handle_configuration_error(
    exc: Exception,
) -> int:

    logger.error(
        f"Configuration error: {exc!s}"
    )

    print(
        f"\nConfiguration Error: {exc!s}"
    )

    print(
        "Please check your environment variables "
        "and configuration settings."
    )

    return 1


def _handle_keyboard_interrupt() -> int:

    logger.info(
        "Travel planning session interrupted by user"
    )

    print(
        "\nTravel planning session interrupted. Goodbye!"
    )

    return 0


def _handle_file_not_found_error(
    exc: Exception,
) -> int:

    logger.error(
        f"File not found: {exc!s}"
    )

    print(
        f"\nError: {exc!s}"
    )

    return 1


def _handle_general_exception(
    exc: Exception,
) -> int:
    """Handle unexpected application errors gracefully."""

    error_text = str(exc)
    error_upper = error_text.upper()

    # Gemini free-tier quota errors are expected to happen when the
    # application has used all available GenerateContent requests.
    # Do not hide the real reason behind a generic exception message.
    if (
        "RESOURCE_EXHAUSTED" in error_upper
        and (
            "QUOTA" in error_upper
            or "FREE_TIER" in error_upper
            or "GENERATE_REQUESTS" in error_upper
        )
    ):
        logger.error(
            "Gemini API quota exhausted: \n"
            f"{traceback.format_exc()}"
        )

        print(
            "\n=================================================="
        )
        print(
            "Gemini API free-tier quota has been exhausted."
        )
        print(
            "The Travel Planner cannot make additional Gemini requests right now."
        )
        print(
            "Please wait for the quota to reset or use another Gemini API project with available quota."
        )
        print(
            "==================================================\n"
        )

        return 1

    # Handle all other unexpected exceptions normally.
    logger.error(
        "Error in main function: "
        f"{error_text}\n"
        f"{traceback.format_exc()}"
    )

    print(
        f"\nError: {error_text}"
    )

    print(
        "An unexpected error occurred. "
        "Please check the logs for more details."
    )

    return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    sys.exit(
        asyncio.run(
            main()
        )
    )