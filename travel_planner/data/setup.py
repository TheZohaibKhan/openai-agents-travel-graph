"""
Setup script for initializing the database schema and tables.

This script handles the initialization of the Supabase database,
applying PostgreSQL migrations, and setting up required tables
and indexes.

Database migrations are executed through a direct PostgreSQL
connection instead of an arbitrary Supabase execute_sql RPC.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from psycopg import connect

from travel_planner.config import TravelPlannerConfig, initialize_config
from travel_planner.data.supabase import SupabaseClient
from travel_planner.utils.logging import get_logger, setup_logging


# ---------------------------------------------------------------------
# LOGGER
# ---------------------------------------------------------------------

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# ENVIRONMENT HELPERS
# ---------------------------------------------------------------------


def get_database_connection():
    """
    Create a direct PostgreSQL connection to the Supabase database.

    The connection uses separate environment variables instead of a
    PostgreSQL URI. This prevents special characters in the database
    password from being interpreted as part of the hostname.

    Required environment variables:

        SUPABASE_DB_HOST
        SUPABASE_DB_PORT
        SUPABASE_DB_NAME
        SUPABASE_DB_USER
        SUPABASE_DB_PASSWORD

    Returns:
        psycopg.Connection
    """

    host = os.getenv("SUPABASE_DB_HOST")
    port = os.getenv("SUPABASE_DB_PORT", "5432")
    database = os.getenv("SUPABASE_DB_NAME", "postgres")
    user = os.getenv("SUPABASE_DB_USER", "postgres")
    password = os.getenv("SUPABASE_DB_PASSWORD")

    if not host:
        raise ValueError(
            "SUPABASE_DB_HOST is not configured."
        )

    if not password:
        raise ValueError(
            "SUPABASE_DB_PASSWORD is not configured."
        )

    try:
        port_number = int(port)
    except ValueError as exc:
        raise ValueError(
            "SUPABASE_DB_PORT must be a valid integer."
        ) from exc

    return connect(
        host=host,
        port=port_number,
        dbname=database,
        user=user,
        password=password,
    )


def validate_database_environment() -> list[str]:
    """
    Validate required PostgreSQL environment variables.

    Returns:
        A list containing the names of missing variables.
    """

    required_variables = [
        "SUPABASE_DB_HOST",
        "SUPABASE_DB_PORT",
        "SUPABASE_DB_NAME",
        "SUPABASE_DB_USER",
        "SUPABASE_DB_PASSWORD",
    ]

    missing_variables = [
        variable
        for variable in required_variables
        if not os.getenv(variable)
    ]

    return missing_variables


# ---------------------------------------------------------------------
# SQL EXECUTION
# ---------------------------------------------------------------------


def execute_sql_sync(sql: str) -> None:
    """
    Execute SQL using a direct PostgreSQL connection.

    Each migration runs inside a PostgreSQL transaction. If the SQL
    fails, the transaction is rolled back automatically when the
    connection context exits.
    """

    if not sql or not sql.strip():
        logger.warning(
            "Received empty SQL. Nothing to execute."
        )
        return

    logger.debug(
        "Connecting to PostgreSQL database..."
    )

    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql)

        connection.commit()

    logger.debug(
        "SQL execution completed successfully."
    )


# ---------------------------------------------------------------------
# DATABASE INITIALIZATION
# ---------------------------------------------------------------------


async def initialize_database(
    config: TravelPlannerConfig,
    reset: bool = False,
) -> bool:
    """
    Initialize the Supabase database with the required schema.

    Args:
        config:
            Application configuration.

        reset:
            Whether to reset the database before applying migrations.

    Returns:
        True if initialization succeeds, otherwise False.
    """

    try:

        # -------------------------------------------------------------
        # VALIDATE SUPABASE API CONFIGURATION
        # -------------------------------------------------------------

        if not config.api.supabase_url:
            raise ValueError(
                "SUPABASE_URL is missing."
            )

        if not config.api.supabase_key:
            raise ValueError(
                "SUPABASE_KEY is missing."
            )

        # -------------------------------------------------------------
        # INITIALIZE SUPABASE CLIENT
        # -------------------------------------------------------------

        # This client is used by the application for normal Supabase
        # operations. PostgreSQL migrations are handled separately
        # through psycopg.
        SupabaseClient(
            url=config.api.supabase_url,
            key=config.api.supabase_key,
        )

        logger.info(
            "Supabase configuration validated."
        )

        # -------------------------------------------------------------
        # VALIDATE POSTGRESQL CONFIGURATION
        # -------------------------------------------------------------

        missing_variables = (
            validate_database_environment()
        )

        if missing_variables:
            raise ValueError(
                "Missing PostgreSQL environment variables: "
                + ", ".join(missing_variables)
            )

        logger.info(
            "PostgreSQL configuration validated."
        )

        # -------------------------------------------------------------
        # TEST POSTGRESQL CONNECTION
        # -------------------------------------------------------------

        logger.info(
            "Testing PostgreSQL connection..."
        )

        with get_database_connection() as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    "SELECT version();"
                )

                version = cursor.fetchone()[0]

        logger.info(
            "PostgreSQL connection successful."
        )

        logger.debug(
            f"PostgreSQL version: {version}"
        )

        # -------------------------------------------------------------
        # APPLY MIGRATIONS
        # -------------------------------------------------------------

        await apply_migrations(
            reset=reset
        )

        logger.info(
            "Supabase database initialization "
            "completed successfully."
        )

        return True

    except Exception as e:

        logger.error(
            f"Error initializing Supabase database: {e}"
        )

        return False


# ---------------------------------------------------------------------
# MIGRATION HANDLING
# ---------------------------------------------------------------------


async def apply_migrations(
    reset: bool = False,
) -> None:
    """
    Apply database migrations from the migrations directory.

    Args:
        reset:
            Whether to reset the database before migrations.
    """

    migrations_dir = (
        Path(__file__).parent / "migrations"
    )

    # -------------------------------------------------------------
    # CHECK MIGRATION DIRECTORY
    # -------------------------------------------------------------

    if not migrations_dir.exists():

        logger.warning(
            "Migrations directory does not exist: "
            f"{migrations_dir}"
        )

        return

    # -------------------------------------------------------------
    # RESET DATABASE
    # -------------------------------------------------------------

    if reset:

        logger.warning(
            "=================================================="
        )

        logger.warning(
            "RESETTING DATABASE"
        )

        logger.warning(
            "ALL DATABASE DATA WILL BE LOST!"
        )

        logger.warning(
            "=================================================="
        )

        reset_sql = """
        DROP VIEW IF EXISTS api_usage_stats CASCADE;

        DROP VIEW IF EXISTS travel_plans_paginated CASCADE;

        DROP VIEW IF EXISTS travel_plan_summaries CASCADE;

        DROP FUNCTION IF EXISTS check_rate_limit(TEXT) CASCADE;

        DROP FUNCTION IF EXISTS get_travel_plan_history(UUID) CASCADE;

        DROP FUNCTION IF EXISTS create_travel_plan_version(
            UUID,
            TEXT,
            JSONB
        ) CASCADE;

        DROP FUNCTION IF EXISTS search_travel_plans(TEXT) CASCADE;

        DROP TABLE IF EXISTS api_rate_limits CASCADE;

        DROP TABLE IF EXISTS api_requests CASCADE;

        DROP TABLE IF EXISTS travel_plans CASCADE;

        DROP TABLE IF EXISTS user_preferences CASCADE;

        DROP TABLE IF EXISTS travel_queries CASCADE;

        DROP TABLE IF EXISTS users CASCADE;
        """

        execute_sql_sync(
            reset_sql
        )

        logger.info(
            "Database reset completed successfully."
        )

    # -------------------------------------------------------------
    # FIND SQL MIGRATIONS
    # -------------------------------------------------------------

    migration_files = sorted(
        migrations_dir.glob("*.sql")
    )

    if not migration_files:

        logger.warning(
            "No SQL migration files found."
        )

        return

    logger.info(
        f"Found {len(migration_files)} migration files."
    )

    # -------------------------------------------------------------
    # APPLY EACH MIGRATION
    # -------------------------------------------------------------

    for migration_file in migration_files:

        logger.info(
            f"Applying migration: "
            f"{migration_file.name}"
        )

        try:

            # -----------------------------------------------------
            # READ SQL FILE
            # -----------------------------------------------------

            with open(
                migration_file,
                "r",
                encoding="utf-8",
            ) as file:

                sql = file.read()

            # -----------------------------------------------------
            # EMPTY MIGRATION CHECK
            # -----------------------------------------------------

            if not sql.strip():

                logger.warning(
                    f"Migration {migration_file.name} "
                    "is empty. Skipping."
                )

                continue

            # -----------------------------------------------------
            # EXECUTE MIGRATION
            # -----------------------------------------------------

            execute_sql_sync(
                sql
            )

            logger.info(
                f"Successfully applied migration: "
                f"{migration_file.name}"
            )

        except Exception as e:

            logger.error(
                f"Error applying migration "
                f"{migration_file.name}: {e}"
            )

            raise


# ---------------------------------------------------------------------
# TEST DATA
# ---------------------------------------------------------------------


async def create_test_data(
    client: SupabaseClient,
) -> None:
    """
    Create test data in the database.

    Args:
        client:
            Initialized Supabase client.
    """

    logger.info(
        "Creating test data..."
    )

    # -------------------------------------------------------------
    # CREATE TEST USER
    # -------------------------------------------------------------

    user_result = (
        await client.client
        .table("users")
        .insert(
            {
                "email": "test@example.com"
            }
        )
        .execute()
    )

    if (
        not user_result.data
        or len(user_result.data) == 0
    ):

        logger.error(
            "Failed to create test user."
        )

        return

    user_id = (
        user_result.data[0]["id"]
    )

    logger.info(
        f"Created test user with ID: {user_id}"
    )

    # -------------------------------------------------------------
    # CREATE TEST TRAVEL QUERY
    # -------------------------------------------------------------

    query_result = (
        await client.client
        .table("travel_queries")
        .insert(
            {
                "user_id": user_id,

                "raw_query": (
                    "I want to visit Tokyo "
                    "for a week in October"
                ),

                "destination": "Tokyo",

                "origin": "New York",

                "departure_date": "2025-10-01",

                "return_date": "2025-10-08",

                "travelers": 2,

                "budget_min": 3000,

                "budget_max": 5000,

                "purpose": "vacation",
            }
        )
        .execute()
    )

    if (
        not query_result.data
        or len(query_result.data) == 0
    ):

        logger.error(
            "Failed to create test travel query."
        )

        return

    query_id = (
        query_result.data[0]["id"]
    )

    logger.info(
        f"Created test travel query with ID: {query_id}"
    )

    # -------------------------------------------------------------
    # CREATE TEST USER PREFERENCES
    # -------------------------------------------------------------

    await (
        client.client
        .table("user_preferences")
        .insert(
            {
                "user_id": user_id,

                "travel_query_id": query_id,

                "travel_class": "economy",

                "direct_flights_only": True,

                "accommodation_types": [
                    "hotel",
                    "apartment",
                ],

                "hotel_rating": 4,

                "amenities": [
                    "wifi",
                    "pool",
                    "gym",
                ],

                "transportation_modes": [
                    "public_transit",
                    "taxi",
                ],

                "activity_types": [
                    "cultural",
                    "sightseeing",
                    "culinary",
                ],
            }
        )
        .execute()
    )

    logger.info(
        "Test data creation completed successfully."
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


async def main(
    args: argparse.Namespace,
) -> int:
    """
    Main function for the setup script.

    Args:
        args:
            Parsed command-line arguments.

    Returns:
        Exit code.
    """

    # -------------------------------------------------------------
    # LOGGING
    # -------------------------------------------------------------

    setup_logging(
        log_level=args.log_level
    )

    logger.info(
        "Starting database setup..."
    )

    # -------------------------------------------------------------
    # LOAD ENVIRONMENT
    # -------------------------------------------------------------

    if os.path.exists(".env"):

        load_dotenv()

        logger.debug(
            ".env file loaded."
        )

    # -------------------------------------------------------------
    # INITIALIZE CONFIGURATION
    # -------------------------------------------------------------

    config = initialize_config(
        custom_config_path=args.config
    )

    # -------------------------------------------------------------
    # VALIDATE SUPABASE API CONFIGURATION
    # -------------------------------------------------------------

    if not config.api.validate():

        logger.error(
            "Missing required API keys for Supabase."
        )

        logger.error(
            "Please ensure SUPABASE_URL and "
            "SUPABASE_KEY are set in your environment."
        )

        return 1

    # -------------------------------------------------------------
    # VALIDATE POSTGRESQL CONFIGURATION
    # -------------------------------------------------------------

    missing_variables = (
        validate_database_environment()
    )

    if missing_variables:

        logger.error(
            "Missing PostgreSQL configuration:"
        )

        for variable in missing_variables:

            logger.error(
                f"  - {variable}"
            )

        logger.error(
            "Please configure these variables "
            "in your .env file."
        )

        return 1

    # -------------------------------------------------------------
    # INITIALIZE DATABASE
    # -------------------------------------------------------------

    success = await initialize_database(
        config=config,
        reset=args.reset,
    )

    if not success:

        logger.error(
            "Database initialization failed."
        )

        return 1

    # -------------------------------------------------------------
    # CREATE TEST DATA
    # -------------------------------------------------------------

    if args.test_data:

        try:

            client = SupabaseClient(
                url=config.api.supabase_url,
                key=config.api.supabase_key,
            )

            await create_test_data(
                client
            )

        except Exception as e:

            logger.error(
                f"Error creating test data: {e}"
            )

            # Test data failure does not cause the entire
            # database setup to fail.

    # -------------------------------------------------------------
    # COMPLETE
    # -------------------------------------------------------------

    logger.info(
        "Database setup completed successfully."
    )

    return 0


# ---------------------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Set up the Supabase database "
            "for the travel planner"
        )
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Reset the database "
            "(drop and recreate tables)"
        ),
    )

    parser.add_argument(
        "--test-data",
        action="store_true",
        help=(
            "Create test data "
            "in the database"
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        help=(
            "Path to custom configuration file"
        ),
    )

    parser.add_argument(
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
        help=(
            "Set the logging level"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------------------


if __name__ == "__main__":

    args = parse_args()

    exit_code = asyncio.run(
        main(args)
    )

    sys.exit(exit_code)