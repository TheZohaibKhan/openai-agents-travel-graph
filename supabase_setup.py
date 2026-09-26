#!/usr/bin/env python3
"""
Supabase Setup Tool

A command-line tool for initializing and managing the Supabase database
for the Travel Planner application.
"""

import argparse
import asyncio
import sys

from psycopg import connect

from travel_planner.config import initialize_config
from travel_planner.data.setup import (
    create_test_data,
    initialize_database,
)
from travel_planner.data.supabase import SupabaseClient
from travel_planner.utils.logging import get_logger, setup_logging

# Initialize logger
logger = get_logger(__name__)


def get_postgres_connection():
    """
    Create a direct PostgreSQL connection using environment variables.
    """
    import os

    return connect(
        host=os.getenv("SUPABASE_DB_HOST"),
        port=int(os.getenv("SUPABASE_DB_PORT", "5432")),
        dbname=os.getenv("SUPABASE_DB_NAME", "postgres"),
        user=os.getenv("SUPABASE_DB_USER", "postgres"),
        password=os.getenv("SUPABASE_DB_PASSWORD"),
    )


async def handle_init_command(args: argparse.Namespace) -> int:
    """
    Handle the 'init' command to initialize the Supabase database.
    """
    logger.info("Initializing Supabase database...")

    config = initialize_config(
        custom_config_path=args.config,
        raise_on_error=False,
    )

    if not config.api.supabase_url or not config.api.supabase_key:
        logger.error("Supabase URL and API key must be configured")
        logger.error(
            "Please set SUPABASE_URL and SUPABASE_KEY environment variables"
        )
        return 1

    success = await initialize_database(config, reset=args.reset)

    if not success:
        logger.error("Failed to initialize Supabase database")
        return 1

    if args.test_data:
        try:
            client = SupabaseClient(
                url=config.api.supabase_url,
                key=config.api.supabase_key,
            )
            await create_test_data(client)
            logger.info("Test data created successfully")
        except Exception as e:
            logger.error(f"Error creating test data: {e}")

    logger.info("Supabase database initialized successfully")
    return 0


async def handle_status_command(args: argparse.Namespace) -> int:
    """
    Handle the 'status' command to check database status.

    Uses a direct PostgreSQL connection instead of the old
    execute_sql Supabase RPC function.
    """
    logger.info("Checking Supabase database status...")

    config = initialize_config(
        custom_config_path=args.config,
        raise_on_error=False,
    )

    if not config.api.supabase_url or not config.api.supabase_key:
        logger.error("Supabase URL and API key must be configured")
        return 1

    connection = None

    try:
        connection = get_postgres_connection()

        with connection.cursor() as cursor:
            # Check database connection
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]

            # Get public tables
            cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            )

            tables = cursor.fetchall()

            if not tables:
                logger.warning("No tables found in the database")
                return 1

            print("Supabase connection successful")
            print(f"PostgreSQL: {version.split(',')[0]}")
            print("Tables found in database:")

            for table in tables:
                print(f"  - {table[0]}")

            # Count users
            cursor.execute("SELECT COUNT(*) FROM public.users")
            users_count = cursor.fetchone()[0]

            # Count travel plans
            cursor.execute("SELECT COUNT(*) FROM public.travel_plans")
            plans_count = cursor.fetchone()[0]

            print(f"Users in database: {users_count}")
            print(f"Travel plans in database: {plans_count}")

        return 0

    except Exception as e:
        logger.error(f"Error connecting to PostgreSQL/Supabase: {e}")
        return 1

    finally:
        if connection:
            connection.close()


async def handle_reset_command(args: argparse.Namespace) -> int:
    """
    Handle the 'reset' command to reset the database.
    """
    logger.warning("Resetting Supabase database - ALL DATA WILL BE LOST!")

    if not args.force:
        prompt = "Are you sure you want to reset the database? "
        prompt += "This will delete ALL data. Type 'yes' to confirm: "
        confirmation = input(prompt)

        if confirmation.lower() != "yes":
            logger.info("Database reset cancelled")
            return 0

    config = initialize_config(
        custom_config_path=args.config,
        raise_on_error=False,
    )

    success = await initialize_database(config, reset=True)

    if not success:
        logger.error("Failed to reset Supabase database")
        return 1

    if args.test_data:
        try:
            client = SupabaseClient(
                url=config.api.supabase_url,
                key=config.api.supabase_key,
            )
            await create_test_data(client)
            logger.info("Test data created successfully")
        except Exception as e:
            logger.error(f"Error creating test data: {e}")

    logger.info("Supabase database reset successfully")
    return 0


async def main(args: argparse.Namespace) -> int:
    """
    Main entry point for the Supabase setup tool.
    """
    setup_logging(log_level=args.log_level)

    command_handlers = {
        "init": handle_init_command,
        "status": handle_status_command,
        "reset": handle_reset_command,
    }

    handler = command_handlers.get(args.command)

    if handler:
        return await handler(args)

    return 0


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Supabase database setup tool for Travel Planner"
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
        help="Set the logging level",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Command to execute",
    )

    # Initialize command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize the database",
    )

    init_parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the database (drop and recreate tables)",
    )

    init_parser.add_argument(
        "--test-data",
        action="store_true",
        help="Create test data in the database",
    )

    # Status command
    subparsers.add_parser(
        "status",
        help="Check database status",
    )

    # Reset command
    reset_parser = subparsers.add_parser(
        "reset",
        help="Reset the database (drop and recreate tables)",
    )

    reset_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reset without confirmation (dangerous!)",
    )

    reset_parser.add_argument(
        "--test-data",
        action="store_true",
        help="Create test data after reset",
    )

    parser.set_defaults(command="init")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(main(args))
    sys.exit(exit_code)