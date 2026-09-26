"""
Destination Research Agent for the travel planner system.

This module implements the specialized agent responsible for researching
destination information, analyzing travel advisories, providing weather
insights, and identifying points of interest for potential travel destinations.
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from travel_planner.agents.base import (
    AgentConfig,
    AgentContext,
    BaseAgent,
)
from travel_planner.utils import (
    AgentExecutionError,
    AgentLogger,
    handle_errors,
    with_retry,
)


@dataclass
class DestinationInfo:
    """Information about a travel destination."""

    name: str
    country: str
    description: str = ""
    weather: dict[str, Any] = field(default_factory=dict)
    best_times_to_visit: list[str] = field(default_factory=list)
    points_of_interest: list[dict[str, Any]] = field(
        default_factory=list
    )
    local_transportation: list[dict[str, Any]] = field(
        default_factory=list
    )
    travel_advisories: list[dict[str, Any]] = field(
        default_factory=list
    )
    visa_requirements: str = ""
    language: str = ""
    currency: str = ""
    timezone: str = ""
    cost_index: float = 0.0


class DestinationContext(AgentContext):
    """
    Context for the destination research agent.

    AgentContext inherits from Pydantic BaseModel, so Pydantic
    Field(default_factory=...) is used for mutable defaults.
    """

    query: str = ""

    destinations: list[DestinationInfo] = Field(
        default_factory=list
    )

    selected_destination: DestinationInfo | None = None

    travel_dates: dict[str, str] = Field(
        default_factory=dict
    )

    search_results: dict[str, Any] = Field(
        default_factory=dict
    )


class DestinationResearchAgent(
    BaseAgent[DestinationContext]
):
    """
    Specialized agent for researching travel destinations.

    This agent is responsible for:

    1. Analyzing user preferences to suggest appropriate destinations.
    2. Researching detailed information about destinations.
    3. Checking travel advisories and visa requirements.
    4. Providing weather and seasonal information.
    5. Identifying key points of interest and activities.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
    ):
        """
        Initialize the destination research agent.

        Args:
            config:
                Configuration for the agent.
        """

        default_config = AgentConfig(
            name="Destination Research",
            instructions=(
                "You are an AI destination research specialist "
                "for travel planning. "
                "Your expertise is in providing comprehensive, "
                "accurate information about travel destinations "
                "worldwide. Research and analyze destinations based "
                "on user preferences, provide detailed information "
                "about points of interest, local travel conditions, "
                "weather patterns, and travel advisories. "
                "Your goal is to help travelers make informed "
                "decisions about their destinations."
            ),
            tools=[
                # Future tools can be added here:
                # - Travel information search
                # - Weather API
                # - Travel advisories
                # - Points of interest
                # - Visa information
            ],
        )

        super().__init__(
            config or default_config,
            DestinationContext,
        )

        self.logger = AgentLogger(self.name)

    async def run(
        self,
        input_data: str | list[dict[str, Any]],
        context: DestinationContext | None = None,
    ) -> dict[str, Any]:
        """
        Run the destination research agent.

        Args:
            input_data:
                User input or conversation history.

            context:
                Optional destination research context.

        Returns:
            Updated context and research results.
        """

        self.logger.info(
            "Running destination research agent with input: "
            f"{input_data if isinstance(input_data, str) else '...'}"
        )

        # ---------------------------------------------------------
        # Initialize context
        # ---------------------------------------------------------

        if context is None:
            context = DestinationContext()

        # ---------------------------------------------------------
        # Extract query
        #
        # The orchestration layer may pass either:
        #
        #   1. A plain string
        #   2. A list of chat messages
        #
        # Previously only strings were handled, which resulted in:
        #
        #   Processing destination research for query:
        #
        # with an empty query.
        # ---------------------------------------------------------

        if isinstance(input_data, str):

            context.query = input_data.strip()

        elif isinstance(input_data, list):

            user_messages: list[str] = []

            for message in input_data:

                if not isinstance(message, dict):
                    continue

                role = message.get("role")
                content = message.get("content")

                if role != "user":
                    continue

                if content is None:
                    continue

                if isinstance(content, str):

                    content_text = content.strip()

                    if content_text:
                        user_messages.append(
                            content_text
                        )

                else:

                    # Handle non-string content safely.
                    content_text = str(content).strip()

                    if content_text:
                        user_messages.append(
                            content_text
                        )

            if user_messages:

                # The latest user message normally contains the
                # actual travel request.
                context.query = user_messages[-1]

        # ---------------------------------------------------------
        # Safety fallback
        # ---------------------------------------------------------

        if not context.query:

            context.query = (
                "Plan a travel destination based on the "
                "available user travel preferences."
            )

        self.logger.info(
            f"Destination research query resolved to: "
            f"{context.query}"
        )

        try:

            result = await self.process(
                input_data,
                context,
            )

            return {
                "context": context,
                "result": result,
            }

        except Exception as e:

            error_msg = (
                "Error in destination research agent: "
                f"{e!s}"
            )

            self.logger.error(error_msg)

            raise AgentExecutionError(
                error_msg,
                self.name,
                original_error=e,
            ) from e

    @handle_errors(error_cls=AgentExecutionError)
    async def process(
        self,
        input_data: str | list[dict[str, Any]],
        context: DestinationContext,
    ) -> dict[str, Any]:
        """
        Process the destination research request.

        Args:
            input_data:
                User input or conversation history.

            context:
                Destination research context.

        Returns:
            Research results.
        """

        self.logger.info(
            "Processing destination research for query: "
            f"{context.query}"
        )

        # Prepare messages for the base agent.
        self._prepare_messages(input_data)

        # ---------------------------------------------------------
        # If a destination is already selected, research it.
        #
        # Otherwise ask Gemini to suggest destinations.
        # ---------------------------------------------------------

        if not context.selected_destination:

            result = await self._suggest_destinations(
                context
            )

        else:

            result = await self._research_destination(
                context.selected_destination.name,
                context,
            )

        return result

    async def _suggest_destinations(
        self,
        context: DestinationContext,
    ) -> dict[str, Any]:
        """
        Suggest destinations based on user preferences.

        Args:
            context:
                Destination research context.

        Returns:
            Dictionary containing destination suggestions.
        """

        self.logger.info(
            "Suggesting destinations for query: "
            f"{context.query}"
        )

        suggestion_prompt = (
            "Based on the user's travel request, suggest "
            "3-5 suitable travel destinations. "
            "For each destination provide: "
            "destination name, country, brief description, "
            "why it matches the request, best time to visit, "
            "and notable attractions. "
            "Return the answer as valid JSON."
        )

        messages = [
            {
                "role": "system",
                "content": self.instructions,
            },
            {
                "role": "user",
                "content": context.query,
            },
            {
                "role": "system",
                "content": suggestion_prompt,
            },
        ]

        response = await self._call_model(
            messages
        )

        return {
            "suggestions": response.get(
                "content",
                "",
            )
        }

    async def _research_destination(
        self,
        destination: str,
        context: DestinationContext,
    ) -> dict[str, Any]:
        """
        Research detailed information about a destination.

        Args:
            destination:
                Name of the destination.

            context:
                Destination research context.

        Returns:
            Dictionary containing detailed research.
        """

        self.logger.info(
            f"Researching destination: {destination}"
        )

        research_prompt = (
            f"Provide comprehensive information about "
            f"{destination} as a travel destination. "
            "Include details about the location, weather, "
            "best times to visit, main attractions, "
            "local transportation options, visa requirements, "
            "local currency, language, approximate travel costs, "
            "and relevant travel advisories. "
            "Return the answer as valid JSON."
        )

        messages = [
            {
                "role": "system",
                "content": self.instructions,
            },
            {
                "role": "user",
                "content": (
                    f"Research {destination} "
                    "as a travel destination."
                ),
            },
            {
                "role": "system",
                "content": research_prompt,
            },
        ]

        response = await self._call_model(
            messages
        )

        return {
            "research": response.get(
                "content",
                "",
            )
        }

    @with_retry(max_attempts=3)
    async def _call_model(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Call the configured Gemini model through the
        Gemini compatibility layer.

        IMPORTANT:
        The old OpenAI-specific rate limiter has deliberately
        been removed.

        The previous implementation contained:

            @rate_limited("openai")

        That limiter expected OpenAI rate-limit configuration
        and was causing:

            Amount must be a number between 0 and the
            maximum capacity

        Gemini retry/rate handling is now performed by the
        Gemini compatibility layer.
        """

        self.logger.info(
            f"Calling model with {len(messages)} messages"
        )

        self.logger.log_llm_input(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
        )

        try:

            response = (
                await self.client
                .chat
                .completions
                .create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
            )

            self.logger.log_llm_output(
                model=self.config.model,
                response=response,
            )

            if (
                response.choices
                and len(response.choices) > 0
            ):

                content = (
                    response.choices[0]
                    .message
                    .content
                )

                return {
                    "content": content
                }

            return {
                "content": "No response generated."
            }

        except Exception as e:

            self.logger.error(
                f"Error calling model: {e!s}"
            )

            raise

    # -------------------------------------------------------------
    # Future specialized methods
    # -------------------------------------------------------------
    #
    # These can later be implemented with external APIs/tools:
    #
    # - Checking weather forecasts
    # - Retrieving travel advisories
    # - Searching for points of interest
    # - Analyzing visa requirements
    # - Checking flight information
    # - Checking hotel information
    #