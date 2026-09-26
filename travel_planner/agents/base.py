"""
Base agent class for the travel planner system.

This module implements the foundational Agent class that all specialized
agents in the travel planner system will inherit from. It provides common
functionality and standardized interfaces for all agents.

The model provider is Gemini through the local Gemini compatibility layer.
"""

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from travel_planner.agents.gemini_compat import GeminiCompatClient


# Type variable for context
T = TypeVar("T")


class AgentContext(BaseModel):
    """Base class for agent context that can be passed between agents."""

    pass


class TravelPlannerAgentError(Exception):
    """Base exception for all agent-related errors."""

    pass


class InvalidConfigurationError(TravelPlannerAgentError):
    """Exception raised when agent configuration is invalid."""

    pass


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str
    instructions: str

    # Kept for compatibility with the existing project.
    # Actual model selection is handled by GEMINI_MODEL.
    model: str = "gemini-3.8-flash"

    temperature: float = 0.7
    max_tokens: int | None = None
    tools: list[Any] = field(default_factory=list)


class BaseAgent(Generic[T]):
    """
    Base class for all travel planner agents.

    This class provides the foundation for specialized agents that handle
    different aspects of travel planning, such as destination research,
    flight search, accommodation booking, etc.

    The BaseAgent uses Gemini through GeminiCompatClient while preserving
    the existing agent interface used throughout the application.
    """

    def __init__(
        self,
        config: AgentConfig,
        context_type: type[T] | None = None,
    ):
        """
        Initialize a base agent.

        Args:
            config: Configuration for the agent.
            context_type: Type of context this agent handles.
        """
        self.config = config

        # Gemini client replacing the previous OpenAI client.
        self.client = GeminiCompatClient()

        self.context_type = context_type or AgentContext

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self.config.name

    @property
    def instructions(self) -> str:
        """Get the instructions for the agent."""
        return self.config.instructions

    async def run(
        self,
        input_data: str | list[dict[str, Any]],
        context: T | None = None,
    ) -> Any:
        """
        Run the agent with the provided input and context.

        Args:
            input_data: User input or conversation history.
            context: Optional context for the agent.

        Returns:
            Agent response or result.
        """
        raise NotImplementedError(
            "Subclasses must implement run method"
        )

    async def process(self, *args, **kwargs) -> Any:
        """
        Process the input according to the agent's specialized function.

        Specialized agents implement this method to perform their
        specific tasks.
        """
        raise NotImplementedError(
            "Subclasses must implement process method"
        )

    def _validate_config(self) -> bool:
        """Validate the agent configuration."""

        if not self.config.name:
            raise InvalidConfigurationError(
                "Agent name cannot be empty"
            )

        if not self.config.instructions:
            raise InvalidConfigurationError(
                "Agent instructions cannot be empty"
            )

        return True

    def _prepare_messages(
        self,
        input_data: str | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Prepare messages for the Gemini API compatibility layer.

        Args:
            input_data:
                User input or conversation history.

        Returns:
            List of messages formatted for the model.
        """

        if isinstance(input_data, str):

            messages = [
                {
                    "role": "system",
                    "content": self.instructions,
                },
                {
                    "role": "user",
                    "content": input_data,
                },
            ]

        elif input_data and input_data[0].get("role") != "system":

            messages = [
                {
                    "role": "system",
                    "content": self.instructions,
                },
                *input_data,
            ]

        else:
            messages = input_data

        return messages