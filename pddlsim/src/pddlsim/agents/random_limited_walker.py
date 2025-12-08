"""Code for a completely random agent, giving up on reaching a step maximum.

For a random agent with no maximum step limit, see
`pddlsim.agents.random_walker`.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NewType, override

from pddlsim.ast import GroundedAction
from pddlsim.remote.client import (
    ConfigurableAgent,
    GiveUpAction,
    SimulationAction,
    SimulationClient,
)
from pddlsim.simulation import Seed

MaxSteps = NewType("MaxSteps", int)


@dataclass(frozen=True)
class RandomLimitedWalkerConfiguration:
    """Configuration for the `RandomnLimitedWalker` agent."""

    max_steps: int
    """Maximum number of steps the agent will take before giving up."""
    seed: Seed | None = None
    """Optional seed to power the agent's random actions."""


@dataclass
class RandomLimitedWalker(ConfigurableAgent[RandomLimitedWalkerConfiguration]):
    """An agent performing actions at random, with a maximum number of steps.

    Once the maximum number of simulation steps is reached, the agent gives up.
    """

    _client: SimulationClient
    _max_steps: int
    _random: random.Random
    _current_steps: int

    @override
    @classmethod
    async def _initialize(
        cls,
        client: SimulationClient,
        configuration: RandomLimitedWalkerConfiguration,
    ) -> "RandomLimitedWalker":
        return cls(
            client,
            configuration.max_steps,
            random.Random(configuration.seed),
            0,
        )

    def _pick_grounded_action(
        self, actions: Sequence[GroundedAction]
    ) -> GroundedAction:
        return self._random.choice(actions)

    @override
    async def _get_next_action(self) -> SimulationAction:
        if self._current_steps >= self._max_steps:
            return GiveUpAction(f"maximum steps reached: {self._max_steps}")

        self._current_steps += 1

        options = await self._client.get_grounded_actions()

        match len(options):
            case 0:
                return GiveUpAction.from_dead_end()
            case 1:
                return options[0]
            case _:
                return self._pick_grounded_action(options)
