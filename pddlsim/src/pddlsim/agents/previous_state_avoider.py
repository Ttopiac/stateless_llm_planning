"""Code for an agent that attempts to avoid retracing its steps.

This means that in the current state of the environment, the agent will
remember the environment's previous state, and if possible, perform an action
that is believed to not cause the agent to repeat that state. If such an action
is not believed to be possible, the agent will perform a random action.
"""

import random
from dataclasses import dataclass
from typing import override

from pddlsim.ast import Domain, GroundedAction, Problem
from pddlsim.remote.client import (
    ConfigurableAgent,
    SimulationAction,
    SimulationClient,
)
from pddlsim.simulation import Seed, Simulation
from pddlsim.state import SimulationState


@dataclass
class PreviousStateAvoider(ConfigurableAgent[Seed]):
    """An agent that attempts to avoid the previous state if possible.

    See `pddlsim.agents.previous_state_avoider` for more information.
    """

    _client: SimulationClient
    _domain: Domain
    _problem: Problem
    _random: random.Random
    _previous_state: SimulationState | None = None

    @override
    @classmethod
    async def _initialize(
        cls,
        client: SimulationClient,
        configuration: Seed,
    ) -> "PreviousStateAvoider":
        domain = await client.get_domain()
        problem = await client.get_problem()

        return PreviousStateAvoider(
            client, domain, problem, random.Random(configuration)
        )

    async def _is_action_backtracking(
        self, grounded_action: GroundedAction
    ) -> bool:
        simulation = Simulation.from_domain_and_problem(
            self._domain,
            self._problem,
            await self._client.get_perceived_state(),
        )
        simulation.apply_grounded_action(grounded_action)

        return simulation.state == self._previous_state

    @override
    async def _get_next_action(self) -> SimulationAction:
        """Perform a single simulation step."""
        grounded_actions = await self._client.get_grounded_actions()
        non_backtracking_actions = [
            grounded_action
            for grounded_action in grounded_actions
            if not await self._is_action_backtracking(grounded_action)
        ]

        possibilities = (
            non_backtracking_actions
            if non_backtracking_actions
            else grounded_actions
        )

        picked_action = self._random.choice(possibilities)

        self._previous_state = await self._client.get_perceived_state()

        return picked_action
