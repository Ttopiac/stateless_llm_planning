import asyncio
import logging
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
    act_in_simulation,
)

MaxSteps = NewType("MaxSteps", int)


@dataclass
class RandomLimitedWalker(ConfigurableAgent[MaxSteps]):
    """An agent performing actions at random, with a maximum number of steps.

    Once the maximum number of simulation steps is reached, the agent gives up.
    """

    _client: SimulationClient
    _max_steps: int
    _current_steps: int

    @override
    @classmethod
    async def _initialize(
        cls, client: SimulationClient, max_steps: MaxSteps
    ) -> "RandomLimitedWalker":
        return cls(client, max_steps, 0)

    def _pick_grounded_action(
        self, actions: Sequence[GroundedAction]
    ) -> GroundedAction:
        return random.choice(actions)

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


async def main() -> None:
    port = int(input("What port to connect to? (0-65535): "))
    summary = await act_in_simulation(
        "127.0.0.1", port, RandomLimitedWalker.configure(MaxSteps(300))
    )

    print(f"Finished with: {summary}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    asyncio.run(main())
