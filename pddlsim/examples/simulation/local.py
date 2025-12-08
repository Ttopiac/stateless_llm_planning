import asyncio
import logging
import random
from collections.abc import Sequence

from pddlsim.ast import GroundedAction
from pddlsim.local import simulate_configuration
from pddlsim.remote.client import (
    GiveUpAction,
    SimulationAction,
    SimulationClient,
    with_no_initializer,
)
from pddlsim.remote.server import SimulatorConfiguration


def pick_grounded_action(
    actions: Sequence[GroundedAction],
) -> GroundedAction:
    return random.choice(actions)


async def get_next_action(simulation: SimulationClient) -> SimulationAction:
    options = await simulation.get_grounded_actions()

    match len(options):
        case 0:
            return GiveUpAction.from_dead_end()
        case 1:
            return options[0]
        case _:
            return pick_grounded_action(options)


async def main() -> None:
    summary = await simulate_configuration(
        SimulatorConfiguration.from_domain_and_problem_files(
            "assets/problems/gripper/domain.pddl",
            "assets/problems/gripper/instance.pddl",
        ),
        with_no_initializer(get_next_action),
    )
    print(f"Finished with: {summary}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    asyncio.run(main())
