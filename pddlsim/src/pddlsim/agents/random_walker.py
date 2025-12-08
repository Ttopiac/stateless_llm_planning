"""Code for a completely random agent, running until reaching a dead end.

For a random agent which has a maximum step limit, see
`pddlsim.agents.random_limited_walker`.
"""

# The code in this module is designed not to use `pddlsim.agent.Agent`, but
# rather do things in a functional style, with `pddlsim.agent.AgentInitializer`
# directly. While this style can be useful, for any non-trivial agent,
# subclassing from `Agent` is recommended.

import random
from collections.abc import Sequence

from pddlsim.ast import GroundedAction
from pddlsim.remote.client import (
    AgentInitializer,
    GiveUpAction,
    SimulationAction,
    SimulationClient,
    with_no_initializer,
)
from pddlsim.simulation import Seed


def configure(seed: Seed | None = None) -> AgentInitializer:
    """Configure an initializer for the random walker agent.

    The agent simply chooses a random action at each possible step.
    """
    random_ = random.Random(seed)

    def _pick_grounded_action(
        actions: Sequence[GroundedAction],
    ) -> GroundedAction:
        return random_.choice(actions)

    async def _get_next_action(
        simulation: SimulationClient,
    ) -> SimulationAction:
        options = await simulation.get_grounded_actions()

        match len(options):
            case 0:
                return GiveUpAction.from_dead_end()
            case 1:
                return options[0]
            case _:
                return _pick_grounded_action(options)

    return with_no_initializer(_get_next_action)
