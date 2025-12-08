"""Code for a simple, planner-based agent.

As this agent relies on a classical planner, the following PDDL features
are unsupported:

- `:probabilistic-effects`
- `:fallible-actions`
- `:revealables`
- `:multiple-goals`
"""

from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar, override

from pddlsim.ast import GroundedAction, Identifier, Object, Requirement
from pddlsim.remote.client import (
    Agent,
    GiveUpAction,
    SimulationAction,
    SimulationClient,
)


@dataclass
class Planner(Agent):
    """A planner-based agent, without support for any custom PDDL features.

    See also `pddlsim.agents.multiple_goal_planner` for a planner-based
    agent with support for some of PDDLSIM's custom PDDL features.
    """

    _client: SimulationClient
    _plan_steps: deque[GroundedAction] = field(default_factory=deque)

    UNSUPPORTED_DOMAIN_REQUIREMENTS: ClassVar = {
        Requirement.PROBABILISTIC_EFFECTS
    }
    """Domain requirements unsupported by `MultipleGoalPlanner`."""
    UNSUPPORTED_PROBLEM_REQUIREMENTS: ClassVar = {
        Requirement.FALLIBLE_ACTIONS,
        Requirement.MULTIPLE_GOALS,
        Requirement.REVEALABLES,
    }
    """Problem requirements unsupported by `MultipleGoalPlanner`."""

    @override
    @classmethod
    async def _initialize(
        cls, client: SimulationClient, _configuration: None
    ) -> "Planner":
        # Lazy import for performance
        import unified_planning.shortcuts as ups  # type: ignore
        from unified_planning.io import PDDLReader  # type: ignore

        domain = await client.get_domain()
        problem = await client.get_problem()

        for requirement in cls.UNSUPPORTED_DOMAIN_REQUIREMENTS:
            if requirement in domain.requirements_section:
                raise ValueError(
                    f"`{requirement}` requirement is not supported"
                )

        for requirement in cls.UNSUPPORTED_PROBLEM_REQUIREMENTS:
            if requirement in problem.requirements_section:
                raise ValueError(
                    f"`{requirement}` requirement are not supported"
                )

        up_problem: ups.Problem = PDDLReader().parse_problem_string(
            repr(domain), repr(problem)
        )

        ups.get_environment().credits_stream = None  # Disable credits

        with ups.OneshotPlanner(problem_kind=up_problem.kind) as planner:
            plan_steps = deque(
                GroundedAction(
                    Identifier(action_instance.action.name),
                    tuple(
                        Object(parameter.object().name)
                        for parameter in action_instance.actual_parameters
                    ),
                )
                for action_instance in planner.solve(up_problem).plan.actions
            )

        return Planner(client, plan_steps)

    @override
    async def _get_next_action(self) -> SimulationAction:
        if not self._plan_steps:
            return GiveUpAction("plan has ended, but problem unsolved")

        return self._plan_steps.popleft()
