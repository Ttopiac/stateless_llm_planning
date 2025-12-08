"""Code for an agent using a planner to solve problems with multiple goals.

This agent is essentially an extension of `pddlsim.agents.planner` to
support problems that may have multiple goals. It works by attempting to achieve
goals sequentially. It first picks an as-of-yet uncompleted goal, and then
constructs a new PDDL problem with that particular goal as its only goal, and
with the current perceived state as its state. After that, it instructs a
planner to solve that problem, and uses its plan to act in the simulation.

As this agent relies on a classical planner, the following PDDL features
are unsupported:

- `:probabilistic-effects`
- `:fallible-actions`
- `:revealables`
"""

from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar, override

from pddlsim.ast import (
    ActionFallibilitiesSection,
    Domain,
    GoalsSection,
    GroundedAction,
    Identifier,
    InitializationSection,
    Object,
    Problem,
    RawProblem,
    Requirement,
    RequirementsSection,
    RevealablesSection,
)
from pddlsim.remote.client import (
    Agent,
    SimulationAction,
    SimulationClient,
)


@dataclass
class MultipleGoalPlanner(Agent):
    """A planner-backed agent with support for multiple problem goals.

    See `pddlsim.agents.multiple_goal_planner` for more information.
    """

    _client: SimulationClient
    _domain: Domain
    _problem: Problem
    _plan_steps: deque[GroundedAction] = field(default_factory=deque)

    UNSUPPORTED_DOMAIN_REQUIREMENTS: ClassVar = {
        Requirement.PROBABILISTIC_EFFECTS
    }
    """Domain requirements unsupported by `MultipleGoalPlanner`."""
    UNSUPPORTED_PROBLEM_REQUIREMENTS: ClassVar = {
        Requirement.FALLIBLE_ACTIONS,
        Requirement.REVEALABLES,
    }
    """Problem requirements unsupported by `MultipleGoalPlanner`."""

    @override
    @classmethod
    async def _initialize(
        cls, client: SimulationClient, _configuration: None
    ) -> "MultipleGoalPlanner":
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

        return MultipleGoalPlanner(client, domain, problem)

    async def _set_plan_for_goal(self, goal_index: int) -> None:
        # Lazy import for performance
        import unified_planning.shortcuts as ups  # type: ignore
        from unified_planning.io import PDDLReader  # type: ignore

        domain = await self._client.get_domain()
        problem = await self._client.get_problem()

        current_state = await self._client.get_perceived_state()

        goal_condition = problem.goals_section[goal_index]

        new_problem = Problem(
            RawProblem(
                problem.name,
                problem.used_domain_name,
                RequirementsSection(
                    {
                        requirement
                        for requirement in problem.requirements_section
                        if requirement is not Requirement.MULTIPLE_GOALS
                    }
                ),
                problem.objects_section,
                ActionFallibilitiesSection(),
                RevealablesSection(),
                InitializationSection(set(current_state)),
                GoalsSection([goal_condition]),
            ),
            domain,
        )

        up_problem: ups.Problem = PDDLReader().parse_problem_string(
            repr(domain), repr(new_problem)
        )

        ups.get_environment().credits_stream = None  # Disable credits

        with ups.OneshotPlanner(problem_kind=up_problem.kind) as planner:
            self._plan_steps = deque(
                GroundedAction(
                    Identifier(action_instance.action.name),
                    tuple(
                        Object(parameter.object().name)
                        for parameter in action_instance.actual_parameters
                    ),
                )
                for action_instance in planner.solve(up_problem).plan.actions
            )

    @override
    async def _get_next_action(self) -> SimulationAction:
        if not self._plan_steps:
            uncompleted_goal_indices = (
                await self._client.get_unreached_goal_indices()
            )
            chosen_index = uncompleted_goal_indices[0]

            await self._set_plan_for_goal(chosen_index)

        return self._plan_steps.popleft()
