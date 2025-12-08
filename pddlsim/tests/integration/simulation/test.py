import importlib.resources
from dataclasses import dataclass
from importlib.abc import Traversable

import pytest

from pddlsim.agents.previous_state_avoider import PreviousStateAvoider
from pddlsim.ast import Domain, Problem
from pddlsim.local import simulate_configuration
from pddlsim.parser import (
    parse_domain_problem_pair,
)
from pddlsim.remote.server import SimulatorConfiguration
from tests import preprocess_traversables

_RESOURCES = importlib.resources.files(__name__)


@dataclass
class _LocalSimulationCase:
    domain: Domain
    problem: Problem


def _preprocess_local_simulation_case(
    traversable: Traversable,
) -> _LocalSimulationCase:
    domain_text = traversable.joinpath("domain.pddl").read_text()
    problem_text = traversable.joinpath("problem.pddl").read_text()

    domain, problem = parse_domain_problem_pair(domain_text, problem_text)

    return _LocalSimulationCase(domain, problem)


_CASES = preprocess_traversables(
    _RESOURCES.joinpath("cases"), _preprocess_local_simulation_case
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    _CASES.values(),
    ids=_CASES.keys(),
)
async def test_local_simulation(case: _LocalSimulationCase) -> None:
    await simulate_configuration(
        SimulatorConfiguration(case.domain, case.problem, seed=42),
        PreviousStateAvoider.configure(42),
    )
