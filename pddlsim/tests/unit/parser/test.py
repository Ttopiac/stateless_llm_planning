import importlib.resources
from dataclasses import dataclass
from importlib.abc import Traversable

import pytest

from pddlsim.ast import Domain, Problem
from pddlsim.parser import (
    parse_domain_problem_pair,
)
from tests import preprocess_traversables

_RESOURCES = importlib.resources.files(__name__)


@dataclass
class _ParserCase:
    domain: Domain
    problem: Problem


def _preprocess_parser_case(traversable: Traversable) -> _ParserCase:
    domain_text = traversable.joinpath("domain.pddl").read_text()
    problem_text = traversable.joinpath("problem.pddl").read_text()

    domain, problem = parse_domain_problem_pair(domain_text, problem_text)

    return _ParserCase(domain, problem)


_CASES = preprocess_traversables(
    _RESOURCES.joinpath("cases"), _preprocess_parser_case
)


@pytest.mark.parametrize(
    "case",
    _CASES.values(),
    ids=_CASES.keys(),
)
def test_parser_roundtrip(case: _ParserCase) -> None:
    domain, problem = parse_domain_problem_pair(
        case.domain.as_pddl(), case.problem.as_pddl()
    )

    assert domain == case.domain
    assert problem == case.problem
