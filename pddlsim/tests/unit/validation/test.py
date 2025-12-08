import importlib.resources
from dataclasses import dataclass
from importlib.abc import Traversable

import pytest

from pddlsim.parser import parse_domain, parse_problem
from tests import preprocess_traversables

_RESOURCES = importlib.resources.files(__name__)


@dataclass
class _InvalidPDDLCase:
    pddl_text: str
    expected_validation_error: str

    def assert_match(self, error: ValueError) -> None:
        assert self.expected_validation_error == str(error)


def _preprocess_invalid_domain(traversable: Traversable) -> _InvalidPDDLCase:
    domain_text = traversable.joinpath("domain.pddl").read_text()
    expected_output = traversable.joinpath("output.txt").read_text()

    return _InvalidPDDLCase(domain_text, expected_output)


_INVALID_DOMAINS = preprocess_traversables(
    _RESOURCES.joinpath("domains"), _preprocess_invalid_domain
)


def _preprocess_invalid_problem(traversable: Traversable) -> _InvalidPDDLCase:
    problem_text = traversable.joinpath("problem.pddl").read_text()
    expected_output = traversable.joinpath("output.txt").read_text()

    return _InvalidPDDLCase(problem_text, expected_output)


_INVALID_PROBLEMS = preprocess_traversables(
    _RESOURCES.joinpath("problems"), _preprocess_invalid_problem
)
_INVALID_PROBLEMS_DOMAIN = parse_domain(
    _RESOURCES.joinpath("domain.pddl").read_text()
)


@pytest.mark.parametrize(
    "case",
    _INVALID_DOMAINS.values(),
    ids=_INVALID_DOMAINS.keys(),
)
def test_domain_validation(case: _InvalidPDDLCase) -> None:
    with pytest.raises(ValueError) as exception_info:
        parse_domain(case.pddl_text)

    case.assert_match(exception_info.value)


@pytest.mark.parametrize(
    "case",
    _INVALID_PROBLEMS.values(),
    ids=_INVALID_PROBLEMS.keys(),
)
def test_problem_validation(case: _InvalidPDDLCase) -> None:
    with pytest.raises(ValueError) as exception_info:
        parse_problem(case.pddl_text, _INVALID_PROBLEMS_DOMAIN)

    case.assert_match(exception_info.value)
