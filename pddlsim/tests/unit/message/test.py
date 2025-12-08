import importlib
from dataclasses import dataclass
from typing import Any

import pytest

from pddlsim.ast import GroundedAction, Identifier, Object, Predicate
from pddlsim.parser import parse_domain_problem_pair
from pddlsim.remote._message import (
    Custom,
    Error,
    ErrorSource,
    GetGroundedActionsRequest,
    GetGroundedActionsResponse,
    GiveUp,
    GoalsReached,
    GoalTrackingRequest,
    GoalTrackingResponse,
    Message,
    Payload,
    PerceptionRequest,
    PerceptionResponse,
    PerformGroundedActionRequest,
    PerformGroundedActionResponse,
    ProblemSetupRequest,
    ProblemSetupResponse,
    SessionSetupRequest,
    SessionSetupResponse,
    SessionUnsupported,
    Timeout,
)

_RESOURCES = importlib.resources.files(__name__)


@dataclass
class MessageCase:
    payload: Payload
    expected_serialization: Any | None = None


_SAMPLE_DOMAIN, _SAMPLE_PROBLEM = parse_domain_problem_pair(
    _RESOURCES.joinpath("domain.pddl").read_text(),
    _RESOURCES.joinpath("problem.pddl").read_text(),
)


_CASES = [
    MessageCase(
        SessionSetupRequest(3),
        {"type": "session-setup-request", "payload": 3},
    ),
    MessageCase(
        SessionSetupResponse(),
        {"type": "session-setup-response", "payload": None},
    ),
    MessageCase(
        ProblemSetupRequest(),
        {"type": "problem-setup-request", "payload": None},
    ),
    MessageCase(ProblemSetupResponse(_SAMPLE_DOMAIN, _SAMPLE_PROBLEM)),
    MessageCase(
        PerceptionRequest(),
        {"type": "perception-request", "payload": None},
    ),
    MessageCase(
        PerceptionResponse(
            [Predicate(Identifier("at"), (Object("robot"), Object("house")))]
        ),
        {
            "type": "perception-response",
            "payload": [{"name": "at", "assignment": ["robot", "house"]}],
        },
    ),
    MessageCase(
        GoalTrackingRequest(),
        {"type": "goal-tracking-request", "payload": None},
    ),
    MessageCase(
        GoalTrackingResponse([0, 1], [2, 3]),
        {
            "type": "goal-tracking-response",
            "payload": {"reached": [0, 1], "unreached": [2, 3]},
        },
    ),
    MessageCase(
        GoalTrackingResponse([0, 1], [2, 3]),
        {
            "type": "goal-tracking-response",
            "payload": {"reached": [0, 1], "unreached": [2, 3]},
        },
    ),
    MessageCase(
        GetGroundedActionsRequest(),
        {"type": "get-grounded-actions-request", "payload": None},
    ),
    MessageCase(
        GetGroundedActionsResponse(
            [
                GroundedAction(
                    Identifier("move"), (Object("robot"), Object("house"))
                )
            ]
        ),
        {
            "type": "get-grounded-actions-response",
            "payload": [{"name": "move", "grounding": ["robot", "house"]}],
        },
    ),
    MessageCase(
        PerformGroundedActionRequest(
            GroundedAction(
                Identifier("move"), (Object("robot"), Object("house"))
            )
        ),
        {
            "type": "perform-grounded-action-request",
            "payload": {"name": "move", "grounding": ["robot", "house"]},
        },
    ),
    MessageCase(
        PerformGroundedActionResponse(True),
        {"type": "perform-grounded-action-response", "payload": True},
    ),
    MessageCase(
        GoalsReached(),
        {"type": "goals-reached", "payload": None},
    ),
    MessageCase(
        Error(ErrorSource.INTERNAL, "oh no!"),
        {
            "type": "error",
            "payload": {"source": "internal", "reason": "oh no!"},
        },
    ),
    MessageCase(
        Timeout(),
        {
            "type": "timeout",
            "payload": None,
        },
    ),
    MessageCase(
        GiveUp("too hard"),
        {
            "type": "give-up",
            "payload": "too hard",
        },
    ),
    MessageCase(
        SessionUnsupported(),
        {
            "type": "session-unsupported",
            "payload": None,
        },
    ),
    MessageCase(
        Custom("because i can"),
        {
            "type": "custom",
            "payload": "because i can",
        },
    ),
]


@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=lambda case: case.payload.type(),
)
def test_message_roundtrip(case: MessageCase) -> None:
    message = Message(case.payload)

    actual_serialization = message.serialize()

    if case.expected_serialization:
        assert actual_serialization == case.expected_serialization

    assert Message.deserialize(actual_serialization) == message
