import inspect
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self, TypedDict, override

from koda_validate import (
    AlwaysValid,
    BoolValidator,
    IntValidator,
    ListValidator,
    Min,
    NoneValidator,
    OptionalValidator,
    StringValidator,
    TypedDictValidator,
    Validator,
)

from pddlsim._serde import (
    Serdeable,
    SerdeableEnum,
)
from pddlsim.ast import Domain, Object, Predicate, Problem
from pddlsim.parser import parse_domain_problem_pair
from pddlsim.simulation import GroundedAction


class Payload[T](Serdeable[T]):
    payloads: ClassVar[dict[str, type["Payload"]]] = {}

    def __init_subclass__(cls) -> None:
        if not inspect.isabstract(cls):
            cls.payloads[cls.type()] = cls

        super().__init_subclass__()

    @classmethod
    @abstractmethod
    def type(cls) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class SessionSetupRequest(Payload[int]):
    supported_rsp_version: int

    @override
    def serialize(self) -> int:
        return self.supported_rsp_version

    @override
    @classmethod
    def _validator(cls) -> Validator[int]:
        return IntValidator(Min(1))

    @override
    @classmethod
    def _create(cls, value: int) -> "SessionSetupRequest":
        return SessionSetupRequest(value)

    @override
    @classmethod
    def type(cls) -> str:
        return "session-setup-request"


@dataclass(frozen=True)
class EmptyPayload(Payload[None]):
    @override
    def serialize(self) -> None:
        return None

    @override
    @classmethod
    def _validator(cls) -> Validator[None]:
        return NoneValidator()

    @override
    @classmethod
    def _create(cls, _value: None) -> Self:
        return cls()


class SessionSetupResponse(EmptyPayload):
    @override
    @classmethod
    def type(cls) -> str:
        return "session-setup-response"


class ProblemSetupRequest(EmptyPayload):
    @override
    @classmethod
    def type(cls) -> str:
        return "problem-setup-request"


class SerializedProblemSetupResponse(TypedDict):
    domain: str
    problem: str


@dataclass(frozen=True)
class ProblemSetupResponse(Payload[SerializedProblemSetupResponse]):
    domain: Domain
    problem: Problem
    _show_action_fallibilities: bool = field(default=True, compare=False)
    _show_revealables: bool = field(default=True, compare=False)

    @override
    def serialize(self) -> SerializedProblemSetupResponse:
        return SerializedProblemSetupResponse(
            domain=self.domain.as_pddl(),
            problem=self.problem.as_pddl(
                self._show_action_fallibilities, self._show_revealables
            ),
        )

    @override
    @classmethod
    def _validator(cls) -> Validator[SerializedProblemSetupResponse]:
        return TypedDictValidator(SerializedProblemSetupResponse)

    @override
    @classmethod
    def _create(
        cls, value: SerializedProblemSetupResponse
    ) -> "ProblemSetupResponse":
        domain, problem = parse_domain_problem_pair(
            value["domain"], value["problem"]
        )

        return ProblemSetupResponse(domain, problem)

    @override
    @classmethod
    def type(cls) -> str:
        return "problem-setup-response"


class PerceptionRequest(EmptyPayload):
    @override
    @classmethod
    def type(cls) -> str:
        return "perception-request"


@dataclass(frozen=True)
class PerceptionResponse(Payload[list[Any]]):
    true_predicates: list[Predicate[Object]]

    @override
    def serialize(self) -> list[Any]:
        return [predicate.serialize() for predicate in self.true_predicates]

    @override
    @classmethod
    def _validator(cls) -> Validator[list[Any]]:
        return ListValidator(AlwaysValid())

    @override
    @classmethod
    def _create(cls, value: list[Any]) -> "PerceptionResponse":
        return PerceptionResponse(
            [Predicate[Object].deserialize(item) for item in value]
        )

    @override
    @classmethod
    def type(cls) -> str:
        return "perception-response"


class GoalTrackingRequest(EmptyPayload):
    @override
    @classmethod
    def type(cls) -> str:
        return "goal-tracking-request"


class SerializedGoalTrackingResponse(TypedDict):
    reached: list[int]
    unreached: list[int]


@dataclass(frozen=True)
class GoalTrackingResponse(Payload[SerializedGoalTrackingResponse]):
    reached_goal_indices: list[int]
    unreached_goal_indices: list[int]

    @override
    def serialize(self) -> SerializedGoalTrackingResponse:
        return SerializedGoalTrackingResponse(
            reached=self.reached_goal_indices,
            unreached=self.unreached_goal_indices,
        )

    @override
    @classmethod
    def _validator(cls) -> Validator[SerializedGoalTrackingResponse]:
        return TypedDictValidator(SerializedGoalTrackingResponse)

    @override
    @classmethod
    def _create(
        cls, value: SerializedGoalTrackingResponse
    ) -> "GoalTrackingResponse":
        return GoalTrackingResponse(value["reached"], value["unreached"])

    @override
    @classmethod
    def type(cls) -> str:
        return "goal-tracking-response"


class GetGroundedActionsRequest(EmptyPayload):
    @override
    @classmethod
    def type(cls) -> str:
        return "get-grounded-actions-request"


@dataclass(frozen=True)
class GetGroundedActionsResponse(Payload[list[Any]]):
    grounded_actions: list[GroundedAction]

    @override
    def serialize(self) -> list[Any]:
        return [
            grounded_action.serialize()
            for grounded_action in self.grounded_actions
        ]

    @override
    @classmethod
    def _validator(cls) -> Validator[list[Any]]:
        return ListValidator(AlwaysValid())

    @override
    @classmethod
    def _create(cls, value: list[Any]) -> "GetGroundedActionsResponse":
        return GetGroundedActionsResponse(
            [GroundedAction.deserialize(item) for item in value]
        )

    @override
    @classmethod
    def type(cls) -> str:
        return "get-grounded-actions-response"


@dataclass(frozen=True)
class PerformGroundedActionRequest(Payload[Any]):
    grounded_action: GroundedAction

    @override
    def serialize(self) -> Any:
        return self.grounded_action.serialize()

    @override
    @classmethod
    def _validator(cls) -> Validator[Any]:
        return AlwaysValid()

    @override
    @classmethod
    def _create(cls, value: Any) -> "PerformGroundedActionRequest":
        return PerformGroundedActionRequest(GroundedAction.deserialize(value))

    @override
    @classmethod
    def type(cls) -> str:
        return "perform-grounded-action-request"


@dataclass(frozen=True)
class PerformGroundedActionResponse(Payload[bool]):
    success: bool

    @override
    def serialize(self) -> bool:
        return self.success

    @override
    @classmethod
    def _validator(cls) -> BoolValidator:
        return BoolValidator()

    @override
    @classmethod
    def _create(cls, value: bool) -> "PerformGroundedActionResponse":
        return PerformGroundedActionResponse(value)

    @override
    @classmethod
    def type(cls) -> str:
        return "perform-grounded-action-response"


class TerminationPayload[T](Payload[T], Exception):  # noqa: N818
    @abstractmethod
    def description(self) -> str:
        raise NotImplementedError


class GoalsReached(EmptyPayload, TerminationPayload):
    @override
    @classmethod
    def type(cls) -> str:
        return "goals-reached"

    @override
    def description(self) -> str:
        return "goals reached"


class ErrorSource(SerdeableEnum):
    INTERNAL = "internal"
    EXTERNAL = "external"


class ErrorReason(TypedDict):
    source: Any
    reason: str | None


@dataclass(frozen=True)
class Error(TerminationPayload[ErrorReason]):
    source: ErrorSource
    reason: str | None

    @classmethod
    def from_type_mismatch(
        cls,
        expected_payload_type: type[Payload],
        received_payload_type: type[Payload],
    ) -> "Error":
        return Error(
            ErrorSource.EXTERNAL,
            f"expected {expected_payload_type.type()}, got {received_payload_type.type()}",  # noqa: E501
        )

    @classmethod
    def from_communication_channel_closed(cls) -> "Error":
        return Error(ErrorSource.EXTERNAL, "communication channel closed")

    @override
    def serialize(self) -> ErrorReason:
        return ErrorReason(source=self.source.serialize(), reason=self.reason)

    @override
    @classmethod
    def _validator(cls) -> Validator[ErrorReason]:
        return TypedDictValidator(ErrorReason)

    @override
    @classmethod
    def _create(cls, value: ErrorReason) -> "Error":
        return Error(ErrorSource.deserialize(value["source"]), value["reason"])

    @override
    @classmethod
    def type(cls) -> str:
        return "error"

    @override
    def description(self) -> str:
        return (
            f"{self.reason} ({self.source} error)"
            if self.reason
            else f"{self.source} error"
        )


class SessionUnsupported(EmptyPayload, TerminationPayload):
    @override
    @classmethod
    def type(cls) -> str:
        return "session-unsupported"

    @override
    def description(self) -> str:
        return "session unsupported"


class Timeout(EmptyPayload, TerminationPayload):
    @override
    @classmethod
    def type(cls) -> str:
        return "timeout"

    @override
    def description(self) -> str:
        return "timeout"


@dataclass(frozen=True)
class GiveUp(TerminationPayload[str | None]):
    reason: str | None

    @override
    def serialize(self) -> str | None:
        return self.reason

    @override
    @classmethod
    def _validator(cls) -> Validator[str | None]:
        return OptionalValidator(StringValidator())

    @override
    @classmethod
    def _create(cls, value: str | None) -> "GiveUp":
        return GiveUp(value)

    @override
    @classmethod
    def type(cls) -> str:
        return "give-up"

    @override
    def description(self) -> str:
        return (
            f"session given up ({self.reason})"
            if self.reason
            else "session given up"
        )


@dataclass(frozen=True)
class Custom(TerminationPayload[str | None]):
    reason: str | None

    @override
    def serialize(self) -> str | None:
        return self.reason

    @override
    @classmethod
    def _validator(cls) -> Validator[str | None]:
        return OptionalValidator(StringValidator())

    @override
    @classmethod
    def _create(cls, value: str | None) -> "Custom":
        return Custom(value)

    @override
    @classmethod
    def type(cls) -> str:
        return "custom"

    @override
    def description(self) -> str:
        return self.reason if self.reason else "unknown"


class SerializedMessage(TypedDict):
    type: str
    payload: Any


@dataclass(frozen=True)
class Message(Serdeable[SerializedMessage]):
    payload: Payload

    @override
    def serialize(self) -> SerializedMessage:
        return SerializedMessage(
            type=self.payload.type(), payload=self.payload.serialize()
        )

    @override
    @classmethod
    def _validator(cls) -> Validator[SerializedMessage]:
        return TypedDictValidator(SerializedMessage)

    @override
    @classmethod
    def _create(cls, value: SerializedMessage) -> "Message":
        return Message(
            Payload.payloads[value["type"]].deserialize(value["payload"])
        )
