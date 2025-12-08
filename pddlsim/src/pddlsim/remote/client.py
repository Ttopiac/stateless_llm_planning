"""Interface for interacting with simulations, and code to connect to them."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import NoReturn, Self, override

from pddlsim.ast import Domain, GroundedAction, Problem
from pddlsim.remote import (
    _RSP_VERSION,
    _RSPMessageBridge,
)
from pddlsim.remote._message import (
    Error,
    ErrorSource,
    GetGroundedActionsRequest,
    GetGroundedActionsResponse,
    GiveUp,
    GoalsReached,
    GoalTrackingRequest,
    GoalTrackingResponse,
    PerceptionRequest,
    PerceptionResponse,
    PerformGroundedActionRequest,
    PerformGroundedActionResponse,
    ProblemSetupRequest,
    ProblemSetupResponse,
    SessionSetupRequest,
    SessionSetupResponse,
    TerminationPayload,
)
from pddlsim.simulation import SimulationState

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ErrorResult:
    """Represents a session prematurely terminated due to an error."""

    reason: str | None
    """The reason, if any, for the error."""

    @override
    def __str__(self) -> str:
        return f"error ({self.reason})" if self.reason else "error"


@dataclass(frozen=True)
class FailureResult:
    """Represents a session where not all problem goals were reached."""

    reason: str | None
    """The reason, if any, for the failure."""

    @override
    def __str__(self) -> str:
        return f"failure ({self.reason})" if self.reason else "failure"


@dataclass(frozen=True)
class SuccessResult:
    """Represents a session ended by achieving all problem goals."""

    @override
    def __str__(self) -> str:
        return "success"


type SessionResult = SuccessResult | FailureResult | ErrorResult
"""Represents the end results of a session, i.e., what caused it to stop."""


@dataclass
class SessionStatistics:
    """Statistics on the behavior of the agent during the session."""

    actions_attempted: int = 0
    """The number of actions the agent attempted (including failures)."""
    failed_actions: int = 0
    """The number of actions failed due to action fallibilities."""
    perception_requests: int = 0
    """The number of unique perception requests by the agent."""
    goal_tracking_requests: int = 0
    """The number of unique goal tracking requests by the agent.
    
    When calling either `SimulationClient.get_reached_goal_indices` or
    `SimulationClient.get_unreached_goal_indices`, the same underlying
    query is sent, meaning calling both only increases the count by one.
    Repeated calls, without performing actions in between, are likewise
    not counted."""
    get_grounded_actions_requests: int = 0
    """The number of unique "get grounded actions" requests by the agent.
    
    When repeatedly making such requests, without performing actions in between,
    the count only increases by one.
    """


@dataclass(frozen=True)
class SessionSummary:
    """A summary of the behavior of the agent in a simulation session."""

    result: SessionResult
    """The result of the session: success, failure, or error."""
    statistics: SessionStatistics
    """Statistics on the behavior of the agent during the session."""
    seconds_elapsed: float
    """The total time, in seconds, of the session."""

    def is_success(self) -> bool:
        """Check if the result of the session is a success."""
        return isinstance(self.result, SuccessResult)

    @override
    def __str__(self) -> str:
        return str(self.result)


@dataclass
class SimulationClient:
    """Interface with a remote simulation."""

    _bridge: _RSPMessageBridge

    _state: SimulationState | None = None
    _domain_problem_pair: tuple[Domain, Problem] | None = None
    _grounded_actions: list[GroundedAction] | None = None
    _reached_and_unreached_goal_indices: tuple[list[int], list[int]] | None = (
        None
    )

    _statistics: SessionStatistics = field(default_factory=SessionStatistics)

    @classmethod
    def _from_reader_and_writer(
        cls, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> "SimulationClient":
        return SimulationClient(
            _RSPMessageBridge(
                reader,
                writer,
            )
        )

    async def _start_session(self) -> None:
        await self._bridge.send_payload(SessionSetupRequest(_RSP_VERSION))

        _payload = await self._bridge.receive_payload(SessionSetupResponse)
        _LOGGER.info("started simulation session")

    async def _get_problem_setup(self) -> tuple[Domain, Problem]:
        if not self._domain_problem_pair:
            _LOGGER.info("getting domain and problem used in simulation")

            await self._bridge.send_payload(ProblemSetupRequest())

            payload = await self._bridge.receive_payload(ProblemSetupResponse)

            self._domain_problem_pair = (
                payload.domain,
                payload.problem,
            )

        return self._domain_problem_pair

    async def get_domain(self) -> Domain:
        """Get the domain used in the simulation."""
        # This is not wasteful thanks to caching
        return (await self._get_problem_setup())[0]

    async def get_problem(self) -> Problem:
        """Get the problem used in the simulation.

        "Hidden information", in particular revealables
        and action fallibilities, are redacted.
        """
        # This is not wasteful thanks to caching
        return (await self._get_problem_setup())[1]

    async def _get_reached_and_unreached_goals(
        self,
    ) -> tuple[list[int], list[int]]:
        if not self._reached_and_unreached_goal_indices:
            _LOGGER.info("getting reached and unreached goal indices")

            self._statistics.goal_tracking_requests += 1
            await self._bridge.send_payload(GoalTrackingRequest())

            payload = await self._bridge.receive_payload(GoalTrackingResponse)

            self._reached_and_unreached_goal_indices = (
                payload.reached_goal_indices,
                payload.unreached_goal_indices,
            )

        return self._reached_and_unreached_goal_indices

    async def get_reached_goal_indices(self) -> Sequence[int]:
        """Get the indices of the problem goals that have been reached.

        By using `pddlsim.ast.GoalList.get_goal` on the problem's
        `pddlsim.ast.Problem.goals` one can see the condition corresponding to
        the index.
        """
        return (await self._get_reached_and_unreached_goals())[0]

    async def get_unreached_goal_indices(self) -> Sequence[int]:
        """Get the indices of the problem goals that have yet to be reached.

        By using `pddlsim.ast.GoalList.get_goal` on the problem's
        `pddlsim.ast.Problem.goals` one can see the condition corresponding to
        the index.
        """
        return (await self._get_reached_and_unreached_goals())[1]

    async def get_perceived_state(self) -> SimulationState:
        """Get the current state, as perceived by the agent.

        > [!NOTE]
        > This state can differ than the one obtained by using
        > `SimulationClient.get_domain` and `SimulationClient.get_problem`
        > and simulating changes to the state manually, as the problem
        > has its action fallibilities and revealables removed, as these
        > are considered hidden information.
        """
        if not self._state:
            _LOGGER.info("getting perceived state")

            self._statistics.perception_requests += 1
            await self._bridge.send_payload(PerceptionRequest())

            payload = await self._bridge.receive_payload(PerceptionResponse)

            self._state = SimulationState(set(payload.true_predicates))

        return self._state

    async def get_grounded_actions(self) -> Sequence[GroundedAction]:
        """Get all grounded actions  for the agent in the current state.

        > [!NOTE]
        > These actions can differ from those obtained by using
        > `SimulationClient.get_domain` and `SimulationClient.get_problem`
        > and simulating the problem manually, as the problem has its action
        > fallibilities and revealables removed, as these are considered hidden
        > information.
        """
        if not self._grounded_actions:
            _LOGGER.info("getting possible grounded actions for current state")

            self._statistics.get_grounded_actions_requests += 1
            await self._bridge.send_payload(GetGroundedActionsRequest())

            payload = await self._bridge.receive_payload(
                GetGroundedActionsResponse
            )

            self._grounded_actions = payload.grounded_actions

        return self._grounded_actions

    async def _perform_grounded_action(
        self, grounded_action: GroundedAction
    ) -> None:
        self._state = None
        self._grounded_actions = None
        self._reached_and_unreached_goal_indices = None

        self._statistics.actions_attempted += 1
        await self._bridge.send_payload(
            PerformGroundedActionRequest(grounded_action)
        )
        response = await self._bridge.receive_payload(
            PerformGroundedActionResponse
        )
        self._statistics.failed_actions += not response.success

    async def _give_up(self, reason: str | None) -> NoReturn:
        give_up = GiveUp(reason)

        await self._bridge.send_payload(give_up)
        raise give_up


@dataclass(frozen=True)
class GiveUpAction:
    """`SimulationAction` representing that the agent has given up.

    After making this action. the simulation will terminate.
    """

    reason: str | None = None

    @classmethod
    def from_dead_end(cls) -> "GiveUpAction":
        """Construct a `GiveUpAction` which is due to a dead end."""
        return GiveUpAction("dead end")


type SimulationAction = GiveUpAction | GroundedAction
"""An interaction of the agent with a simulation.

This can be a `pddlsim.simulation.GroundedAction`, which will affect the
state of the simulation, an indication by the agent that it is giving up
on the simulation, etc.
"""


type NextActionGetter = Callable[[], Awaitable[SimulationAction]]
"""A simple model of an agent, sequentially returning `SimulationAction`s.

The callable is async as the agent may need to use the `SimulationClient`,
which may involve communication with the simulation server.
"""
type AgentInitializer = Callable[
    [SimulationClient], Awaitable[NextActionGetter]
]
"""An agent initializer, allowing the agent to setup, and then returning it.

This acts as an "agent constructor", in such a way that the agent is expected
to store the `SimulationClient` handle it receives. Finally, assuming
initialization involves using the `SimulationClient`, the callable is
async.
"""


def with_no_initializer(
    get_next_action: Callable[[SimulationClient], Awaitable[SimulationAction]],
) -> AgentInitializer:
    """Wrap stateless agents into initializers.

    Most of PDDLSIM's API expects `AgentInitializer`s, so this function is
    useful for quickly making "dummy initializers" for stateless agents,
    while avoiding boilerplate.
    """

    async def no_op_initializer(client: SimulationClient) -> NextActionGetter:
        return lambda: get_next_action(client)

    return no_op_initializer


class ConfigurableAgent[C](ABC):
    """An agent which sequentially performs actions in a simulation.

    To create an `AgentInitializer`, `Agent.configure` is used. Thus, it is
    the agent's main entry point.
    """

    @classmethod
    def configure(cls, configuration: C) -> AgentInitializer:
        """Pre-configure an agent, but don't initialize it yet.

        Returns an `AgentInitializer ` which can be used throughout PDDLSIM's
        different APIs, such as `pddlsim.local.simulate_configuration`, or
        `pddlsim.remote.client.act_in_simulation`.
        """

        async def initializer(client: SimulationClient) -> NextActionGetter:
            agent = await cls._initialize(client, configuration)

            return agent._get_next_action

        return initializer

    @classmethod
    @abstractmethod
    async def _initialize(
        cls, client: SimulationClient, configuration: C
    ) -> Self:
        raise NotImplementedError

    @abstractmethod
    async def _get_next_action(self) -> SimulationAction:
        raise NotImplementedError


class Agent(ConfigurableAgent[None]):
    """An agent which sequentially performs actions in a simulation.

    To create an `AgentInitializer`, `Agent.configure` is used. Thus, it is
    the agent's main entry point.
    """

    @classmethod
    def configure(cls, _configuration: None = None) -> AgentInitializer:
        """Pre-configure an agent, but don't initialize it yet.

        Returns an `AgentInitializer ` which can be used throughout PDDLSIM's
        different APIs, such as `pddlsim.local.simulate_configuration`, or
        `pddlsim.remote.client.act_in_simulation`.
        """

        async def initializer(client: SimulationClient) -> NextActionGetter:
            agent = await cls._initialize(client, _configuration)

            return agent._get_next_action

        return initializer

    @classmethod
    @abstractmethod
    async def _initialize(
        cls, client: SimulationClient, configuration: None
    ) -> Self:
        raise NotImplementedError

    @abstractmethod
    async def _get_next_action(self) -> SimulationAction:
        raise NotImplementedError


async def act_in_simulation(
    host: str,
    port: int,
    initializer: AgentInitializer,
) -> SessionSummary:
    """Connect to the remote simulation and run the agent on it.

    The remote simulation to connect to is specified as a `host` and
    `port` pair, where `host` is generally an IP address.

    The returned object (`SessionTermination`) represents how the simulation
    session ended.
    """
    reader, writer = await asyncio.open_connection(host, port)

    client = SimulationClient._from_reader_and_writer(reader, writer)

    start = time.monotonic()

    try:
        await client._start_session()

        get_next_action = await initializer(client)
        _LOGGER.info("initialized agent")

        while True:
            action = await get_next_action()

            _LOGGER.info(f"performing action `{action}`")

            match action:
                case GiveUpAction(reason):
                    await client._give_up(reason)
                case GroundedAction():
                    await client._perform_grounded_action(action)
    except TerminationPayload as payload:
        match payload:
            case GoalsReached():
                result: SessionResult = SuccessResult()
            case Error(reason):
                result = ErrorResult(reason)
            case other_payload:
                result = FailureResult(other_payload.description())

        _LOGGER.info(f"finished simulation with {result}")

        return SessionSummary(
            result, client._statistics, time.monotonic() - start
        )
    except Exception as exception:
        exception_reason = str(exception)

        await client._bridge.send_payload(
            Error(
                ErrorSource.INTERNAL,
                exception_reason if exception_reason else None,
            )
        )

        raise exception
