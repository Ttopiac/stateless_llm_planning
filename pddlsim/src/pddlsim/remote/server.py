"""Code for opening a simulation server accessible by the internet."""

import asyncio
import logging
import os
from dataclasses import dataclass

from pddlsim.ast import Domain, GroundedAction, Problem
from pddlsim.parser import parse_domain_problem_pair_from_files
from pddlsim.remote import (
    _RSP_VERSION,
    _RSPMessageBridge,
)
from pddlsim.remote._message import (
    Error,
    ErrorSource,
    GetGroundedActionsRequest,
    GetGroundedActionsResponse,
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
    SessionUnsupported,
    TerminationPayload,
)
from pddlsim.simulation import Seed, Simulation

_LOGGER = logging.getLogger(__name__)


@dataclass
class SimulatorConfiguration:
    """Configuration for simulation servers (local or otherwise).

    The individual fields of `ServerConfiguration` may be modified,
    as it simply stores data.
    """

    domain: Domain
    """The domain to simulate."""
    problem: Problem
    """The problem to simulate."""
    show_revealables: bool = False
    """Whether clients of the simulation should be able to access the revealables of the problem."""  # noqa: E501
    show_action_fallibilities: bool = False
    """Whether clients of the simulation should be able to access the action fallibilities of the problem."""  # noqa: E501
    seed: Seed | None = None
    """Random seed used to derive probabilistics aspects of simulation."""

    @classmethod
    def from_domain_and_problem_files(
        cls, domain_path: str | os.PathLike, problem_path: str | os.PathLike
    ) -> "SimulatorConfiguration":
        """Create a default `ServerConfiguration` from paths to a domain and problem."""  # noqa: E501
        domain, problem = parse_domain_problem_pair_from_files(
            domain_path, problem_path
        )

        return cls(domain, problem)

    async def __call__(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Operates a full simulation using the passed communication objects.

        > [!NOTE]
        > This is a very low-level API, and `start_simulation_server`/
        > `pddlsim.local` should be used instead, if possible.
        """
        peer_host, peer_port = writer.get_extra_info("peername")
        _LOGGER.info(
            f"attempting simulation negotiation with `{peer_host}:{peer_port}`"
        )

        server = await _SimulationServerInstance.start_session(
            _RSPMessageBridge(reader, writer), self
        )
        await server.operate_session()

        _LOGGER.info(
            f"finished simulation session with `{peer_host}:{peer_port}`"
        )


@dataclass(frozen=True)
class _SimulationServerInstance:
    _simulation: Simulation
    _bridge: _RSPMessageBridge
    _configuration: SimulatorConfiguration

    @classmethod
    async def start_session(
        cls,
        bridge: _RSPMessageBridge,
        configuration: SimulatorConfiguration,
    ) -> "_SimulationServerInstance":
        session_setup_request = await bridge.receive_payload(
            SessionSetupRequest
        )

        if session_setup_request.supported_rsp_version != _RSP_VERSION:
            session_unsupported = SessionUnsupported()

            await bridge.send_payload(session_unsupported)
            raise session_unsupported
        else:
            await bridge.send_payload(SessionSetupResponse())

        simulation = Simulation.from_domain_and_problem(
            configuration.domain, configuration.problem, seed=configuration.seed
        )

        return cls(simulation, bridge, configuration)

    async def _handle_problem_setup_request(self) -> None:
        await self._bridge.send_payload(
            ProblemSetupResponse(
                self._simulation.domain,
                self._simulation.problem,
                self._configuration.show_action_fallibilities,
                self._configuration.show_revealables,
            )
        )

    async def _handle_perception_request(self) -> None:
        await self._bridge.send_payload(
            PerceptionResponse(list(self._simulation.state))
        )

    async def _handle_goal_tracking_request(self) -> None:
        await self._bridge.send_payload(
            GoalTrackingResponse(
                self._simulation.reached_goal_indices,
                self._simulation.unreached_goal_indices,
            )
        )

    async def _handle_get_grounded_actions_request(self) -> None:
        await self._bridge.send_payload(
            GetGroundedActionsResponse(
                list(self._simulation.get_grounded_actions())
            )
        )

    async def _handle_perform_grounded_action_request(
        self, grounded_action: GroundedAction
    ) -> None:
        success = self._simulation.apply_grounded_action(grounded_action)

        if not self._simulation.is_solved():
            await self._bridge.send_payload(
                PerformGroundedActionResponse(success)
            )

    async def _handle_request(self) -> None:
        payload = await self._bridge.receive_any_payload()

        match payload:
            case ProblemSetupRequest():
                await self._handle_problem_setup_request()
            case PerceptionRequest():
                await self._handle_perception_request()
            case GoalTrackingRequest():
                await self._handle_goal_tracking_request()
            case GetGroundedActionsRequest():
                await self._handle_get_grounded_actions_request()
            case PerformGroundedActionRequest():
                await self._handle_perform_grounded_action_request(
                    payload.grounded_action
                )
            case _:
                error = Error(
                    ErrorSource.EXTERNAL,
                    f"expected request, found {payload.type()}",
                )

                await self._bridge.send_payload(error)
                raise error

    async def operate_session(self) -> None:
        try:
            while not self._simulation.is_solved():
                await self._handle_request()

            await self._bridge.send_payload(GoalsReached())
        except TerminationPayload:
            pass
        except Exception as exception:
            exception_reason = str(exception)

            await self._bridge.send_payload(
                Error(
                    ErrorSource.INTERNAL,
                    exception_reason if exception_reason else None,
                )
            )
            raise exception


@dataclass(frozen=True)
class SimulationServer:
    """A simulation server that agents can interact with.

    To construct, use `SimulationServer.from_host_and_port`. Alternatively, to
    simply run a simulation, you can use `start_simulation_server`.
    """

    _server: asyncio.Server

    @classmethod
    async def from_host_and_port(
        cls,
        configuration: SimulatorConfiguration,
        host: str,
        port: int | None = None,
    ) -> "SimulationServer":
        """Create a simulation server running on the specified host-port pair.

        The configuration for the actual simulation is handled by
        `SimulationConfiguration`.

        If the port passed is `None`, a random port is chosen by the OS. You
        can later use `SimulationServer.host` to see the chosen port.
        """
        server = await asyncio.start_server(configuration, host, port)
        result = SimulationServer(server)

        _LOGGER.info(
            f"created simulation server on `{result.host}:{result.port}`"
        )

        return result

    @property
    def host(self) -> str:
        """The host that the server is running on."""
        return self._server.sockets[0].getsockname()[0]

    @property
    def port(self) -> int:
        """The port that the server is running on."""
        return self._server.sockets[0].getsockname()[1]

    async def serve(self) -> None:
        """Start the simulation server, allowing agents to connect to it."""
        _LOGGER.info("starting simulation server")

        async with self._server:
            await self._server.serve_forever()


async def start_simulation_server(
    configuration: SimulatorConfiguration,
    host: str,
    port: int | None = None,
) -> None:
    """Open a simulation server for a given configuration.

    `host` and `port` form a pair, specifying the network interface
    to open the server on. If the port is unspecified, it is chosen by the OS.
    """
    server = await SimulationServer.from_host_and_port(
        configuration, host, port
    )

    await server.serve()
