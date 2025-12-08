"""Utilities for local simulation, with a similar API to remote simulation."""

from dataclasses import dataclass
from typing import ClassVar

from pddlsim.remote import client
from pddlsim.remote.server import (
    SimulationServer,
    SimulatorConfiguration,
)


@dataclass(frozen=True)
class LocalSimulator:
    """Local simulator for running multiple local agent simulation sessions.

    The main constructor is `LocalSimulator.from_domain_problem_pair`.
    """

    _server: SimulationServer

    _HOST: ClassVar = "127.0.0.1"

    @classmethod
    async def from_configuration(
        cls, configuration: SimulatorConfiguration
    ) -> "LocalSimulator":
        """Create a `LocalSimulator` from a `pddlsim.ast.Domain` and a `pddlsim.ast.Problem`."""  # noqa: E501
        return LocalSimulator(
            await SimulationServer.from_host_and_port(
                configuration, LocalSimulator._HOST
            )
        )

    async def simulate(
        self, initializer: client.AgentInitializer
    ) -> client.SessionSummary:
        """Run the provided agent on the simulation until termination.

        The returned value is an object representing how the simulation ended.
        """
        return await client.act_in_simulation(
            self._server.host, self._server.port, initializer
        )


async def simulate_configuration(
    configuration: SimulatorConfiguration,
    initializer: client.AgentInitializer,
) -> client.SessionSummary:
    """Simulate the provided agent on a simulation-session configuration.

    The returned value is an object representing how the simulation ended.
    """
    simulator = await LocalSimulator.from_configuration(configuration)

    return await simulator.simulate(initializer)
