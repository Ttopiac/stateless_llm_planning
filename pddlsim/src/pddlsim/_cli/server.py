"""Implementation of the `pddlsim server` command.

Run `pddlsim server --help` for more information.
"""

import asyncio
import os

import click

from pddlsim.remote.server import SimulationServer, SimulatorConfiguration


@click.command("server")
@click.argument("domain_path", type=click.Path(dir_okay=False, exists=True))
@click.argument("problem_path", type=click.Path(dir_okay=False, exists=True))
@click.option(
    "--show-revealables",
    "show_revealables",
    is_flag=True,
    help="Controls if agents should receive the full revealables section when requesting the problem definition.",  # noqa: E501
)
@click.option(
    "--show-fallibilities",
    "show_action_fallibilities",
    is_flag=True,
    help="Controls if agents should receive the full action fallibilities section when requesting the problem definition.",  # noqa: E501
)
@click.option(
    "--seed",
    "seed",
    type=int,
    help="An optional seed used to power the random aspects of the simulation.",
)
@click.option(
    "--host",
    "host",
    default="127.0.0.1",
    help="The host network interface to run the simulation on.",
)
@click.option(
    "--port",
    "port",
    type=int,
    help="The port on the network interface to run the simulation on.",
)
def server_command(
    domain_path: str | os.PathLike,
    problem_path: str | os.PathLike,
    show_revealables: bool,
    show_action_fallibilities: bool,
    seed: int | None,
    host: str,
    port: int | None,
) -> None:
    """Run a simulation server using the given domain and problem."""
    configuration = SimulatorConfiguration.from_domain_and_problem_files(
        domain_path,
        problem_path,
    )

    configuration.show_revealables = show_revealables
    configuration.show_action_fallibilities = show_action_fallibilities
    configuration.seed = seed

    async def run_server() -> None:
        server = await SimulationServer.from_host_and_port(
            configuration, host, port
        )

        await server.serve()

    asyncio.run(run_server())
