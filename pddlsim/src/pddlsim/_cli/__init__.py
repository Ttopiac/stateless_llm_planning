"""Implementation of PDDLSIM's CLI command for running simulation servers.

> [!NOTE]
> To use PDDLSIM's CLI, the `cli` extra **must** be enabled.

For more information. run `pddlsim --help`.
"""

try:
    import click
except ImportError as error:
    raise ValueError(
        "cannot use the PDDLSIM CLI without the `cli` extra enabled"
    ) from error

import logging

from pddlsim._cli.client import client_command
from pddlsim._cli.server import server_command


@click.group("pddlsim")
def pddlsim_command() -> None:
    """Create PDDLSIM simulations and interact with them using agents."""
    logging.basicConfig(level=logging.INFO)


pddlsim_command.add_command(server_command)
pddlsim_command.add_command(client_command)
