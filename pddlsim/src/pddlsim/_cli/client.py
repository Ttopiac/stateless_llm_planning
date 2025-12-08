"""Implementation of the `pddlsim client` subcommand.

> [!NOTE]
> To use this command, beyond the `cli` extra, the `agents` extra must be
> enabled.

Run `pddlsim client --help` for more information.
"""

import asyncio

import click

from pddlsim.agents import random_walker
from pddlsim.agents.multiple_goal_planner import MultipleGoalPlanner
from pddlsim.agents.planner import Planner
from pddlsim.agents.previous_state_avoider import PreviousStateAvoider
from pddlsim.agents.random_limited_walker import (
    RandomLimitedWalker,
    RandomLimitedWalkerConfiguration,
)
from pddlsim.remote.client import AgentInitializer, act_in_simulation


@click.group("client")
@click.option(
    "--host",
    "host",
    default="127.0.0.1",
    help="The address of the network interface to connect to.",
)
@click.option(
    "--port",
    "port",
    type=int,
    required=True,
    help="The port of the network interface to connect to.",
)
def client_command(
    host: str,
    port: int,
) -> None:
    """Run an agent on the PDDLSIM simulation in the given host-port pair."""


@client_command.command("random-walker")
@click.option(
    "--seed",
    "seed",
    type=int,
    help="A seed for powering random functionality in the agent.",
)
def _random_walker(seed: int | None) -> AgentInitializer:
    """Configure an agent performing actions at random."""
    return random_walker.configure(seed)


@client_command.command("random-limited-walker")
@click.option(
    "--max-steps",
    "max_steps",
    type=int,
    required=True,
    help="The maximum amount of steps the agent may take. Once it reaches this maximum, it gives up.",  # noqa: E501
)
@click.option(
    "--seed",
    "seed",
    type=int,
    help="A seed for powering random functionality in the agent.",
)
def _random_limited_walker(
    max_steps: int, seed: int | None
) -> AgentInitializer:
    """Configure an agent performing actions at random.

    This agent has a configurable step-maximum. Once the simulation
    reaches that maximum, it gives up.
    """
    return RandomLimitedWalker.configure(
        RandomLimitedWalkerConfiguration(max_steps, seed)
    )


@client_command.command("previous-state-avoider")
@click.option(
    "--seed",
    "seed",
    type=int,
    help="A seed for powering random functionality in the agent.",
)
def _previous_state_avoider(seed: int | None) -> AgentInitializer:
    """Configure an agent acting randomly while avoiding the previous state.

    If the previous state was `X`, and the current state is `Y`, the agent
    will attempt to avoid state `X`, as it would otherwise render its previous
    action useless.
    """
    return PreviousStateAvoider.configure(seed)


@client_command.command("planner")
def _planner() -> AgentInitializer:
    """Configure an agent acting in a simulation using a planner.

    As this agent is planner-based, some PDDLSIM-specific PDDL extensions
    are not supported by it. Below is a non-exhaustive list:

    - `:probabilistic-effects`
    - `:revealables`
    - `:fallible-actions`
    - `:fallible-actions`
    - `:multiple-goals`
    """
    return Planner.configure()


@client_command.command("multiple-goal-planner")
def _multiple_goal_planner() -> AgentInitializer:
    """Configure an agent acting in a simulation using a planner.

    Unlike the `planner` agent, this agent has support for acting in problems
    with multiple goals. Like `planner` however, as this agent is planner-based,
    some PDDLSIM-specific PDDL extensions are not supported by it.
    Below is a non-exhaustive list:

    - `:probabilistic-effects`
    - `:revealables`
    - `:fallible-actions`
    - `:fallible-actions`
    """
    return MultipleGoalPlanner.configure()


@client_command.result_callback()
def _run_agent(initializer: AgentInitializer, host: str, port: int) -> None:
    asyncio.run(act_in_simulation(host, port, initializer))
