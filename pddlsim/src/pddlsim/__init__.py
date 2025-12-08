"""PDDLSIM is an execution simulator for PDDL domain-problem pairs.

Users can create simulators accessible over the internet, or locally, as well
as agents, interfacing with said simulators. A usage tutorial for PDDLSIM
is available [here](https://github.com/galk-research/pddlsim/wiki/Tutorial).

PDDLSIM is composed of several main submodules:

- `pddlsim.remote.server` and `pddlsim.remote.client` allow the creation
interfaceable simulators and agents
- `pddlsim.local` provides shortcuts for creating local simulations
and interfacing with them
- `pddlsim.ast` provides utilities for representing PDDL domains and problems
- `pddlsim.parser` contains code to `parse` PDDL domains and problems into AST
- `pddlsim.simulation` contains a low-level interface for simulations
- `pddlsim.state` contains a data structure for the representation
of the state of a simulation
- `pddlsim.agents` (available with the `agents` extra) contains several
built-in agents that can be used when interacting with simulations.
"""

import importlib.resources

_RESOURCES = importlib.resources.files(__name__)
