"""Items for creating simulators and interacting with them over the internet.

This module contains two main submodules:

- `pddlsim.remote.client` contains items related to interfacing with simulations
and creating agents
- `pddlsim.remote.server` contains items related to create a simulator server
"""

import asyncio
import logging
from dataclasses import dataclass

import cbor2

from pddlsim.remote._message import (
    Error,
    Message,
    Payload,
    TerminationPayload,
)

_RSP_VERSION = 1

_FRAME_LENGTH_BYTES = 4


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RSPMessageBridge:
    _reader: asyncio.StreamReader
    _writer: asyncio.StreamWriter

    async def send_payload(self, payload: Payload) -> None:
        serialized_message = Message(payload).serialize()
        _LOGGER.debug(f"sending: {serialized_message}")

        data = cbor2.dumps(serialized_message)

        try:
            # If the amount of bytes doesn't fit in the 32-bit unsigned integer,
            # an overflow error is raised, so an invalid message is never sent
            self._writer.write(len(data).to_bytes(_FRAME_LENGTH_BYTES))
            self._writer.write(data)
            await self._writer.drain()
        except ConnectionResetError as exception:
            raise Error.from_communication_channel_closed() from exception

    async def receive_any_payload(self) -> Payload:
        try:
            byte_size = int.from_bytes(
                await self._reader.readexactly(_FRAME_LENGTH_BYTES)
            )
            value_bytes: bytes = await self._reader.readexactly(byte_size)
        except (asyncio.IncompleteReadError, ConnectionResetError) as exception:
            raise Error.from_communication_channel_closed() from exception

        serialized_message = cbor2.loads(value_bytes)
        _LOGGER.debug(f"receiving: {serialized_message}")

        payload = Message.deserialize(serialized_message).payload

        if isinstance(payload, TerminationPayload):
            raise payload

        return payload

    async def receive_payload[P: Payload](
        self, expected_payload_type: type[P]
    ) -> P:
        payload = await self.receive_any_payload()

        if not isinstance(payload, expected_payload_type):
            error = Error.from_type_mismatch(
                expected_payload_type, type(payload)
            )

            await self.send_payload(error)
            raise error

        return payload
