from abc import ABC, ABCMeta, abstractmethod
from enum import EnumType, StrEnum
from typing import Any, Self

from koda_validate import (
    Choices,
    Invalid,
    StringValidator,
    Valid,
    Validator,
)


class Serdeable[T](ABC):
    @abstractmethod
    def serialize(self) -> T:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _validator(cls) -> Validator[T]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _create(cls, value: T) -> Self:
        raise NotImplementedError

    @classmethod
    def deserialize(cls, data: Any) -> Self:
        match cls._validator()(data).map(cls._create):
            case Valid(deserialized_value):
                return deserialized_value
            case Invalid():
                raise ValueError(
                    f"could not deserialize into {cls.__name__} from {data}"
                )


class ABCEnum(ABCMeta, EnumType):
    pass


class SerdeableEnum(Serdeable[str], StrEnum, metaclass=ABCEnum):
    def serialize(self) -> str:
        return self.value

    @classmethod
    def _validator(cls) -> Validator[str]:
        return StringValidator(Choices({source.value for source in cls}))

    @classmethod
    def _create(cls, value: str) -> Self:
        return cls(value)
