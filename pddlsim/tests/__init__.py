from collections.abc import Callable, Mapping
from importlib.abc import Traversable


def preprocess_traversables[T](
    traversable: Traversable, preprocessor: Callable[[Traversable], T]
) -> Mapping[str, T]:
    return {
        traversable.name: preprocessor(traversable)
        for traversable in traversable.iterdir()
    }
