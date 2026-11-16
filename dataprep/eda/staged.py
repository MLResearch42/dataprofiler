"""Decorator that splits a computation into a "gather" and a "finish" stage.

A staged function is written as a generator: it yields the raw results it
gathered and receives them back to build the final ``Intermediate``. Callers
that want to run the two stages separately (``create_report`` does) pass
``_staged=True`` and get ``(raw results, completion function)``.
"""

from typing import Any, Callable, Generator, Tuple, Union, cast

from .intermediate import Intermediate

Decoratee = Callable[..., Generator[Any, Any, Intermediate]]

Completion = Callable[[Any], Intermediate]


def staged(
    func: Decoratee,
) -> Callable[..., Union[Tuple[Any, Completion], Intermediate]]:
    """Transform a two stage computation into a result and a completion function."""

    def staged_imp(
        *args: Any, _staged: bool = False, **kwargs: Any
    ) -> Union[Tuple[Any, Completion], Intermediate]:
        gen = func(*args, **kwargs)

        def completion(computed: Any) -> Intermediate:
            try:
                gen.send(computed)
                raise RuntimeError("Computation didn't stop.")
            except StopIteration as stop:
                return cast(Intermediate, stop.value)

        gathered = next(gen)
        if _staged:
            return gathered, completion
        return completion(gathered)

    staged_imp.__doc__ = func.__doc__
    staged_imp.__name__ = getattr(func, "__name__", "staged")
    return staged_imp
