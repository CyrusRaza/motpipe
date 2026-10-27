"""Name -> strategy lookup used to select detectors and trackers at runtime."""

from __future__ import annotations

import importlib
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps a string name to a strategy class.

    Targets may be given as a class or as a lazy ``"package.module:ClassName"``
    path. Lazy paths keep heavy optional dependencies (torch, ultralytics, ...)
    from being imported until the strategy is actually requested.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._targets: dict[str, type[T] | str] = {}

    def register(self, name: str, target: type[T] | str | None = None):
        """Register ``target`` under ``name``; usable directly or as a decorator."""
        key = name.lower()

        def _add(obj: type[T] | str):
            if key in self._targets:
                raise ValueError(f"{self.kind} '{key}' is already registered")
            self._targets[key] = obj
            return obj

        return _add if target is None else _add(target)

    def names(self) -> list[str]:
        return sorted(self._targets)

    def get(self, name: str) -> type[T]:
        key = name.lower()
        try:
            target = self._targets[key]
        except KeyError:
            raise ValueError(
                f"Unknown {self.kind} '{name}'. Available: {', '.join(self.names())}"
            ) from None
        if isinstance(target, str):
            module_name, _, attr = target.partition(":")
            target = getattr(importlib.import_module(module_name), attr)
            self._targets[key] = target
        return target

    def create(self, name: str, **kwargs: Any) -> T:
        return self.get(name)(**kwargs)
