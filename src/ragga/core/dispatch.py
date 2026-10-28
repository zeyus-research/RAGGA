"""Signal dispatching and handling, make classes subscriptable."""

from collections.abc import Callable


class PropertyWrapper[T]:
    def __init__(self, initial_value: T | None = None):
        self._subscribers: list[Callable[[T], None]] = []
        self._value = initial_value

    def subscribe(self, subscriber: Callable[[T], None]):
        """Subscribe to updates of the associated property."""
        self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber):
        """Unsubscribe from updates of the associated property."""
        self._subscribers.remove(subscriber)

    def dispatch(self, new_value):
        """Dispatch an event to subscribers of the associated property."""
        for subscriber in self._subscribers:
            subscriber(new_value)

    def __get__(self, _instance, _owner) -> T | None:
        return self._value

    def __set__(self, _instance, value: T | None):
        self._value = value
        self.dispatch(new_value=value)
