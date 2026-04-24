from __future__ import annotations

from collections import defaultdict
from typing import Generic, Iterable, Iterator, TypeVar


T = TypeVar("T")


class OrderedGroups(Generic[T]):
    def __init__(self) -> None:
        self._groups: dict[int, list[T]] = defaultdict(list)
        self._item_to_group: dict[T, int] = {}

    def add_element_to_group(self, element: T, group: int) -> None:
        if element in self._item_to_group:
            self.remove(element)
        self._groups[group].append(element)
        self._item_to_group[element] = group

    def remove(self, element: T) -> None:
        group = self._item_to_group.pop(element)
        self._groups[group].remove(element)
        if not self._groups[group]:
            del self._groups[group]

    def group_id(self, element: T) -> int:
        return self._item_to_group[element]

    def num_groups(self) -> int:
        return len(self._groups)

    def num_elements(self) -> int:
        return len(self._item_to_group)

    def groups(self) -> list[int]:
        return sorted(self._groups)

    def elements_in_group(self, group: int) -> list[T]:
        return list(self._groups[group])

    def ordered_elements(self) -> list[T]:
        return [item for group in self.groups() for item in self._groups[group]]

    def __iter__(self) -> Iterator[T]:
        return iter(self.ordered_elements())

    def update(self, elements: Iterable[T], group: int) -> None:
        for element in elements:
            self.add_element_to_group(element, group)


ParameterBlockOrdering = OrderedGroups

