"""One immutable source/target label contract for all pipeline stages."""

from collections.abc import Sequence
from dataclasses import dataclass

UPOS = frozenset(
    "ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X".split()
)
SERVICE_TAGS = frozenset({"ADP", "CCONJ", "SCONJ", "PART"})


class TagMappingError(ValueError):
    """A mapping, source annotation or target sequence violates the label contract."""


def validate_upos(label: object) -> str:
    """Reject invalid original annotations before any mapping can be applied."""
    if type(label) is not str or label not in UPOS:
        raise TagMappingError("original UPOS: outside fixed UD inventory")
    return label


def _labels(labels: Sequence[str]) -> tuple[str, ...]:
    if (
        isinstance(labels, (str, bytes))
        or not isinstance(labels, Sequence)
        or not labels
    ):
        raise TagMappingError("labels: expected a nonempty sequence")
    return tuple(labels)


@dataclass(frozen=True)
class TagMapping:
    """Canonical, immutable mapping; construction validates even direct tuple inputs."""

    entries: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple:
            raise TagMappingError("tag_mapping: expected immutable tuple entries")
        sources: set[str] = set()
        targets: set[str] = set()
        for pair in self.entries:
            if type(pair) is not tuple or len(pair) != 2:
                raise TagMappingError("tag_mapping: expected (source, target) pairs")
            source, target = pair
            if (
                type(source) is not str
                or source not in SERVICE_TAGS
                or source in sources
            ):
                raise TagMappingError("tag_mapping: invalid or duplicate service tag")
            if (
                type(target) is not str
                or not target
                or "\x00" in target
                or any(char.isspace() for char in target)
                or target in UPOS
            ):
                raise TagMappingError(
                    "tag_mapping: target must be one new nonempty whitespace-free label"
                )
            try:
                target.encode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise TagMappingError(
                    "tag_mapping: target must be valid UTF-8"
                ) from exc
            sources.add(source)
            targets.add(target)
        if self.entries and (sources != SERVICE_TAGS or len(targets) != 1):
            raise TagMappingError(
                "tag_mapping: map all four service tags to one new whitespace-free label"
            )
        object.__setattr__(self, "entries", tuple(sorted(self.entries)))

    @classmethod
    def from_dict(cls, value: object) -> TagMapping:
        """Read the configuration object without coercing keys or target values."""
        if not isinstance(value, dict) or any(type(key) is not str for key in value):
            raise TagMappingError("tag_mapping: expected a mapping with string keys")
        # Runtime validation above owns types; do not stringify/coerce input values.
        return cls(tuple(value.items()))

    @property
    def target_inventory(self) -> tuple[str, ...]:
        mapping = dict(self.entries)
        return tuple(sorted({mapping.get(label, label) for label in UPOS}))

    def map_upos(self, labels: Sequence[str]) -> tuple[str, ...]:
        """Validate all original UPOS labels, then return aligned mapped targets."""
        original = tuple(validate_upos(label) for label in _labels(labels))
        mapping = dict(self.entries)
        return tuple(mapping.get(label, label) for label in original)

    def validate_targets(self, labels: Sequence[str]) -> tuple[str, ...]:
        """Validate predictions/pseudo labels as targets, without mapping them."""
        targets = _labels(labels)
        inventory = self.target_inventory
        if any(type(label) is not str or label not in inventory for label in targets):
            raise TagMappingError("target_labels: outside configured target inventory")
        return targets
