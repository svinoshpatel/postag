"""Full-precision Viterbi-label confidence and streaming sentence acceptance."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from crf.data.labels import TagMapping
from crf.data.literature import RawSentence
from crf.data.segmentation import segmentation_metadata
from crf.features.surface import sentence_features
from crf.utils.config import PipelineConfig
from crf.utils.resources import MemoryMonitor, check_memory
from crf.utils.serialization import ValidationError, probability


@dataclass(frozen=True)
class SentenceConfidence:
    """Aligned chosen labels and marginals; empty sentences are ineligible."""

    labels: tuple[str, ...]
    token_confidences: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.token_confidences):
            raise ValidationError("Confidence arrays must align")
        for label in self.labels:
            if type(label) is not str or not label:
                raise ValidationError("Invalid predicted label")
        for value in self.token_confidences:
            probability(value, "token confidence")

    @property
    def min_confidence(self) -> float | None:
        return min(self.token_confidences, default=None)

    def accepts(self, threshold: float) -> bool:
        threshold = probability(threshold, "confidence threshold")
        minimum = self.min_confidence
        return minimum is not None and minimum >= threshold


@dataclass(frozen=True)
class ConfidenceCandidate:
    """Keep original sentence provenance, forms and order together with scores."""

    sentence: RawSentence
    confidence: SentenceConfidence


def viterbi_confidence(
    labels: Sequence[str],
    marginals: Sequence[Mapping[str, float]],
    *,
    target_inventory: Sequence[str],
) -> SentenceConfidence:
    """Read the marginal of each chosen label, never the largest marginal."""
    if len(labels) != len(marginals):
        raise ValidationError("Predictions and marginals must align")
    inventory = frozenset(target_inventory)
    confidences: list[float] = []
    for label, row in zip(labels, marginals, strict=True):
        if label not in inventory or label not in row:
            raise ValidationError("Missing or unknown Viterbi label")
        for tag, value in row.items():
            if tag not in inventory:
                raise ValidationError("Marginal label outside target inventory")
            probability(value, "label marginal")
        confidences.append(probability(row[label], "Viterbi marginal"))
    return SentenceConfidence(tuple(labels), tuple(confidences))


def score_sentence(
    forms: Sequence[str],
    *,
    model: Any,
    config: PipelineConfig,
    monitor: MemoryMonitor | None = None,
) -> SentenceConfidence:
    """Score one bounded sequence with the gate's recorded configuration.

    Caller must supply the gate estimator and its matching bundle configuration.
    A shared monitor remains caller-owned; results are provisional until its exit.
    """
    check_memory(monitor, expected_budget_mib=config.resources.memory_budget_mib)
    if not forms:
        return SentenceConfidence((), ())
    features = sentence_features(
        forms,
        schema_version=config.features.schema_version,
        normalizer_version=config.preprocessing.normalizer_version,
    )
    labels = tuple(model.predict_single(features))
    check_memory(monitor, expected_budget_mib=config.resources.memory_budget_mib)
    marginals = model.predict_marginals_single(features)
    check_memory(monitor, expected_budget_mib=config.resources.memory_budget_mib)
    if len(labels) != len(forms):
        raise ValidationError("Predictions must preserve every token")
    return viterbi_confidence(
        labels,
        marginals,
        target_inventory=TagMapping(config.tag_mapping).target_inventory,
    )


def iter_confidence_candidates(
    sentences: Iterable[RawSentence],
    *,
    model: Any,
    config: PipelineConfig,
    monitor: MemoryMonitor | None = None,
) -> Iterator[ConfidenceCandidate]:
    """Stream the pool for all thresholds after caller-owned contamination checks.

    No sampling, output publication or accumulation is performed here. Every
    retained candidate can be tested independently against each frozen threshold.
    """
    thresholds = config.self_training.thresholds
    if not thresholds:
        raise ValidationError("Confidence threshold grid must not be empty")
    for threshold in thresholds:
        probability(threshold, "confidence threshold")
    minimum = min(thresholds)
    expected = segmentation_metadata(
        adapter_version=config.preprocessing.tokenizer_adapter_version,
        normalizer_version=config.preprocessing.normalizer_version,
    )
    for sentence in sentences:
        if sentence.source.source_role != "literature":
            raise ValidationError("Pseudo-label confidence requires literature")
        if sentence.preprocessing != expected:
            raise ValidationError("Sentence preprocessing disagrees with gate")
        confidence = score_sentence(
            sentence.feature_forms, model=model, config=config, monitor=monitor
        )
        if confidence.accepts(minimum):
            yield ConfidenceCandidate(sentence, confidence)
