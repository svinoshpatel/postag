"""Text-only token features shared by the unsupervised experiments."""

from collections import Counter

# Edit these settings to change the feature set for ALL three word positions.
PREFIX_LENGTHS = (1, 2, 3)
SUFFIX_LENGTHS = (1, 2, 3, 4)

# Each function takes the original token text and returns one feature value.
# Add/remove entries here, e.g. "word.has_hyphen": lambda word: "-" in word.
# Strings are categorical features; booleans are flags; numbers are magnitudes.
WORD_FEATURES = {
    "word.lower()": str.lower,
    "word.length": len,
    "word.istitle()": str.istitle,
    "word.isupper()": str.isupper,
    "word.isdigit()": str.isdigit,
}


def token_text(token):
    """Read text from a plain string or a tutorial-style (word, ...) record.

    This accepts tokenizer output such as ["Київ", "гарний", "."] as well as
    annotated sentences such as [("Київ", "PROPN"), ...]. All annotation fields
    are ignored: the model must never receive the target POS tag as a feature.
    """
    return token if isinstance(token, str) else token[0]


def build_word_frequencies(sentences):
    """Return lowercase token -> relative frequency (TF) for a training corpus.

    TF(word) = number of occurrences of word / total number of tokens.
    All tokens count, including punctuation and numbers. Lowercasing merges
    "Київ" and "київ". An empty corpus gives an empty dictionary.

    Build this once from TRAINING sentences, then reuse the same dictionary for
    training, test data, and future predictions. Do not count test tokens here.
    If using cross-validation, fit this table within each training fold for
    strictly independent validation. Exploratory clustering may use the full
    corpus, provided that choice is recorded in the experiment.
    """
    counts = Counter(
        token_text(token).lower()
        for sentence in sentences
        for token in sentence
    )
    total = sum(counts.values())
    return {word: count / total for word, count in counts.items()}


def token_features(word, word_frequencies):
    """Extract the shared text features for one token, without its context.

    Affixes are lowercase so capitalization has its own separate flags.
    Python slices safely return the whole token if it is shorter than the
    requested affix. Length counts Unicode characters, including apostrophes
    and punctuation. istitle() uses Python's title-case rules, not just the
    first character; isdigit() recognizes digit tokens, not decimals like 3.14.
    """
    features = {name: extract(word) for name, extract in WORD_FEATURES.items()}
    normalized = word.lower()
    for length in PREFIX_LENGTHS:
        features[f"word.prefix{length}"] = normalized[:length]
    for length in SUFFIX_LENGTHS:
        features[f"word.suffix{length}"] = normalized[-length:]
    # Unseen tokens receive 0.0 (no corpus occurrences).
    features["word.tf"] = word_frequencies.get(normalized, 0.0)
    return features


def word2features(sent, i, word_frequencies):
    """Return features for token i and its immediate neighbors in sent.

    `sent` is a sequence of strings or (word, ...) records; `i` is zero-based.
    `word_frequencies` is the table returned by build_word_frequencies().
    Use {} explicitly if no corpus statistics are available (all TFs are zero).

    Current-word keys have no prefix. Neighbor keys have '-1:' or '+1:',
    e.g. '-1:word.suffix2'. All positions use the same token_features() helper,
    so additions there automatically apply to the neighbors as well.
    BOS/EOS mark sentence boundaries; no context crosses those boundaries.
    """
    if not 0 <= i < len(sent):
        raise IndexError("word index must be within the sentence")

    features = {}
    for offset in (-1, 0, 1):
        position = i + offset
        if position < 0:
            features["BOS"] = True
        elif position >= len(sent):
            features["EOS"] = True
        else:
            prefix = "" if offset == 0 else f"{offset:+d}:"
            word = token_text(sent[position])
            features.update({
                prefix + name: value
                for name, value in token_features(word, word_frequencies).items()
            })
    return features


def sent2features(sent, word_frequencies):
    """Return one feature dictionary per token, in original sentence order."""
    return [word2features(sent, i, word_frequencies) for i in range(len(sent))]
