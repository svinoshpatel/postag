"""Read saved tokenizer output and sample occurrences without losing context."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import unicodedata

from .features import word2features


def token_kind(text):
    if text.isdigit():
        return "number"
    if any(c.isalpha() for c in text):
        return "lexical"
    if all(unicodedata.category(c).startswith("P") for c in text):
        return "punctuation"
    return "other"


def read_document(path):
    """Validate the tokenizer contract, including empty documents/sentences."""
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"{path}: invalid UTF-8 JSON: {error}") from error
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of sentence objects")
    for i, item in enumerate(data):
        if (not isinstance(item, dict) or not isinstance(item.get("sentence"), str)
                or not isinstance(item.get("tokens"), list)
                or any(not isinstance(t, str) or not t or t.isspace()
                       for t in item["tokens"])):
            raise ValueError(f"{path}: invalid sentence/tokens at index {i}")
    return data, hashlib.sha256(raw).hexdigest()


def scan_corpus(root, row_selection="all"):
    """First pass: full-corpus TF, document counts, and exact input hashes."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Input directory does not exist: {root}")
    files = sorted(p for p in root.rglob("*.json")
                   if p.is_file() and not p.is_symlink()
                   and not any(parent.is_symlink() for parent in p.parents))
    if not files:
        raise ValueError(f"No tokenized JSON files in {root}")
    if row_selection not in ("all", "lexical"):
        raise ValueError("row_selection must be all or lexical")
    counts, kinds, documents = Counter(), Counter(), []
    for path in files:
        data, digest = read_document(path)
        total = eligible = 0
        for item in data:
            for token in item["tokens"]:
                counts[token.lower()] += 1
                kind = token_kind(token)
                kinds[kind] += 1
                total += 1
                eligible += row_selection == "all" or kind == "lexical"
        documents.append({"path": path.relative_to(root).as_posix(),
                          "sha256": digest, "sentences": len(data),
                          "tokens": total, "eligible_tokens": eligible})
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Corpus has no tokens")
    frequencies = {word: count / total for word, count in counts.items()}
    manifest = {"documents": documents, "tokens": total,
                "sentences": sum(d["sentences"] for d in documents),
                "unique_lowercase_tokens": len(counts), "token_kinds": dict(kinds),
                "tf_scope": "full corpus, lowercase, all tokens including punctuation",
                "row_selection": row_selection}
    return frequencies, manifest


def sample_quotas(capacities, limit, seed):
    """Approximately equal document quotas, redistributing short documents' share.

    This is a document-balanced exploratory sample, NOT a uniform corpus sample.
    limit=0 explicitly selects all eligible occurrences.
    """
    if limit < 0:
        raise ValueError("sample size cannot be negative")
    remaining = min(limit or sum(capacities), sum(capacities))
    quotas = [0] * len(capacities)
    order = list(range(len(capacities)))
    random.Random(seed).shuffle(order)
    while remaining:
        active = [i for i in order if quotas[i] < capacities[i]]
        share = max(1, remaining // len(active))
        for i in active:
            take = min(share, capacities[i] - quotas[i], remaining)
            quotas[i] += take
            remaining -= take
    return quotas


def sample_rows(root, manifest, frequencies, limit=10000, seed=42):
    """Second pass: original-position features and stable aligned metadata."""
    root = Path(root)
    documents = manifest["documents"]
    quotas = sample_quotas([d["eligible_tokens"] for d in documents], limit, seed)
    rows, features = [], []
    for info, quota in zip(documents, quotas):
        data, digest = read_document(root / info["path"])
        if digest != info["sha256"]:
            raise ValueError(f"Corpus changed between passes: {info['path']}")
        # Seed each document independently so results never depend on hash randomization.
        rng = random.Random(f"{seed}:{info['path']}")
        selected = set(rng.sample(range(info["eligible_tokens"]), quota))
        eligible_index = 0
        for sentence_index, item in enumerate(data):
            tokens = item["tokens"]
            for token_index, token in enumerate(tokens):
                kind = token_kind(token)
                if manifest["row_selection"] == "lexical" and kind != "lexical":
                    continue
                chosen = eligible_index in selected
                eligible_index += 1
                if not chosen:
                    continue
                start, end = max(0, token_index - 8), min(len(tokens), token_index + 9)
                rows.append({
                    "row_id": json.dumps([info["path"], sentence_index, token_index],
                                         ensure_ascii=False, separators=(",", ":")),
                    "document": info["path"], "sentence_index": sentence_index,
                    "token_index": token_index, "token": token, "kind": kind,
                    "context_tokens": tokens[start:end], "context_start": start,
                    "sentence_token_count": len(tokens),
                })
                features.append(word2features(tokens, token_index, frequencies))
    if not rows:
        raise ValueError("No tokens match the requested row selection")
    return rows, features
