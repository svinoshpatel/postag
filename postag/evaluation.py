"""Dictionary reference labels and clustering metrics (never model features)."""

import bz2
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure


# Intentionally a documented coarse VESUM grouping, not a Universal Dependencies
# conversion. Participles remain ADJ; adverbial participles have their own class.
POS_MAP = {"noun": "NOUN", "adj": "ADJ", "verb": "VERB", "adv": "ADV",
           "advp": "ADVP", "prep": "PREP", "conj": "CONJ", "part": "PART",
           "intj": "INTJ", "numr": "NUM", "number": "NUM",
           "noninfl": "NONINFL", "onomat": "ONOMAT"}
LOOKUP_POLICY = "lowercase; U+2019/U+02BC apostrophes -> ASCII apostrophe; union all analyses"


def lookup_key(text):
    return text.lower().replace("’", "'").replace("ʼ", "'")


def coarse_pos(tags):
    parts = tags.split(":")
    base = parts[0]
    if base not in POS_MAP:
        raise ValueError(f"Unknown dict_uk POS tag: {base}")
    if "pron" in parts:
        return "PRON"
    if base == "noun" and "prop" in parts:
        return "PROPN"
    if base == "adj" and "numr" in parts:
        return "NUM"
    return POS_MAP[base]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dictionary(path, wanted, source_path=None, dictionary_format="vis"):
    """Stream expanded vis (form tags) or lt (form lemma tags) export.

    Store only requested forms, but read EVERY analysis so ambiguity cannot be
    hidden by early termination. Comments/indentation in vis are ignored; lemma
    header lines are themselves valid surface forms and are included.
    """
    path = Path(path)
    digest = file_sha256(path)
    provenance = {"path": str(path.resolve()), "sha256": digest,
                  "format": dictionary_format, "lookup_policy": LOOKUP_POLICY,
                  "pos_mapping": POS_MAP,
                  "overrides": "pron -> PRON; noun:prop -> PROPN; adj:numr -> NUM"}
    if source_path:
        source = json.loads(Path(source_path).read_text(encoding="utf-8"))
        if source.get("sha256") != digest:
            raise ValueError("Dictionary checksum differs from the source manifest")
        if source.get("format", dictionary_format) != dictionary_format:
            raise ValueError("Dictionary format differs from the source manifest")
        provenance["source"] = source
    if dictionary_format not in ("vis", "lt"):
        raise ValueError("Dictionary format must be vis or lt")
    opener = bz2.open if path.suffix == ".bz2" else open
    lookup = defaultdict(set)
    entries = 0
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            parts = line.split("#", 1)[0].split()
            if not parts:
                continue
            expected = 2 if dictionary_format == "vis" else 3
            if len(parts) != expected:
                raise ValueError(f"{path}:{line_number}: expected {expected} columns "
                                 f"for {dictionary_format} export, got {len(parts)}")
            form, tags = parts[0], parts[-1]
            try:
                pos = coarse_pos(tags)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            entries += 1
            key = lookup_key(form)
            if key in wanted:
                lookup[key].add(pos)
    if not entries:
        raise ValueError("Dictionary contains no analyses")
    provenance["analysis_count"] = entries
    provenance["matched_lookup_forms"] = len(lookup)
    return dict(lookup), provenance


def reference_labels(rows, lookup):
    references = []
    status_counts = Counter()
    distinct = defaultdict(set)
    for row in rows:
        key = lookup_key(row["token"])
        tags = sorted(lookup.get(key, set())) if row["kind"] == "lexical" else []
        if row["kind"] != "lexical":
            status = "nonlexical"
        elif not tags:
            status = "unmatched"
        elif len(tags) > 1:
            status = "ambiguous"
        else:
            status = "eligible"
        references.append({"row_id": row["row_id"], "lookup_key": key,
                           "status": status, "pos_candidates": tags,
                           "pos": tags[0] if status == "eligible" else None})
        status_counts[status] += 1
        distinct[status].add(key)
    lexical = len(rows) - status_counts["nonlexical"]
    lexical_forms = set().union(*(distinct[s] for s in ("eligible", "ambiguous", "unmatched")))
    covered_forms = distinct["eligible"] | distinct["ambiguous"]
    coverage = {"occurrences": {s: status_counts[s] for s in
                                ("eligible", "ambiguous", "unmatched", "nonlexical")},
                "distinct_lookup_forms": {s: len(distinct[s]) for s in
                                          ("eligible", "ambiguous", "unmatched", "nonlexical")},
                "lexical_occurrences": lexical,
                "lexical_forms": len(lexical_forms),
                "dictionary_coverage": ((status_counts["eligible"] + status_counts["ambiguous"])
                                        / lexical if lexical else None),
                "unambiguous_coverage": status_counts["eligible"] / lexical if lexical else None,
                "dictionary_form_coverage": len(covered_forms) / len(lexical_forms)
                if lexical_forms else None}
    return references, coverage


def partition_scores(truth, labels):
    if len(truth) != len(labels):
        raise ValueError("Truth and cluster labels must align")
    if not len(truth):
        return {"status": "unavailable", "reason": "No scored occurrences", "count": 0,
                "v_measure": None, "ari": None, "purity": None}
    groups = defaultdict(Counter)
    for tag, label in zip(truth, labels):
        groups[int(label)][tag] += 1
    purity = sum(max(counts.values()) for counts in groups.values()) / len(truth)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(truth, labels)
    return {"status": "ok", "count": len(truth), "clusters": len(groups),
            "pos_classes": len(set(truth)), "homogeneity": float(homogeneity),
            "completeness": float(completeness), "v_measure": float(v_measure),
            "ari": float(adjusted_rand_score(truth, labels)), "purity": purity}


def evaluate_clusters(references, labels):
    if len(references) != len(labels):
        raise ValueError("References and labels must align")
    truth = [r["pos"] for r in references if r["status"] == "eligible"]
    eligible_labels = np.asarray([int(label) for r, label in zip(references, labels)
                                  if r["status"] == "eligible"], dtype=int)
    assigned = eligible_labels >= 0
    assigned_truth = [tag for tag, keep in zip(truth, assigned) if keep]
    count = len(truth)
    return {"eligible_count": count, "assigned_count": int(assigned.sum()),
            "eligible_noise_fraction": float((~assigned).mean()) if count else None,
            "non_noise": partition_scores(assigned_truth, eligible_labels[assigned]),
            "including_noise_as_one_label": partition_scores(truth, eligible_labels),
            "one_cluster_all_eligible": partition_scores(truth, [0] * count),
            "one_cluster_same_non_noise": partition_scores(assigned_truth, [0] * len(assigned_truth))}


def cluster_details(rows, labels, probabilities, references=None, examples=5):
    """Representative distinct forms plus original token-window context."""
    groups = defaultdict(list)
    for i, label in enumerate(labels):
        groups[int(label)].append(i)
    details = []
    for label, indices in sorted(groups.items()):
        forms = Counter(rows[i]["token"] for i in indices)
        pos = Counter(references[i]["pos"] for i in indices
                      if references and references[i]["status"] == "eligible")
        # Highest membership first; stable row order breaks ties. Noise gets an
        # ordinary illustrative sample, not a claim of representative membership.
        ordered = sorted(indices, key=lambda i: (-probabilities[i], i))
        seen, selected = set(), []
        for i in ordered:
            token = rows[i]["token"]
            if token in seen:
                continue
            seen.add(token)
            selected.append({**rows[i], "membership_strength": float(probabilities[i]),
                             "pos_candidates": references[i]["pos_candidates"] if references else []})
            if len(selected) >= examples:
                break
        details.append({"label": label, "is_noise": label < 0, "size": len(indices),
                        "distinct_forms": len(forms), "top_forms": forms.most_common(10),
                        "token_kinds": dict(Counter(rows[i]["kind"] for i in indices)),
                        "pos_counts": dict(pos),
                        "majority_pos": pos.most_common(1)[0][0] if pos else None,
                        "ending_counts": {suffix: sum(rows[i]["token"].lower().endswith(suffix)
                                                     for i in indices)
                                          for suffix in ("ий", "ій", "ти")},
                        "examples": selected})
    return details
