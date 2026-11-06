"""SQL-backed selection summaries; no corpus-sized Python collections."""

import sqlite3
from typing import Any


def selection_summary(
    db: sqlite3.Connection, threshold: float, cap: int
) -> dict[str, object]:
    """Summarize each stage using its actual received-stream denominator."""

    def counts(where: str, params: tuple[object, ...] = ()) -> dict[str, int]:
        sentences, tokens = db.execute(
            f"SELECT COUNT(*), COALESCE(SUM(tokens), 0) FROM pool {where}", params
        ).fetchone()
        return dict(sentences=sentences, tokens=tokens)

    received = counts("")
    eligible = counts("WHERE confidence >= ?", (threshold,))
    selected = counts("WHERE fingerprint IN (SELECT fingerprint FROM selected)")

    def groups(column: str) -> list[dict[str, Any]]:
        result = []
        for key, total, accepted, chosen in db.execute(
            f"SELECT {column}, COUNT(*), SUM(confidence >= ?), SUM(fingerprint IN (SELECT fingerprint FROM selected)) FROM pool GROUP BY {column} ORDER BY {column}",
            (threshold,),
        ):
            result.append(
                dict(
                    value=key,
                    received_sentences=total,
                    eligible_sentences=accepted,
                    selected_sentences=chosen,
                    confidence_acceptance_rate=accepted / total,
                )
            )
        return result

    pos = []
    for label, total, accepted, chosen in db.execute(
        "SELECT label, SUM(n), SUM(n * (confidence >= ?)), SUM(n * (pool.fingerprint IN (SELECT fingerprint FROM selected))) FROM pos JOIN pool USING (fingerprint) GROUP BY label ORDER BY label",
        (threshold,),
    ):
        pos.append(
            dict(
                label=label,
                received_tokens=total,
                eligible_tokens=accepted,
                selected_tokens=chosen,
                confidence_acceptance_rate=accepted / total,
            )
        )
    return dict(
        threshold=threshold,
        received=received,
        eligible=eligible,
        selected=selected,
        confidence_excluded={key: received[key] - eligible[key] for key in received},
        budget_excluded={key: eligible[key] - selected[key] for key in received},
        token_cap=cap,
        cap_utilization=selected["tokens"] / cap if cap else None,
        augmentation_selected=selected["sentences"] > 0,
        by_source=groups("source"),
        by_sentence_length=groups("tokens"),
        by_predicted_pos=pos,
    )
