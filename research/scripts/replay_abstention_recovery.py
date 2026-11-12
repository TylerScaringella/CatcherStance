"""Replay conservative recovery policies against an abstention audit report.

This is deliberately decision-only: it never changes production thresholds or
manufactures a label when identity, timing, or coverage evidence is missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VARIANTS = {
    "baseline": {"min_vote_share": 0.65},
    "review_only_vote_relaxation": {"min_vote_share": 0.60},
}


def replay(rows: list[dict], min_vote_share: float) -> dict:
    recovered = []
    blocked = []
    for row in rows:
        if row.get("accepted"):
            continue
        if row.get("abstention_reason") != "ambiguous_stance_vote":
            blocked.append({"id": row.get("id"), "reason": row.get("abstention_reason")})
            continue
        vote_share = row.get("vote_share")
        safe_context = (
            vote_share is not None
            and float(vote_share) >= min_vote_share
            and row.get("event_available") is True
            and (row.get("window_coverage") or 0.0) >= 0.9
            and (row.get("identity_jumps") or 0) == 0
        )
        if safe_context:
            recovered.append(row.get("id"))
        else:
            blocked.append({"id": row.get("id"), "reason": "insufficient_recovery_evidence"})
    return {
        "min_vote_share": min_vote_share,
        "recovered_count": len(recovered),
        "recovered_ids": recovered,
        "blocked_count": len(blocked),
        "blocked": blocked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, default=Path("research/outputs/generated/abstention_clip_report.jsonl"), nargs="?")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.report.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = {name: replay(rows, values["min_vote_share"]) for name, values in VARIANTS.items()}
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
