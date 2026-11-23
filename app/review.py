"""Read-only review of an archived run against an ordered stage recipe.

The cold archive never stores raw readings.  For every measured minute the
review therefore works from a *range* of possible original readings:

* a retained minute keeps its original reading exactly, so its range is a
  single value;
* any other measured minute was reconstructed as ``r`` with a guaranteed
  error of at most ``tolerance``; the original reading must also lie inside
  the temperature range the API accepts and must keep the same process-band
  classification as ``r`` (the archive guarantees that classification).

A review places 1..6 holding stages in the given order.  Every stage takes
exactly ``hold_minutes`` consecutive measured minutes (a missing minute can
never be part of a stage).  Stages do not overlap, and the number of minutes
between one stage's end and the next stage's start must not exceed
``max_gap_minutes``; missing minutes inside such a gap still count against
that allowance.

Verdicts:

``confirmed``
    A window sequence exists and every minute's whole range lies inside its
    stage band.
``insufficient_evidence``
    No fully-contained sequence exists, but a sequence exists in which every
    minute's range at least intersects its stage band.  The archive cannot
    prove the recipe held, nor rule it out.
``not_met``
    No ordered sequence satisfies even the intersection condition.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .planner import MAX_RUN_MINUTES, classify

# Inclusive temperature range the API accepts, in tenths of a degree Celsius.
MIN_TENTHS_C = -10_000
MAX_TENTHS_C = 50_000

MAX_REVIEW_STAGES = 6

CONFIRMED = "confirmed"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
NOT_MET = "not_met"

VERDICT_TEXT = {
    CONFIRMED: "确认达标",
    INSUFFICIENT_EVIDENCE: "证据不足",
    NOT_MET: "不达标",
}


@dataclass(frozen=True)
class ValueRange:
    """Inclusive range of the original reading of one measured minute."""

    low: int
    high: int

    def fully_inside(self, band_low: int, band_high: int) -> bool:
        return band_low <= self.low and self.high <= band_high

    def intersects(self, band_low: int, band_high: int) -> bool:
        return self.high >= band_low and self.low <= band_high


def infer_value_range(
    reconstructed: int,
    *,
    exact: bool,
    tolerance: int,
    band_low: int,
    band_high: int,
) -> ValueRange:
    """Infer the original reading's range for one measured minute.

    ``exact`` is true for retained minutes, whose original value is stored.
    Otherwise the range is bounded by the reconstruction error, the API's
    accepted temperature range and the original band classification (which
    matches the reconstructed value's classification).
    """
    if exact:
        return ValueRange(reconstructed, reconstructed)

    low = max(reconstructed - tolerance, MIN_TENTHS_C)
    high = min(reconstructed + tolerance, MAX_TENTHS_C)
    state = classify(reconstructed, band_low, band_high)
    if state < 0:
        high = min(high, band_low - 1)
    elif state > 0:
        low = max(low, band_high + 1)
    else:
        low = max(low, band_low)
        high = min(high, band_high)
    # ``reconstructed`` itself satisfies every bound above, so low <= high.
    return ValueRange(low, high)


def infer_value_ranges(
    values: Sequence[Optional[int]],
    anchors: Mapping[int, int],
    *,
    tolerance: int,
    band_low: int,
    band_high: int,
) -> list[Optional[ValueRange]]:
    """Infer value ranges for a whole run; ``None`` marks a missing minute."""
    return [
        None
        if value is None
        else infer_value_range(
            value,
            exact=offset in anchors,
            tolerance=tolerance,
            band_low=band_low,
            band_high=band_high,
        )
        for offset, value in enumerate(values)
    ]


@dataclass(frozen=True)
class StageSpec:
    """One holding stage of the recipe under review."""

    band_low_tenths_c: int
    band_high_tenths_c: int
    hold_minutes: int


@dataclass(frozen=True)
class ReviewWindow:
    """A chosen stage window, as a run offset and a length."""

    stage_index: int  # 1-based position in the recipe
    start_offset: int
    hold_minutes: int


@dataclass(frozen=True)
class ReviewOutcome:
    verdict: str
    windows: tuple[ReviewWindow, ...]
    failed_stage_index: Optional[int]
    reason: str


def _valid_starts(
    ranges: Sequence[Optional[ValueRange]],
    stage: StageSpec,
    *,
    require_full_containment: bool,
) -> list[int]:
    """Earliest-to-latest offsets where this stage can be placed."""
    minute_count = len(ranges)
    hold = stage.hold_minutes
    if hold <= 0 or hold > minute_count:
        return []
    starts: list[int] = []
    for start in range(minute_count - hold + 1):
        good = True
        for offset in range(start, start + hold):
            value_range = ranges[offset]
            if value_range is None:
                good = False
                break
            if require_full_containment:
                inside = value_range.fully_inside(
                    stage.band_low_tenths_c, stage.band_high_tenths_c
                )
            else:
                inside = value_range.intersects(
                    stage.band_low_tenths_c, stage.band_high_tenths_c
                )
            if not inside:
                good = False
                break
        if good:
            starts.append(start)
    return starts


def _find_chain(
    starts_per_stage: Sequence[Sequence[int]],
    holds: Sequence[int],
    max_gap_minutes: int,
) -> Optional[list[int]]:
    """Find an ordered, non-overlapping window sequence, earliest first.

    ``feasible[i]`` holds the starts of stage ``i`` from which every later
    stage can still be placed.  Building it right-to-left, then walking
    left-to-right and taking the earliest start that stays feasible, yields
    the lexicographically earliest whole sequence.
    """
    stage_count = len(starts_per_stage)
    if stage_count == 0:
        return None

    feasible: list[list[int]] = [[] for _ in range(stage_count)]
    feasible[-1] = list(starts_per_stage[-1])
    for index in range(stage_count - 2, -1, -1):
        later = feasible[index + 1]
        current: list[int] = []
        for start in starts_per_stage[index]:
            earliest_next = start + holds[index]
            latest_next = earliest_next + max_gap_minutes
            position = bisect_left(later, earliest_next)
            if position < len(later) and later[position] <= latest_next:
                current.append(start)
        feasible[index] = current

    if not feasible[0]:
        return None

    chain = [feasible[0][0]]
    for index in range(1, stage_count):
        earliest_next = chain[-1] + holds[index - 1]
        latest_next = earliest_next + max_gap_minutes
        position = bisect_left(feasible[index], earliest_next)
        # feasible[0..index-1] were built from this exact list, so a start
        # inside [earliest_next, latest_next] is guaranteed to be present.
        chain.append(feasible[index][position])
        assert chain[-1] <= latest_next
    return chain


def review_run(
    ranges: Sequence[Optional[ValueRange]],
    stages: Sequence[StageSpec],
    max_gap_minutes: int,
) -> ReviewOutcome:
    """Review inferred minute ranges against an ordered multi-stage recipe."""
    if not 1 <= len(stages) <= MAX_REVIEW_STAGES:
        raise ValueError(f"a review takes 1..{MAX_REVIEW_STAGES} stages")
    if max_gap_minutes < 0:
        raise ValueError("max_gap_minutes must be non-negative")
    if len(ranges) > MAX_RUN_MINUTES:
        raise ValueError(f"a run has at most {MAX_RUN_MINUTES} minutes")
    holds = [stage.hold_minutes for stage in stages]

    for require_full_containment in (True, False):
        starts_per_stage = [
            _valid_starts(
                ranges,
                stage,
                require_full_containment=require_full_containment,
            )
            for stage in stages
        ]
        chain = _find_chain(starts_per_stage, holds, max_gap_minutes)
        if chain is None:
            continue
        windows = tuple(
            ReviewWindow(
                stage_index=index + 1,
                start_offset=start,
                hold_minutes=holds[index],
            )
            for index, start in enumerate(chain)
        )
        if require_full_containment:
            return ReviewOutcome(
                verdict=CONFIRMED,
                windows=windows,
                failed_stage_index=None,
                reason=(
                    "every minute in the chosen windows has its whole "
                    "inferred original-value range inside the corresponding "
                    "stage band"
                ),
            )
        return ReviewOutcome(
            verdict=INSUFFICIENT_EVIDENCE,
            windows=windows,
            failed_stage_index=None,
            reason=(
                "the stages can be placed, but at least one minute's "
                "inferred original-value range only intersects its stage "
                "band; the archive cannot prove or disprove compliance"
            ),
        )

    # No sequence at all: report the first stage that cannot be placed on
    # its own, otherwise the inter-stage gap constraint.
    starts_per_stage = [
        _valid_starts(ranges, stage, require_full_containment=False)
        for stage in stages
    ]
    for index, starts in enumerate(starts_per_stage):
        if not starts:
            return ReviewOutcome(
                verdict=NOT_MET,
                windows=(),
                failed_stage_index=index + 1,
                reason=(
                    f"stage {index + 1} has no block of {holds[index]} "
                    "consecutive measured minutes whose inferred value "
                    "range intersects its band"
                ),
            )
    return ReviewOutcome(
        verdict=NOT_MET,
        windows=(),
        failed_stage_index=None,
        reason=(
            "the stages cannot be placed in the given order while keeping "
            f"at most {max_gap_minutes} minute(s) between adjacent stages"
        ),
    )
