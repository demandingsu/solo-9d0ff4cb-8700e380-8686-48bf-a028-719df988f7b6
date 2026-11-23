"""Tests for the read-only holding-stage review.

Covers the inferred original-value ranges, ordering / adjacency / gap rules,
missing minutes, all three verdicts, error paths and read-only behaviour.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterator, Optional

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.review import (
    CONFIRMED,
    INSUFFICIENT_EVIDENCE,
    NOT_MET,
    StageSpec,
    ValueRange,
    infer_value_range,
    infer_value_ranges,
    review_run,
)


@pytest.fixture()
def store_path(tmp_path: Any) -> str:
    return str(tmp_path / "review.sqlite3")


@pytest.fixture()
def client(store_path: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("FURNACE_DB_PATH", store_path)
    with TestClient(create_app()) as test_client:
        yield test_client


def _archive_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": "HT-REVIEW",
        "start_minute_utc": 1_000,
        "readings": [800] * 20,
        "band_low_tenths_c": 700,
        "band_high_tenths_c": 900,
        "tolerance_tenths_c": 0,
        "max_retained_points": 20,
    }
    payload.update(overrides)
    return payload


def _ranges(values: list[Optional[int]]) -> list[Optional[ValueRange]]:
    return [None if value is None else ValueRange(value, value) for value in values]


def _stage(low: int, high: int, hold: int) -> StageSpec:
    return StageSpec(band_low_tenths_c=low, band_high_tenths_c=high, hold_minutes=hold)


def _window_tuples(outcome: Any) -> list[tuple[int, int, int]]:
    return [
        (window.stage_index, window.start_offset, window.hold_minutes)
        for window in outcome.windows
    ]


# ---------------------------------------------------------------------------
# Range inference
# ---------------------------------------------------------------------------


def test_retained_minute_keeps_its_exact_reading() -> None:
    value_range = infer_value_range(
        812, exact=True, tolerance=50, band_low=700, band_high=900
    )
    assert value_range == ValueRange(812, 812)


def test_range_is_clipped_by_tolerance_and_original_band() -> None:
    inside = infer_value_range(
        800, exact=False, tolerance=10, band_low=780, band_high=820
    )
    assert inside == ValueRange(790, 810)

    # Below the original band, so the original must stay strictly below it.
    below = infer_value_range(
        770, exact=False, tolerance=10, band_low=780, band_high=820
    )
    assert below == ValueRange(760, 779)

    # Above the original band, so the original must stay strictly above it.
    above = infer_value_range(
        821, exact=False, tolerance=10, band_low=780, band_high=820
    )
    assert above == ValueRange(821, 831)


def test_range_is_clipped_by_the_api_temperature_limits() -> None:
    low = infer_value_range(
        -9_990, exact=False, tolerance=100, band_low=-10_000, band_high=0
    )
    assert low == ValueRange(-10_000, -9_890)
    high = infer_value_range(
        49_990, exact=False, tolerance=100, band_low=0, band_high=50_000
    )
    assert high == ValueRange(49_890, 50_000)


def test_infer_ranges_marks_missing_minutes() -> None:
    ranges = infer_value_ranges(
        [800, None, 800],
        {0: 800},
        tolerance=10,
        band_low=780,
        band_high=820,
    )
    assert ranges[0] == ValueRange(800, 800)
    assert ranges[1] is None
    assert ranges[2] == ValueRange(790, 810)


# ---------------------------------------------------------------------------
# Planner rules
# ---------------------------------------------------------------------------


def test_confirmed_places_stages_in_order_and_prefers_earliest() -> None:
    outcome = review_run(
        _ranges([800] * 12), [_stage(800, 800, 3), _stage(800, 800, 3)], 0
    )
    assert outcome.verdict == CONFIRMED
    # max_gap_minutes == 0 forces the second stage to start right after the
    # first, and the earliest whole chain is chosen.
    assert _window_tuples(outcome) == [(1, 0, 3), (2, 3, 3)]


def test_stage_order_is_fixed() -> None:
    # Only the low plateau occurs first and the high plateau second, so the
    # recipe is satisfiable as written but not when the stages are reversed.
    ranges = _ranges([100] * 5 + [300] * 5)
    forward = review_run(ranges, [_stage(100, 100, 2), _stage(300, 300, 2)], 100)
    assert forward.verdict == CONFIRMED
    reversed_outcome = review_run(
        ranges, [_stage(300, 300, 2), _stage(100, 100, 2)], 100
    )
    assert reversed_outcome.verdict == NOT_MET


def test_gap_allowance_is_inclusive_and_counts_missing_minutes() -> None:
    # Measured minutes 0..9 at 800, then five missing minutes, then 15..19 at
    # 900.  The only place the second stage can go is after the gap.
    values: list[Optional[int]] = [800] * 10 + [None] * 5 + [900] * 5
    stages = [_stage(800, 800, 3), _stage(900, 900, 3)]

    # The earliest stage-1 window ends at minute 9; stage 2 starts at 15, so
    # exactly the five missing minutes fall into the gap.
    outcome = review_run(_ranges(values), stages, 5)
    assert outcome.verdict == CONFIRMED
    assert _window_tuples(outcome) == [(1, 7, 3), (2, 15, 3)]

    # One minute less and no chain exists: missing minutes count in the gap.
    tight = review_run(_ranges(values), stages, 4)
    assert tight.verdict == NOT_MET
    assert tight.windows == ()
    assert tight.failed_stage_index is None
    assert "order" in tight.reason


def test_missing_minute_cannot_be_part_of_a_stage() -> None:
    values: list[Optional[int]] = [800, 800, None, 800, 800]
    outcome = review_run(_ranges(values), [_stage(800, 800, 4)], 10)
    assert outcome.verdict == NOT_MET
    assert outcome.failed_stage_index == 1


def test_stages_may_not_overlap() -> None:
    # Two 6-minute stages cannot share minutes even with a generous gap.
    outcome = review_run(
        _ranges([800] * 6), [_stage(800, 800, 6), _stage(800, 800, 6)], 100
    )
    assert outcome.verdict == NOT_MET


def test_insufficient_evidence_when_only_intersection_holds() -> None:
    # Minute 0 is exact; the rest can be 790..810.
    ranges = [ValueRange(800, 800)] + [ValueRange(790, 810)] * 5
    outcome = review_run(ranges, [_stage(795, 805, 4)], 5)
    assert outcome.verdict == INSUFFICIENT_EVIDENCE
    assert _window_tuples(outcome) == [(1, 0, 4)]


def test_confirmed_outranks_insufficient_evidence() -> None:
    ranges = [ValueRange(800, 800)] + [ValueRange(790, 810)] * 5
    assert review_run(ranges, [_stage(780, 820, 4)], 5).verdict == CONFIRMED


def test_not_met_when_no_intersection_exists() -> None:
    outcome = review_run(
        [ValueRange(800, 800)] * 5, [_stage(850, 860, 2)], 5
    )
    assert outcome.verdict == NOT_MET
    assert outcome.failed_stage_index == 1


def test_stage_count_is_bounded() -> None:
    with pytest.raises(ValueError):
        review_run(_ranges([800] * 5), [], 0)
    with pytest.raises(ValueError):
        review_run(_ranges([800] * 30), [_stage(800, 800, 1)] * 7, 0)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_review_confirmed_returns_utc_windows(client: TestClient) -> None:
    created = client.post("/api/v1/archives", json=_archive_payload())
    assert created.status_code == 201, created.text

    response = client.post(
        "/api/v1/archives/HT-REVIEW/review",
        json={
            "stages": [
                {"band_low_tenths_c": 800, "band_high_tenths_c": 800, "hold_minutes": 3},
                {"band_low_tenths_c": 800, "band_high_tenths_c": 800, "hold_minutes": 2},
            ],
            "max_gap_minutes": 0,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["verdict"] == "confirmed"
    assert body["verdict_text"] == "确认达标"
    assert body["stage_windows"] == [
        {
            "stage_index": 1,
            "start_minute_utc": 1_000,
            "end_minute_utc": 1_002,
            "hold_minutes": 3,
            "band_low_tenths_c": 800,
            "band_high_tenths_c": 800,
        },
        {
            "stage_index": 2,
            "start_minute_utc": 1_003,
            "end_minute_utc": 1_004,
            "hold_minutes": 2,
            "band_low_tenths_c": 800,
            "band_high_tenths_c": 800,
        },
    ]


def test_review_insufficient_evidence_from_tolerance(client: TestClient) -> None:
    # tolerance=10 leaves the non-retained minutes as 790..810.
    client.post(
        "/api/v1/archives",
        json=_archive_payload(
            run_id="HT-TOL",
            readings=[800] * 10,
            tolerance_tenths_c=10,
            max_retained_points=2,
        ),
    )
    response = client.post(
        "/api/v1/archives/HT-TOL/review",
        json={
            "stages": [
                {"band_low_tenths_c": 795, "band_high_tenths_c": 805, "hold_minutes": 3}
            ],
            "max_gap_minutes": 0,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["verdict"] == "insufficient_evidence"
    assert body["verdict_text"] == "证据不足"
    assert body["stage_windows"][0]["start_minute_utc"] == 1_000
    assert body["stage_windows"][0]["end_minute_utc"] == 1_002

    # A band that touches none of the ranges is a clear failure.
    missed = client.post(
        "/api/v1/archives/HT-TOL/review",
        json={
            "stages": [
                {"band_low_tenths_c": 850, "band_high_tenths_c": 860, "hold_minutes": 3}
            ],
            "max_gap_minutes": 0,
        },
    )
    assert missed.json()["verdict"] == "not_met"
    assert missed.json()["stage_windows"] == []


def test_review_gap_counts_missing_minutes_over_http(client: TestClient) -> None:
    readings: list[Optional[int]] = [800] * 10 + [None] * 5 + [900] * 5
    client.post(
        "/api/v1/archives",
        json=_archive_payload(
            run_id="HT-GAP-REVIEW", readings=readings, max_retained_points=4
        ),
    )
    stages = [
        {"band_low_tenths_c": 800, "band_high_tenths_c": 800, "hold_minutes": 3},
        {"band_low_tenths_c": 900, "band_high_tenths_c": 900, "hold_minutes": 3},
    ]
    ok = client.post(
        "/api/v1/archives/HT-GAP-REVIEW/review",
        json={"stages": stages, "max_gap_minutes": 5},
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["verdict"] == "confirmed"
    assert [(w["start_minute_utc"], w["end_minute_utc"]) for w in ok.json()["stage_windows"]] == [
        (1_007, 1_009),
        (1_015, 1_017),
    ]

    tight = client.post(
        "/api/v1/archives/HT-GAP-REVIEW/review",
        json={"stages": stages, "max_gap_minutes": 4},
    )
    assert tight.json()["verdict"] == "not_met"


def test_review_is_read_only(client: TestClient, store_path: str) -> None:
    client.post("/api/v1/archives", json=_archive_payload())
    before = client.get("/api/v1/archives/HT-REVIEW").json()

    response = client.post(
        "/api/v1/archives/HT-REVIEW/review",
        json={
            "stages": [
                {"band_low_tenths_c": 700, "band_high_tenths_c": 900, "hold_minutes": 5}
            ],
            "max_gap_minutes": 0,
        },
    )
    assert response.status_code == 200

    after = client.get("/api/v1/archives/HT-REVIEW").json()
    assert after == before
    connection = sqlite3.connect(store_path)
    try:
        count = connection.execute(
            "SELECT COUNT(*) FROM furnace_archives"
        ).fetchone()[0]
    finally:
        connection.close()
    assert count == 1


def test_review_unknown_run_returns_404(client: TestClient) -> None:
    response = client.post(
        "/api/v1/archives/NOPE/review",
        json={
            "stages": [
                {"band_low_tenths_c": 700, "band_high_tenths_c": 900, "hold_minutes": 1}
            ],
            "max_gap_minutes": 0,
        },
    )
    assert response.status_code == 404
    assert response.json()["reason_code"] == "archive_not_found"


def test_review_corrupt_archive_returns_500(client: TestClient, store_path: str) -> None:
    client.post("/api/v1/archives", json=_archive_payload())
    connection = sqlite3.connect(store_path)
    try:
        row = connection.execute(
            "SELECT retained_points_json FROM furnace_archives WHERE run_id = ?",
            ("HT-REVIEW",),
        ).fetchone()
        points = json.loads(row[0])
        points[0][1] += 1
        connection.execute(
            "UPDATE furnace_archives SET retained_points_json = ? WHERE run_id = ?",
            (json.dumps(points), "HT-REVIEW"),
        )
        connection.commit()
    finally:
        connection.close()

    response = client.post(
        "/api/v1/archives/HT-REVIEW/review",
        json={
            "stages": [
                {"band_low_tenths_c": 700, "band_high_tenths_c": 900, "hold_minutes": 1}
            ],
            "max_gap_minutes": 0,
        },
    )
    assert response.status_code == 500
    assert response.json()["reason_code"] == "archive_corrupt"


@pytest.mark.parametrize(
    "payload",
    [
        {"stages": [], "max_gap_minutes": 0},
        {
            "stages": [
                {"band_low_tenths_c": 800, "band_high_tenths_c": 800, "hold_minutes": 1}
            ]
            * 7,
            "max_gap_minutes": 0,
        },
        {
            "stages": [
                {"band_low_tenths_c": 800, "band_high_tenths_c": 800, "hold_minutes": 0}
            ],
            "max_gap_minutes": 0,
        },
        {
            "stages": [
                {"band_low_tenths_c": 900, "band_high_tenths_c": 800, "hold_minutes": 1}
            ],
            "max_gap_minutes": 0,
        },
        {
            "stages": [
                {"band_low_tenths_c": 800, "band_high_tenths_c": 800, "hold_minutes": 1}
            ],
            "max_gap_minutes": -1,
        },
    ],
)
def test_review_invalid_requests_are_422(client: TestClient, payload: Any) -> None:
    client.post("/api/v1/archives", json=_archive_payload())
    response = client.post("/api/v1/archives/HT-REVIEW/review", json=payload)
    assert response.status_code == 422
    assert response.json()["reason_code"] == "invalid_request"
