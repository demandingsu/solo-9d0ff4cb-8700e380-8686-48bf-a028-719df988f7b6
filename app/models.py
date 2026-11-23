"""Request and response models for the furnace archive API."""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, StringConstraints, model_validator

RunId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=64),
]

TenthsC = Annotated[int, Field(ge=-10_000, le=50_000)]


class ArchiveCreateRequest(BaseModel):
    """One furnace run to compress and archive."""

    run_id: RunId = Field(
        description="Furnace run number, unique across the archive.",
        examples=["HT-2026-0042"],
    )
    start_minute_utc: Annotated[
        int,
        Field(
            ge=0,
            le=10_000_000_000,
            description="UTC minute of the first reading, minutes since epoch.",
        ),
    ]
    readings: Annotated[
        list[Optional[TenthsC]],
        Field(
            min_length=1,
            max_length=360,
            description=(
                "Per-minute furnace temperature in tenths of a degree Celsius, "
                "starting at start_minute_utc. null marks a missing sample."
            ),
        ),
    ]
    band_low_tenths_c: TenthsC = Field(
        description="Lower process-band limit, inclusive."
    )
    band_high_tenths_c: TenthsC = Field(
        description="Upper process-band limit, inclusive."
    )
    tolerance_tenths_c: Annotated[
        int,
        Field(
            ge=0,
            le=5000,
            description="Largest allowed reconstruction error, in tenths C.",
        ),
    ]
    max_retained_points: Annotated[
        int,
        Field(
            ge=0,
            le=360,
            description="Maximum retained points allowed for the whole run.",
        ),
    ]

    @model_validator(mode="after")
    def _check_band_and_samples(self) -> "ArchiveCreateRequest":
        if self.band_low_tenths_c > self.band_high_tenths_c:
            raise ValueError(
                "band_low_tenths_c must not exceed band_high_tenths_c"
            )
        if all(value is None for value in self.readings):
            raise ValueError("readings must contain at least one measured minute")
        return self


class RetainedPoint(BaseModel):
    minute_utc: int
    value_tenths_c: int


class ArchiveSummary(BaseModel):
    run_id: str
    start_minute_utc: int
    end_minute_utc: int
    minute_count: int
    band_low_tenths_c: int
    band_high_tenths_c: int
    tolerance_tenths_c: int
    max_retained_points: int
    retained_points_count: int
    measured_minutes: int
    missing_minutes: int
    segment_count: int
    payload_sha256: str
    planner_version: str
    created_at_utc: str


class ArchiveCreatedResponse(BaseModel):
    status: str = "archived"
    archive: ArchiveSummary


class MinuteSample(BaseModel):
    minute_utc: int
    value_tenths_c: Optional[int]
    missing: bool
    band_state: str = Field(description="below | in_band | above; 'unknown' if missing")


class BandInterval(BaseModel):
    start_minute_utc: int
    end_minute_utc: int
    duration_minutes: int


class ArchiveQueryResponse(BaseModel):
    run_id: str
    start_minute_utc: int
    end_minute_utc: int
    band_low_tenths_c: int
    band_high_tenths_c: int
    tolerance_tenths_c: int
    max_retained_points: int
    retained_points_count: int
    query_start_minute_utc: int
    query_end_minute_utc: int
    minutes: list[MinuteSample]
    in_band_intervals: list[BandInterval]
    missing_minutes: list[int]
    payload_sha256: str


class ArchiveListItem(BaseModel):
    run_id: str
    start_minute_utc: int
    end_minute_utc: int
    minute_count: int
    retained_points_count: int
    created_at_utc: str


class ArchiveListResponse(BaseModel):
    archives: list[ArchiveListItem]


class ReviewStage(BaseModel):
    """One holding stage of the recipe under review."""

    band_low_tenths_c: TenthsC = Field(
        description="Lower stage band limit, inclusive."
    )
    band_high_tenths_c: TenthsC = Field(
        description="Upper stage band limit, inclusive."
    )
    hold_minutes: Annotated[
        int,
        Field(
            ge=1,
            le=360,
            description="Exact number of consecutive measured minutes held.",
        ),
    ]

    @model_validator(mode="after")
    def _check_band(self) -> "ReviewStage":
        if self.band_low_tenths_c > self.band_high_tenths_c:
            raise ValueError(
                "band_low_tenths_c must not exceed band_high_tenths_c"
            )
        return self


class ReviewRequest(BaseModel):
    """Ordered holding stages to check against one archived run."""

    stages: Annotated[
        list[ReviewStage],
        Field(
            min_length=1,
            max_length=6,
            description=(
                "1..6 holding stages, in the order they were performed. "
                "Stages do not overlap and every stage takes exactly its "
                "hold_minutes of consecutive measured minutes."
            ),
        ),
    ]
    max_gap_minutes: Annotated[
        int,
        Field(
            ge=0,
            le=10_000_000_000,
            description=(
                "Largest number of minutes allowed between the end of one "
                "stage and the start of the next. Missing minutes inside a "
                "gap still count against this allowance."
            ),
        ),
    ]


class ReviewStageWindow(BaseModel):
    stage_index: int = Field(description="1-based position in the recipe.")
    start_minute_utc: int
    end_minute_utc: int = Field(description="Inclusive last minute of the stage.")
    hold_minutes: int
    band_low_tenths_c: int
    band_high_tenths_c: int


class ReviewResponse(BaseModel):
    run_id: str
    verdict: Literal["confirmed", "insufficient_evidence", "not_met"]
    verdict_text: str = Field(description="确认达标 | 证据不足 | 不达标")
    reason: str
    max_gap_minutes: int
    stages: list[ReviewStage]
    stage_windows: list[ReviewStageWindow]
    failed_stage_index: Optional[int] = Field(
        default=None,
        description=(
            "1-based stage that could not be placed at all; null when the "
            "failure is the gap constraint or when a verdict was reached."
        ),
    )
    tolerance_tenths_c: int
    payload_sha256: str


class ErrorResponse(BaseModel):
    detail: str
    reason_code: str
    retryable: bool = False
    minimum_retained_points: Optional[int] = None
    max_retained_points: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    database: str
    schema_version: str
