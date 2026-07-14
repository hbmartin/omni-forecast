"""The emitted forecast document: typed, versioned, JSON-serializable."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class MinutelyPoint:
    valid_time: str
    minutes_ahead: int
    temp_c: float | None = None
    humidity_pct: float | None = None
    dew_point_c: float | None = None
    wind_speed_ms: float | None = None
    precip_intensity_mmh: float | None = None
    pop: float | None = None


@dataclass(frozen=True, slots=True)
class HourlyPoint:
    valid_time: str
    lead_hours: float
    lead_bucket: str | None = None
    values: dict[str, float | None] = field(default_factory=dict)
    methods: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DailyPoint:
    date_local: str
    lead_days: int
    values: dict[str, float | None] = field(default_factory=dict)
    methods: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Forecast:
    schema_version: int
    issued_at: str
    latitude: float
    longitude: float
    dataset_fingerprint: str
    sources: list[str]
    observation_at: str | None
    minutely: list[MinutelyPoint]
    hourly: list[HourlyPoint]
    daily: list[DailyPoint]

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=_encode)


def _encode(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    msg = f"cannot serialize {type(value).__name__}"
    raise TypeError(msg)
