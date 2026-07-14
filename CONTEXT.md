# omni-forecast

Blends multiple providers' weather forecasts into station-grounded minutely/hourly/daily
forecasts for a single personal weather station, with a backtest leaderboard as the sole
arbiter of which method ships.

## Language

### Time

**Issue time**:
The moment a forecast snapshot is taken (from the archive's `fetched_at`/`completed_at`).
_Avoid_: run time, fetch time (as distinct concepts)

**Valid time**:
The moment or interval a forecast is about.
_Avoid_: target time, forecast time

**Lead**:
Valid time minus issue time. Always recomputed from timestamps, never read from a stored column.
_Avoid_: horizon (in code; the upstream column `horizon_hours` is untrusted)

**Lead bucket**:
One of the fixed left-closed lead intervals used to stratify fitting and evaluation
(hourly: 0–1, 1–3, 3–6, 6–12, 12–24, 24–48, 48–96, 96–168, 168–240, 240+ h; daily: D1, D2, D3–4, D5–7, D8–10).

**Snapshot**:
The as-of view of all sources at one issue time: for each source, its latest forecast
fetched at or before that moment and not older than the staleness cap.
_Avoid_: vintage

### Data

**Observation**:
One raw station sample row (~1/minute), imperial units, unvalidated.

**Truth**:
The QC'd, metric-normalized, time-aggregated observation series used for training and
scoring. Truth is either present and trusted or null — never imputed or corrected.

**Truth semantics**:
Which aggregation defines hourly truth for a state variable: instantaneous (±5-min
centered mean at the valid hour) or interval mean over the hour. Chosen per
provider×variable by the alignment study.

**Provider**:
An upstream forecast API, identified by its slug (e.g. `open_meteo`).

**Source**:
One provider+model forecast stream; the unit a blender weighs. Slug: provider, or
`provider_model` when a provider exposes multiple models.
_Avoid_: expert (except in the online-experts method), member

**Source kind**:
Whether supervised rows came from the live archive (`live`) or the Previous Runs
backfill (`synthetic`). Never pooled silently.

**Availability mask**:
The per-row boolean pattern of which sources have a usable forecast; every blender must
renormalize over available sources.

### Methods

**Grounding**:
Per-source correction toward the station, fitted per variable × lead bucket. A
bias (intercept) correction by default; the slope is opt-in (see ADR 0004).
_Avoid_: calibration (reserved for probabilistic calibration), MOS (in code)

**Blending**:
Combining grounded sources into one forecast via weights or a learned stacker.
_Avoid_: ensembling, averaging (as method names)

**Anchoring**:
Short-lead correction of a blend toward the latest observed residual, decaying
exponentially with lead.
_Avoid_: nowcasting (the anchored minutely product is a nowcast; the technique is anchoring)

**Blender**:
Any method implementing the shared protocol `fit(train) / predict(matrix) -> point (+quantiles)`,
baselines included.
_Avoid_: model (overloaded with provider models)

**Product**:
An emitted forecast bundle: minutely (next 60 min), hourly (next 48 h), or daily (next 10 days).

**Self-verification**:
Scoring the system's own emitted products against later truth, alongside providers and
backtest expectations.
