# Hourly truth semantics are calibrated empirically per provider

Providers do not document whether an hourly value means "instantaneous state at the
timestamp" or "mean over the hour", and a large share of apparent provider bias in
systems like this is actually bucket misalignment. Instead of hardcoding one convention,
the dataset layer materializes BOTH hourly truths for state variables — instantaneous
(±5-minute centered mean, ≥1 sample within ±10 min) and interval mean over [H, H+1)
(≥80% minute coverage) — and an alignment study selects, per provider × variable, the
semantics that correlates best with that provider's forecasts. Mixed-convention defaults
(instantaneous for state variables; interval sum/max for precip/gusts, which are
unambiguous) apply until enough data accumulates. The cost is double truth columns and an
extra study artifact; the benefit is that grounding corrects real bias rather than our
own alignment error.
