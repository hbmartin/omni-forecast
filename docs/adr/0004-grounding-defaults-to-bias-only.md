# Grounding defaults to a bias-only correction, not a free-slope affine fit

Grounding corrects each source toward the station as `y ~ a + b*x`, but the
slope is shrunk to `b = 1` by default, making it a pure bias correction. This
looks like timidity and is not: an unconstrained least-squares slope reliably
lands *below* 1 (regression dilution — a noisy predictor is shrunk toward the
mean of the training sample). Inside the training distribution that lowers MSE,
but it makes the correction a function of the training-period mean. When the
evaluation period is seasonally different, "shrink toward the training mean"
injects a mean-dependent tilt and re-introduces precisely the bias grounding
exists to remove.

We measured this on 13 months of Open-Meteo Previous Runs backfill against the
Crestline station. With truth available only in summer 2025 and spring 2026,
the fitted slopes came out at 0.76–0.89 and the free-slope correction carried a
**+1.2 to +1.4 °C warm bias** into every test fold, losing to an *uncorrected*
equal-weight blend at 48–168 h leads. Switching the default to bias-only
removed the bias (−0.4 to +0.7 °C) and beat the free-slope variant in every
lead bucket.

The slope is therefore an opt-in (`slope_shrinkage`, 0 = bias-only, 1 = free),
and both variants stay registered (`grounded_equal_weight`, `affine_equal_weight`)
so the leaderboard — not this document — decides when a longer, seasonally
representative archive has earned the slope back.
