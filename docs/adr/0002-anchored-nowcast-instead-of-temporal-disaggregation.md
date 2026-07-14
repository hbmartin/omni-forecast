# Anchored nowcast instead of temporal disaggregation or reconciliation

The minutely product is a nowcast — the current station observation's residual decayed
into the hourly blend — plus a native blend of the few providers that emit minutely
precipitation. We deliberately rejected the statistical routes the research reports
surveyed: Denton/Chow-Lin-style temporal disaggregation (nothing in the product needs
statistically disaggregated per-minute values; per-minute truth's job is aggregating UP to
score coarser forecasts) and MinT/THieF hierarchical reconciliation (our daily fields are
dominated by nonlinear aggregates — max/min temperature, max gust — which are outside
linear reconciliation's scope by construction; the linear ones are derived from the
blended hourly path and are therefore coherent by construction). If precip-sum coherence
ever proves valuable, temporal MinT can be added downstream without unwinding anything.
