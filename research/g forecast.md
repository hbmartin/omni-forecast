# Advanced Methodologies for Mixed-Frequency, Unbalanced Forecast Combination

The integration of high-frequency continuous observations with heterogeneous, multi-horizon, and mixed-frequency external forecasts represents a highly complex challenge at the intersection of time series econometrics, online machine learning, and signal processing. The problem space is defined by three primary complexities: temporal heterogeneity, where ground truth is recorded at a per-minute resolution while external forecasts are provided at per-minute, per-hour, or per-day scales; unbalanced panels and ragged edges, where forecast providers issue predictions for varying horizons and may exhibit missing data; and dynamic model uncertainty, where the predictive superiority of any single forecast provider shifts over time and across different lead times.

This report comprehensively analyzes the theoretical frameworks and computational approaches required to produce a robust, blended forecast. The analysis encompasses temporal disaggregation to resolve frequency mismatches, dynamic forecast combination and online learning to optimize provider weights, the handling of "sleeping experts" to address unbalanced horizons, and mixed-frequency state-space modeling for continuous real-time updating.

## Identification of Systemic Ambiguities and Architectural Queries

Before a definitive mathematical pipeline can be established, the architecture of the forecasting system requires the resolution of several structural ambiguities. The formulation of an optimal combination strategy depends heavily on the specific operational parameters of the target output. To ensure the ensuing architectural recommendations perfectly align with the deployment environment, the following design parameters must be clarified.

First, the target resolution of the blended forecast must be explicitly defined. If the objective is to produce a blended per-minute forecast into the future, all low-frequency forecasts must be temporally disaggregated, which introduces estimation noise. Conversely, if the operational requirement only demands an hourly or daily blended forecast, the per-minute ground truth and high-frequency forecasts must be temporally aggregated, which destroys high-frequency variance.

Second, the structure of the multi-day forecasts requires clarification. When a provider issues a forecast for "10 days out," it must be determined whether this represents a single point forecast for the 10th day, a cumulative forecast for the 10-day period, or a sequence of 10 daily forecasts. This distinction dictates whether the system must interpolate the intervening days or simply align the specific horizons.

Third, the nature of the missing forecasts must be classified. The absence of a forecast beyond a provider's maximum horizon represents deterministic missingness, necessitating an online "sleeping experts" framework. However, if providers occasionally fail to deliver forecasts within their stated horizons due to latency or system failures, this represents stochastic missingness, commonly referred to as the "ragged edge" problem, which is best handled via state-space filtering.

Fourth, the evaluation metric and loss function must be specified. Minimizing Mean Squared Error assumes a symmetric, quadratic loss, which favors mean-based combinations. If extreme deviations are highly penalized, or if the forecast requires asymmetric risk management, quantile loss functions or Mean Absolute Error must be applied, necessitating median-based, stochastic dominance, or quantile regression combinations.

Finally, the computational latency budget must be established. Generating a per-minute blended forecast using complex covariance matrix inversions or Bayesian Markov Chain Monte Carlo sampling may exceed real-time processing limits. The latency budget determines whether the system can employ computationally heavy hierarchical reconciliations or must rely on lightweight, recursive online learning updates.

## Temporal Alignment: Resolving Frequency Heterogeneity

To combine per-minute, per-hour, and per-day forecasts, all time series must be mathematically projected into a common dimensional space. Researchers working with mixed-frequency time series typically face a choice between aggregating high-frequency observations to the lowest available frequency or interpolating low-frequency data to the highest available frequency. Because aggregating the per-minute ground truth to a daily level would destroy massive amounts of potentially useful volatility and behavioral information, temporal disaggregation is the preferred approach.

### Classical Econometric Temporal Disaggregation

Temporal disaggregation converts low-frequency time series into high-frequency estimates while strictly preserving an aggregation constraint, meaning the sum, average, first, or last value of the high-frequency values must exactly equal the low-frequency aggregate. Classical methods leverage a high-frequency indicator series to guide the intra-period distribution of the low-frequency aggregate. In the proposed system, the continuous per-minute ground truth serves as the ideal high-frequency indicator to disaggregate the daily and hourly external forecasts.

The Denton and Denton-Cholette methods rely on the principle of movement preservation. They estimate the high-frequency series by minimizing the difference between the target series and the indicator series, subject to the aggregation constraint. The proportional first-differences variant minimizes the squared relative deviations. This approach assumes that short-term fluctuations have a multiplicative effect and focuses on preserving period-to-period changes. The matrix formulation seeks to find an unknown high-frequency series whose sums are consistent with a known low-frequency series, utilizing a distribution matrix that penalizes deviations from the indicator. This method is highly effective when the primary goal is to map the per-minute volatility of the ground truth onto the daily forecasts without assuming a strict structural economic model.

Alternatively, the Chow-Lin, Fernandez, and Litterman methods treat temporal disaggregation as a regression problem. These methods perform a Generalized Least Squares regression of the low-frequency values on the aggregated high-frequency indicator series. The critical assumption is that the linear relationship holding at the low frequency also holds at the high frequency. The methods differ in their assumptions regarding the residual error structure. Chow-Lin assumes an Autoregressive moving average process for the errors, making it suitable for stationary or cointegrated series. Fernandez assumes a random walk error, and Litterman assumes an ARIMA error structure, which are better suited for non-cointegrated, trending series. While computationally elegant, regression-based methods can occasionally produce negative values if the intra-period variance is high. Modern computational implementations utilize post-estimation corrections or non-negative least squares optimization to redistribute these values proportionally, ensuring that the aggregate values still match the original data without violating physical constraints.

### Machine Learning and Heuristic Disaggregation

For sub-hourly disaggregation, classical econometric models often face computational bottlenecks due to the massive size of the covariance matrices required to map daily values to minute-level intervals. Furthermore, linear methods may fail to capture complex, non-linear intra-day dynamics.

The K-Nearest Neighbors Method of Fragments is a non-parametric approach that identifies historical "analogue days" where the total daily volume closely matches the forecasted daily volume. The per-minute fractional distribution of the analogue day is extracted as a vector of weights that sum to one. These weights are then multiplied by the new daily forecast to generate the high-frequency series. This method is highly computationally efficient and guarantees that the generated high-frequency series maintains realistic statistical properties, as it is sampled directly from historical ground truth.

Recent advancements leverage Artificial Neural Networks and Recurrent Neural Networks for temporal downscaling. Long Short-Term Memory networks can be trained to learn the non-linear mappings between daily aggregates and intra-day sequences, capturing complex temporal dependencies that linear models miss. Peak-Over-Threshold frameworks utilizing neural networks have also been developed to specifically ensure that the extreme values generated during the disaggregation process accurately reflect historical extremes, mitigating the tendency of linear models to overly smooth high-frequency estimates.

| **Disaggregation Method** | **Core Mechanism**                               | **Error/Residual Assumption** | **Optimal Use Case**                                         |
| ------------------------- | ------------------------------------------------ | ----------------------------- | ------------------------------------------------------------ |
| Denton-Cholette           | Movement preservation via quadratic minimization | Independent, unmodeled        | Preserving exact proportional volatility of the ground truth. |
| Chow-Lin                  | Generalized Least Squares regression             | AR(1) stationary              | Cointegrated series with strong linear relationships.        |
| Litterman                 | Generalized Least Squares regression             | ARIMA(1,1,0)                  | Trending, non-stationary low-frequency forecasts.            |
| Method of Fragments       | K-Nearest Neighbors historical sampling          | Non-parametric                | Highly non-linear intra-day seasonality.                     |
| LSTM Networks             | Recurrent sequence-to-sequence mapping           | Learned non-linear            | Massive datasets where sub-hourly dependencies span multiple days. |



### Cross-Temporal Hierarchical Reconciliation

An alternative to strict interpolation is modeling the mixed-frequency forecasts as a temporal hierarchy. A time series recorded at per-minute intervals can be aggregated into hours and days, forming a structured hierarchy where the base level represents the highest frequency and the top level represents the lowest frequency.

If external providers issue forecasts at different temporal levels independently, these forecasts will likely be non-coherent, meaning the sum of the per-minute forecasts will not equal the hourly forecast, and the sum of the hourly forecasts will not equal the daily forecast. Hierarchical forecast reconciliation projects these base forecasts onto a coherent subspace, ensuring that the forecasts respect the linear aggregation constraints of the hierarchy.

The Minimum Trace estimator achieves optimal reconciliation by minimizing the trace of the expected covariance matrix of the coherent forecast errors. This approach requires an estimate of the variance-covariance matrix of the base time series, which can be computationally intensive as the size of the matrix grows with the square of the number of observations. To scale this to per-minute data, recent frameworks utilize closed-form estimators of the covariance structure or decompose large hierarchies into sub-hierarchies that can be reconciled separately, enabling parallel computation while maintaining global coherence. By applying temporal reconciliation, the system can natively integrate the 10-day provider's daily forecasts with the 3-day provider's minute-level forecasts, producing a unified predictive distribution across all frequencies.

## The Forecast Combination Puzzle and Static Weighting

Once the data is temporally aligned, the multiple external forecasts must be combined. The combination of forecasts reduces the risk of model misspecification, effectively hedging against the failure of any single provider. Furthermore, no single model can perfectly capture the true data generating process; combination integrates information gleaned from different sources, leading to consistent gains in predictability.

The most enduring empirical finding in the forecasting literature is the "forecast combination puzzle"—the observation that simple, equally-weighted averages of multiple forecasts routinely outperform complex, optimally weighted combinations in out-of-sample testing. This puzzle arises because optimal combination weights depend on the covariance matrix of the forecast errors, which is notoriously difficult to estimate with precision. When the number of forecasts is large relative to the historical sample size, the estimation variance of the optimal weights overwhelms the theoretical bias reduction, leading to poor generalization. The simple average forces all weights to be equal, trading a small amount of bias for a massive reduction in estimation variance, which makes it an incredibly robust baseline.

### Regularization and Shrinkage

When historical data is highly abundant, as is the case with continuous per-minute ground truth, the estimation error diminishes, and sophisticated weighting schemes become viable. The classical Granger-Ramanathan method involves regressing the ground truth against the matrix of predictions to find weights that minimize the in-sample combined forecast error. To guarantee an unbiased combined forecast, the weights are often constrained to sum to one and remain non-negative.

However, because individual forecasts are often highly correlated as they predict the same underlying event, the Ordinary Least Squares covariance matrix is prone to severe multicollinearity. Regularized estimation methods, such as Ridge or Lasso regression, shrink the combination weights to mitigate the variance inflation caused by this multicollinearity. The Egalitarian Lasso specifically shrinks weights toward the equal-weight average, creating a continuum between the optimal OLS weights and the simple average, allowing the system to dynamically trade off bias and variance based on the amount of available historical data.

An alternative, computationally lighter approach assigns weights inversely proportional to each provider's historical out-of-sample error variance or performance metric. The Bayesian shrinkage interpretation of this approach suggests that if there is weak evidence that one method is better than another, they should be weighted equally; if there is strong evidence of superiority, the weight vector should reflect the strength of that evidence.

### Bayesian Model Averaging and Predictive Likelihood

Traditional Bayesian Model Averaging provides a rigorous framework for forecast combination by weighting the forecasts from each model by their posterior model probabilities, which are derived from the marginal likelihood of the models. However, the marginal likelihood is a joint assessment of how well the entire model fits the data. For forecasting applications, a model might fit the historical data well but predict poorly.

To circumvent this, advanced Bayesian forecast combinations base their weights on the predictive likelihood at the relevant forecast horizon. This focuses the evaluation strictly on the predictive performance for the variables of interest. By utilizing a series of short-horizon predictive likelihoods, the system can continuously update the posterior probabilities of each external provider, assigning higher weights to providers whose predictive densities consistently assign high probability to the realized per-minute ground truth. Furthermore, information criteria approaches, such as Akaike weights, can be utilized to evaluate the relative strength of evidence for the best model over the others, providing a computationally efficient alternative to full Bayesian integration.

| **Weighting Strategy** | **Mathematical Basis**           | **Primary Vulnerability**          | **Operational Use Case**                                     |
| ---------------------- | -------------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| Simple Average         | Arithmetic mean                  | Ignores provider skill differences | Default baseline; high noise environments.                   |
| Granger-Ramanathan     | OLS regression of errors         | Multicollinearity; overfitting     | Large sample sizes with low forecast correlation.            |
| Egalitarian Lasso      | L1 penalized regression          | Requires hyperparameter tuning     | High-dimensional panels of highly correlated providers.      |
| Inverse Variance       | Proportional to inverse MSE      | Sensitive to recent outlier errors | Fast, variance-based performance weighting.                  |
| Predictive Likelihood  | Bayesian posterior probabilities | Computationally intensive          | Density forecast combination with rigorous probability theory. |



### Horizon-Specific Weighting and Stochastic Dominance

Forecast accuracy degrades differently for different providers as the lead time increases. A model that excels at a 1-day horizon may rely on high-frequency autoregressive features that decay rapidly, while a model predicting 10 days out may rely on slow-moving structural features. Therefore, combination weights must be strictly horizon-specific. Rather than calculating a single weight for a provider, the system must calculate a vector of weights for each forecasting horizon. This ensures that short-term experts dominate the near-term blended forecast, while long-term experts dominate the distant horizon.

Furthermore, weighting can be optimized across different quantiles of the forecast error distribution using Stochastic Dominance Efficiency. For the optimal forecast combination, Stochastic Dominance Efficiency minimizes the cumulative density functions of the levels of loss at different quantiles of the forecast error distribution by combining different forecasts. This approach provides differential forecast weights across the distribution, ensuring that the blended forecast remains robust even in the extreme tails of the distribution, outperforming standard quadratic loss optimization in volatile environments.

## Dynamic Combination and Online Learning in Non-Stationary Environments

The performance of external forecast providers is rarely stationary. Providers routinely update their internal algorithms, and structural shifts in the underlying data generating process cause certain models to fail while others succeed. For instance, models incorporating macroeconomic factors may perform best during recessions, while parsimonious models without such factors excel during low-volatility expansions. Consequently, backward-looking static weights frequently fail to adapt to regime changes.

To address non-stationarity, the system must utilize dynamic, time-varying weights. This is formalized through the mathematical framework of Prediction with Expert Advice, drawn from game theory and online machine learning. In this framework, the external providers are treated as "experts," and the system updates its trust in each expert sequentially as new per-minute ground truth data arrives, operating without making any stochastic assumptions about the data generating process.

### Online Prediction by Expert Aggregation

At each time step, the algorithm incurs a loss based on the discrepancy between the combined forecast and the ground truth, and updates the weights for the next time step. The Exponentially Weighted Average forecaster updates the weight of an expert based on its cumulative loss. The weight assigned to an expert is proportional to the exponential of its negative cumulative loss multiplied by a learning rate. This guarantees that the blended forecast will perform asymptotically as well as the single best expert in hindsight, yielding a provable upper bound on the "regret".

While Exponentially Weighted Average is powerful, it assumes bounded, homoskedastic losses. Given that per-minute time series data often exhibits severe heteroskedasticity and volatility clustering, Bernstein Online Aggregation introduces a second-order refinement. It utilizes both the cumulative loss and the squared variations of the loss to adjust the learning rate dynamically for each expert. By incorporating the variance of the prediction errors, Bernstein Online Aggregation adapts significantly faster during turbulent market or environmental regimes, penalizing experts that exhibit highly volatile errors even if their average error remains acceptable.

## Managing Ragged Edges: Sleeping Experts and Incomplete Panels

A critical constraint outlined in the system architecture is that different providers forecast for different maximum lengths of time. Some provide forecasts 3 days out, others 7 days, and others 10 days. Furthermore, forecast data is often delivered as an unbalanced panel due to the frequent entry, exit, or temporary latency of individual providers.

Standard online learning algorithms and ordinary least squares regressions require all experts to submit predictions at every time step. If the panel is trimmed to only include the periods where all providers are active, massive amounts of data are discarded. Conversely, if a 3-day expert ceases to provide forecasts for days 4 through 10, standard aggregation algorithms fail because the loss function for the missing expert cannot be evaluated.

### The Sleeping Experts Framework

This scenario maps directly onto the Sleeping Experts, or Specialized Experts, problem in online learning. In this framework, the environment dictates an "awake set" of available experts at each step.

To aggregate sleeping experts, the algorithm tracks the weights only over the subsets of time where the experts are active. The "Follow-the-Awake-Leader" algorithm and its randomized variants dynamically renormalize the weights exclusively among the awake set at each specific forecasting horizon. The mathematical reduction from the sleeping expert problem to the regular expert problem involves ignoring the weights of asleep experts and renormalizing the probabilities of the awake experts. To maintain the integrity of the underlying regret bounds, a synthetic loss is assigned to the asleep experts—typically the weighted average loss of the awake experts for that round—ensuring that an expert is neither penalized nor rewarded for being asleep.

The definition of regret must also be modified. Rather than comparing the algorithm to the best single expert over the entire period, which is impossible if no expert is awake the whole time, regret is measured against the best ordering of experts or the best convex combination of available experts in hindsight. This guarantees that the 10-day blended forecast seamlessly transitions its reliance from the dense cluster of short-term experts during the first 3 days to the sparser cluster of long-term experts for the remaining 7 days, without mathematical discontinuity or structural penalties.

## Unified Mixed-Frequency State-Space Modeling

While online expert aggregation provides robust, model-agnostic weighting, it treats the individual forecasts as black boxes. To construct a truly unified mathematical environment that natively handles both the mixed frequencies and the stochastic missing data of the ragged edges, the system can be cast as a Mixed-Frequency State-Space model solved via the Kalman Filter.

### The State-Space Formulation and the Kalman Filter

In the state-space paradigm, the true underlying value of the forecasted variable is treated as an unobserved, high-frequency continuous latent state vector. Both the per-minute ground truth observations and the external low-frequency forecasts are treated as noisy measurements of this latent state.

The system is defined by two equations. The transition equation describes how the high-frequency latent state evolves over time, incorporating trend changes, seasonal adjustments, and autoregressive dynamics. The measurement equation links the latent state to the available observations, which includes both the ground truth and the various providers' forecasts.

The Kalman Filter provides an elegant, recursive solution to the missing data problem inherent in unbalanced panels and varying horizons. The measurement matrix serves as an aggregation and selection matrix. When a daily forecast is available, this matrix maps the daily measurement to the sum of the corresponding intra-day high-frequency states.

Crucially, if a provider drops out, their maximum horizon is exceeded, or their API fails, the observation for that specific provider is registered as a missing value. The Kalman Filter naturally handles this by executing the prediction step, which propagates the latent state and its covariance matrix forward in time, but skips the update step for that specific missing measurement. The filter relies entirely on the remaining active providers and the system's internal dynamics to bridge the gap. This methodology allows the system to continuously ingest the per-minute ground truth to dynamically adjust the intercept and trajectory of the multi-day forecasts in real time, executing a mathematically optimal, continuous nowcast that blends all available information regardless of frequency or horizon.

## Probabilistic Ensembling and Uncertainty Quantification

Providing a single blended point forecast is rarely sufficient for high-stakes decision-making; the system must also output prediction intervals that quantify the uncertainty of the blend. The degree of uncertainty affects the weight a decision-maker places on the most likely outcome versus the risk of alternative scenarios.

### Density Combination and Optimal Transport

If the external providers supply probabilistic density forecasts, combining them requires probability pooling. Linear pooling, which involves averaging the densities, often results in overly dispersed or multi-modal distributions if the external providers strongly disagree. This can lead to uncalibrated prediction intervals that do not accurately reflect the true uncertainty. Advanced methods, such as those utilizing Wasserstein Barycenters, use optimal transport theory to merge the densities by advecting probability mass. This preserves the geometric shape of the underlying forecast distributions rather than just flattening them, offering a more sophisticated integration of diverse predictive distributions.

### Conformal Prediction for Time Series

If the external providers only output point forecasts, the blending system must independently generate valid prediction intervals. Traditional interval generation relies on strict distributional assumptions, such as Gaussian errors, or the assumption of exchangeability, meaning the data points are independent and order does not matter. Both assumptions are heavily violated in auto-correlated time series, leading standard uncertainty methods to fail and produce "bursty" error rates where coverage guarantees collapse during volatile periods.

Conformal Prediction provides distribution-free uncertainty bands with rigorous mathematical guarantees on marginal coverage, regardless of the underlying data distribution. For time series, algorithms like Ensemble Batch Prediction Intervals bypass the exchangeability requirement.

This method applies a block-bootstrap to the historical data, fitting an ensemble of combination models to mimic the variability of retraining without massive computational costs. It calculates leave-one-out non-conformity scores, which are the absolute residual errors between the ground truth and the out-of-bag ensemble predictions. By aggregating these sequentially updated residuals, the algorithm produces dynamic prediction intervals that widen during periods of high volatility and narrow during stable regimes. This allows the intervals to "breathe" with the data, guaranteeing that exactly a pre-determined percentage of future observations will fall within the bands, providing a highly reliable measure of uncertainty for the blended forecast.

## Computational Architecture and Implementation Ecosystem

To operationalize this theoretical framework, the processing pipeline must be structured using highly optimized computational libraries designed for large-scale time series manipulation. The landscape of open-source frameworks provides robust tools for each stage of the pipeline.

| **Pipeline Stage**             | **Algorithmic Objective**                                    | **Primary Implementation Frameworks**                        |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Temporal Alignment**         | Disaggregate daily/hourly forecasts into per-minute signals, or aggregate signals into temporal hierarchies. | `tempdisagg` (Python/R) for Chow-Lin/Denton optimization, offering non-negative least squares adjustments. `HierarchicalForecast` (Nixtla) for MinT cross-temporal reconciliation. |
| **Static Base Ensembling**     | Establish a robust baseline combination using regularized regression or simple averaging. | `sktime` (EnsembleForecaster, Multiplexer) for pipeline integration, transformation, and automated cross-validation. |
| **Online Dynamic Weighting**   | Adjust weights sequentially per minute, handling sleeping experts across varying horizons. | `opera` (Online Prediction by Expert Aggregation) for executing Bernstein Online Aggregation, Exponentially Weighted Averages, and MLpol. |
| **Uncertainty Quantification** | Wrap the blended point forecast in distribution-free prediction intervals based on sequential residuals. | Bespoke `EnbPI` implementations combined with `StatsForecast` probabilistic evaluation tools. |



## Conclusion

Creating a coherent, high-fidelity blended forecast from multi-frequency, multi-horizon, and unbalanced inputs requires a deliberate departure from static econometric models. The optimal methodology dictates a multi-stage architecture that must be tuned to the specific latency and output requirements of the deployment environment.

First, the temporal mismatch must be resolved either through high-frequency disaggregation utilizing the Denton-Cholette or Chow-Lin methods, or via structural cross-temporal hierarchical reconciliation utilizing the Minimum Trace estimator. Second, the differing lengths of forecast availability must be handled inherently by the weighting algorithm. Because forecast accuracy degrades non-linearly across horizons and expert performance shifts across regimes, static combinations are insufficient. The system must implement an online learning framework, specifically utilizing algorithms capable of handling "sleeping experts" to dynamically route the ensemble weights to the best available providers at any given minute.

Alternatively, casting the entire system into a Mixed-Frequency State-Space model allows the Kalman filter to natively ingest the continuous per-minute ground truth, naturally stepping over the ragged edges of missing multi-day forecasts, and yielding a mathematically optimal, dynamically updated predictive state. Finally, the application of time-series Conformal Prediction ensures that the blended output is accompanied by rigorous, distribution-free uncertainty intervals, establishing a highly resilient and adaptable forecasting engine.