# Related Work And Contribution Positioning

Phase 19A positions Regime Alpha Benchmark against the literature before adding
more encoder variants. The goal is to prevent the project from sounding like a
generic trading bot or an unsupported novelty claim.

The defensible contribution is:

> Regime Alpha Benchmark benchmarks classical sequential regime models against
> contrastive time-series representation methods, then tests whether HMM-guided
> weak supervision improves learned regime structure and downstream alpha
> robustness under purged financial validation.

This is not a claim that the project invents market regimes, invents HMMs, or
proves a profitable trading strategy. It is a controlled empirical benchmark
around regime discovery, regime-conditioned alpha modeling, validation hygiene,
and the limits of deep representation learning on noisy financial time series.
HMM states are not ground truth; they are classical proxy/reference states used
for comparison and weak supervision.

## Cluster 1: Contrastive Time-Series Representation Learning

Modern time-series contrastive learning tries to learn useful representations
without manual labels. This project builds on that idea, but evaluates it in a
financial regime-detection setting rather than in generic classification or
forecasting benchmarks.

Important anchors:

- TS2Vec learns universal time-series representations through hierarchical
  contrastive objectives over augmented contexts.
- TNC defines positive neighborhoods through local temporal smoothness and
  stationarity assumptions.
- CoST separates seasonal and trend representations and uses both time-domain
  and frequency-domain contrastive signals.
- TS-TCC and TF-C reinforce the broader lesson that time-series representation
  quality depends heavily on how temporal and frequency views are constructed.

Relevance to Regime Alpha Benchmark:

- The initial encoder used adjacent-window contrastive learning, which can
  reward local smoothness even when a market is crossing a regime boundary.
- Phase 18 changes the objective by using HMM states as weak labels for
  positives and boundary-aware hard negatives.
- Future ablations should compare vanilla NT-Xent, HMM-guided contrastive
  learning, and time-frequency views rather than simply increasing model size.

Research gap:

Most time-series contrastive papers evaluate generic downstream tasks. This
project asks whether those learned representations help financial regime-aware
alpha modeling under purged walk-forward validation and transaction costs.

## Cluster 2: Classical Regime-Switching Models In Finance

Regime-switching models are a long-standing financial econometrics tool.
Hamilton's Markov-switching model made latent economic states a standard way to
model non-stationary time series. Gaussian HMMs and Markov-switching GARCH
models extend that logic to hidden states, volatility, and transition dynamics.

Important anchors:

- Hamilton-style Markov switching provides the classical sequential-state
  framing.
- Gaussian HMMs give a tractable hidden-state baseline with transition
  probabilities and posterior state uncertainty.
- Markov-switching GARCH models show why volatility regimes are natural in
  financial data, especially when calm and turbulent periods alternate.

Relevance to Regime Alpha Benchmark:

- The raw-feature HMM is not a strawman baseline. It is the classical method
  the learned encoder must beat.
- The current strongest result is that sequential consistency matters: HMM and
  contrastive-HMM improve on vanilla contrastive assignment.
- HMM states are treated as reference/proxy states, not ground-truth regimes.

Research gap:

Classical regime models are interpretable and sequentially disciplined, but
they rely on selected input features. Deep encoders can learn richer
representations, but may create smooth partitions that are not alpha-relevant.
Regime Alpha Benchmark tests this tradeoff directly.

## Cluster 3: Financial Machine Learning Validation

Financial labels overlap in time, samples are serially dependent, and naive
cross-validation can leak future information. The methodology therefore follows
the financial ML discipline associated with triple-barrier labels, purging,
embargoes, and careful backtest interpretation.

Important anchors:

- Triple-barrier labeling creates directional, neutral, and stop/profit
  outcomes instead of relying only on fixed-horizon return signs.
- Purging and embargoing reduce leakage when labels depend on future bars.
- Statistical and multiple-testing checks are needed because many tested
  strategy variants can create false discoveries.

Relevance to Regime Alpha Benchmark:

- The primary target is `tb_label_8h`.
- Walk-forward validation uses a 5-day embargo and label-horizon purge checks.
- Phase 12 audits leakage and coverage.
- Phase 15 adds fold-level confidence intervals, paired tests, DM-style
  forecast-loss checks, multiple-testing correction, and Probabilistic Sharpe
  diagnostics.

Research gap:

Many regime-learning demos stop at visual clusters or a single backtest. This
project evaluates regime methods through a financial ML validation stack and
reports weak or mixed results honestly.

## Cluster 4: Regime-Conditioned Alpha Models

Regime-conditioned alpha modeling asks whether a predictor should behave
differently in different market states. The practical idea is old: momentum,
mean reversion, volatility, and liquidity signals can behave differently across
calm, trending, and stressed periods.

Relevance to Regime Alpha Benchmark:

- The benchmark compares a global LightGBM classifier against one model per
  regime.
- Alpha predictions are evaluated under the same target, same folds, same
  transaction-cost assumptions, and the same common test universe.
- The project separates three questions that are often mixed together:
  regime quality, alpha prediction quality, and trading-risk quality.

Research gap:

The project does not only ask whether a regime method makes a better-looking
chart. It asks whether regime conditioning improves IC, Sharpe, drawdown,
turnover, calibration, and robustness under fair comparison.

## Contribution Statement For The Paper

The paper-style contribution should be framed as:

1. A reproducible benchmark comparing learned and classical market-regime
   methods under purged walk-forward financial validation.
2. A diagnostic result showing that vanilla contrastive regimes are smooth but
   weakly aligned with the classical HMM reference and often weaker downstream.
3. An HMM-guided contrastive objective that uses sequential states as weak
   supervision, improving structural alignment in the Phase 18 smoke run.
4. A disciplined evaluation layer: validation audit, fold-local regime refit,
   robustness matrices, stress tests, confidence intervals, and multiple-testing
   controls.

## What Not To Claim

- Do not claim HMM states are true market regimes.
- Do not claim the strategy is profitable.
- Do not claim the HMM-guided encoder beats HMM on alpha until Phase 20/21
  downstream tests are complete.
- Do not claim broad asset-class generalization until a multi-asset extension
  is run and passes the written gate.
- Do not describe the current encoder as a full Temporal Fusion Transformer.

## Current Paper Hypotheses

| ID | Hypothesis | Current Status |
|---|---|---|
| H1 | Sequential regime consistency matters for downstream alpha modeling. | Supported directionally by HMM and contrastive-HMM results. |
| H2 | Vanilla contrastive embeddings create smooth but not necessarily alpha-relevant regimes. | Supported by Phase 16 regime-quality diagnostics. |
| H3 | HMM-guided contrastive learning can improve learned-regime structure. | Strongly supported structurally by the 30-epoch Phase 19B run. |
| H4 | Improved regime structure improves fold-local alpha performance. | Directionally supported by Phase 20 and strongly stress-supported by Phase 21; fold-level IC significance remains inconclusive. |
| H5 | The structural finding generalizes beyond BTC and ETH. | Supported structurally on Crypto-20; predictive dominance remains unsupported. |

## Reviewer Risk Register

| Risk | Why It Matters | Mitigation |
|---|---|---|
| HMM is not ground truth | Agreement metrics could be misread as accuracy. | Describe HMM as a classical proxy/reference only. |
| Single-asset-class crypto scope | Reviewers may question transfer beyond correlated crypto assets. | Report Crypto-20 as within-crypto evidence and require a registered external holdout for broader claims. |
| Encoder trained offline | Representation learning may leak descriptive structure into predictive claims. | Treat fully fold-local encoder training as a critical validity requirement before the final predictive claim. |
| Many tested variants | Multiple testing can inflate claims. | Use Phase 15 correction outputs for paper language. |
| Weak trading performance | Negative Sharpe can look like failure. | Frame the project as a benchmark about regime structure and validation, not a deployed strategy. |

## Reading Map

The compact source matrix lives in `reports/literature_matrix.csv`. It is the
paper-planning artifact. This note is the narrative version used for the README
and research report.
