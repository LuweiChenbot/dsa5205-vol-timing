# Exploratory model extensions: frozen implementation contract

User request: implement the proposed modelling improvements and update PR #1.
This extends the modelling work, not portfolio construction, backtesting or
statistical significance analysis. The earlier one-day experiment remains intact.
These extensions were proposed after seeing its evaluation results and must be
identified as exploratory; design-only tuning does not erase that exposure.

## Recovery model

- Reuse the existing ten-session, +/-0.75 realised-volatility barrier definition
  and next-close entry. Label EVERY decline-trigger day as -1 continuation,
  0 unresolved, +1 recovery. Never turn unresolved events into continuation.
- Treat each label as available only at close(s+11), including early barrier hits.
  Only completed labels enter fitting. Emit probabilities on decline days only;
  no implicit signal or trade is generated on other days.
- Fit L2 multinomial logistic regression with nested small feature sets:
  price=[dec_dd, dec_mom]; vix_level=price+[vix_log]; term_structure=vix_level+
  [ts_ratio]. Exclude constant decline flags and redundant decline interactions.
- Use an expanding training history beginning with the saved 2014 feature sample.
  Group triggers using the existing causal non-overlapping 10-session event-start
  rule. Within each training window, assign each episode total sample weight 1.
  This limits long episodes' influence but does not make overlapping labels
  independent. Fit the existing PastScaler on completed training rows only.
- C grid=[0.01,0.1,1,10], lbfgs, max_iter=2000, tol=1e-8. Tune by episode-weighted
  three-class log loss on 2017–2018 decisions with outcomes complete by design end;
  exact ties choose the smaller C. Freeze choices before generating evaluation
  forecasts. Also export term_structure at the vix_level model's C.
- Include a training-only episode-weighted class-prior benchmark with one
  pseudocount per class. If fewer than 12 completed episodes or not all three
  classes are present, use this prior for all models and record the fallback.
  With no history the prior is uniform. Convergence warnings are errors.
- Preserve the existing event labels/files. The new dense labels are separate
  modelling outcomes; episode counts and classes are always reported explicitly.

## Ten-day ridge and penalty investigation

- Generalise the existing ridge engine to horizon=10 with the existing compounded
  SPY-minus-BIL target, exactly 756 matured labels, and the same four feature sets
  plus rolling mean and matched-alpha control. Preserve default horizon=1 behavior.
- Use [0.1,1,10,100,1000,10000,100000] for this exploratory ten-day model. Fit and
  select on completed 2017–2018 labels only. Generalise every timing audit field.
- Run the same expanded penalty grid on ONE-DAY DESIGN DATA ONLY as a sensitivity
  investigation. Save its candidate forecasts and selection; do not replace or
  generate new one-day evaluation forecasts from these selected penalties.
- Use only the committed calendar. The current snapshot lacks the final outcome
  session for the last ten-day decision. Record an explicit tail-coverage exclusion
  rather than inventing a trading date, shortening a horizon, or filling a target.

## Outputs and checks

- One offline command runs the extensions. Separate output directories for the
  classifier and ten-day ridge, each with dated long/wide signals, a validated
  loader, design predictions, separate outcomes, configuration and hashes.
- Record selected settings, coverage, coefficients and training history. Probability
  columns must be named, finite, bounded and sum to one; absent/non-decline signals
  are not zero probabilities. Ten-day return forecasts are not daily returns.
- Do not add evaluation performance metrics, trading rules or confidence intervals.
  Member 4 receives predictions and outcomes for their own explicitly aligned work.
- Tests cover horizons/maturity, synthetic barrier outcomes, future shocks,
  training-only episode weights and scaling, fallback and convergence behavior,
  selection cutoffs/ties, valid handoff and unchanged original forecasts. Run a
  complete real-data build, existing checks and independent code reviews.
