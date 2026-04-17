# Roadmap — Predicting Graduate Loan Distress

Step-by-step guide for Research Question 1: *What institutional features
predict graduate loan distress?*

Stack: **R + tidymodels** (`recipes`, `parsnip`, `workflows`, `tune`,
`glmnet`, `xgboost`, `lme4`, `shapviz`).

---

## Phase 0 — Nail down the outcome

Before any modeling, the group has to decide on **one primary outcome**.
The candidates from `panel.csv`:

| Variable | What it measures | Coverage | Best for |
|---|---|---|---|
| `CDR3` | 3-yr cohort default rate | 90%+ 2011–2023 | Headline outcome. Most direct "can't repay" signal. |
| `RPY_3YR_RT` | 3-yr repayment rate | 84%, only 2009–2016 | Secondary / robustness check. |
| `GRAD_DEBT_MDN` | Median completer debt | ~80% 1997–2020 | "In debt" but not "in distress" — borrow amount ≠ inability to pay. |
| `GRAD_DEBT_MDN / MD_EARN_WNE_P10` | Debt-to-earnings ratio | Limited by earnings years (7 cohorts) | Economically meaningful but shrinks the dataset. |

**Recommendation:** primary outcome = `CDR3`. Secondary (robustness) = `RPY_3YR_RT` on its overlap window.

**Decision to make:** model `CDR3` as
- (a) continuous regression (preserves info), or
- (b) binary classification (e.g. `CDR3 > 0.10` = "distressed"; more
  interpretable, easier SHAP visuals)?

Pick one as primary; optionally run the other as robustness.

---

## Phase 1 — Data preparation

**Script:** `notebooks/distressed_analysis/distress_clean.R`.

1. Load `data/processed/panel.csv`.
2. Filter to rows where `CDR3` is non-null — this also applies the
   year window (2011–2023) since that's where `CDR3` is populated.
3. **Backfill time-invariant flags** (`HBCU`, `HSI`, `LOCALE`) within
   each `UNITID` using `tidyr::fill(.direction = "updown")`. `HBCU` is
   genuinely time-invariant; `HSI` and `LOCALE` change rarely and
   "updown" preserves real variation where it exists. Caveat: `HSI`
   is an enrollment-threshold designation that can flip year to year,
   so this is a simplification worth flagging in the writeup.
4. **Drop outcome-side columns explicitly** — these shouldn't be
   predictors of contemporaneous distress:
   - `MD_EARN_WNE_P10` / `MD_EARN_WNE_P6` (sparse + circular).
   - `RPY_3YR_RT` (reserved as robustness outcome).
   - `GRAD_DEBT_MDN10YR` (redundant with `GRAD_DEBT_MDN`).
5. **Coverage-based drop with a protected list.** Run the ≥50%
   coverage check *after* the year filter, but protect:
   - `SAT_AVG` / `ADM_RATE` — selectivity signal; NA is informative
     (open-enrollment / non-test-requiring schools). XGBoost handles
     NA natively; for linear models add a missingness indicator.
   - `FIRST_GEN` / `MD_FAMINC` — populated only 2011–2016 within the
     CDR3 window; handled by the two-dataset split below.
6. Feature engineering:
   - Convert `CONTROL`, `PREDDEG`, `REGION`, `LOCALE` to factors with
     labeled levels.
   - Log-transform `UGDS`, `AVGFACSAL`, `INEXPFTE`, `TUITFTE` via
     `log1p` (right-skewed by construction; log1p handles zeros).
     Keep raw columns alongside logged versions.
   - Leave the 38 `PCIP*` columns as-is (already proportions in [0,1]).
7. **Split into two analysis datasets:**
   - `distress_primary.csv` — full CDR3 window (2011–2023),
     `FIRST_GEN` / `MD_FAMINC` dropped. This is the main modeling
     dataset.
   - `distress_firstgen.csv` — 2011–2016 overlap where `FIRST_GEN`
     and `MD_FAMINC` are populated. Shorter panel, but lets the model
     use family-income and first-generation signals as a secondary
     comparison.

**Deliverable:** both CSVs in `data/processed/`, a message log of
which columns were dropped by the coverage check, and a one-paragraph
note in the EDA Rmd explaining why each protected / dropped column
was treated the way it was.

---

## Phase 2 — Outcome-focused EDA

Goal: understand the target before fitting models.

1. **Distribution of `CDR3`** — histogram, by year. Does the distribution
   shift over time? (If yes, year must be a predictor or the data must
   be split by year.)
2. **Within-school vs between-school variance in `CDR3`.** Critical —
   if between-school variance dominates (as it does for earnings), the
   random-effects model in Phase 6 won't add much beyond a fixed school
   effect.
3. **Univariate plots** of `CDR3` against the top candidate predictors:
   `CONTROL`, `PREDDEG`, `PCTPELL`, `ADM_RATE`, `C150`, `GRAD_DEBT_MDN`,
   `NPT4`, `FIRST_GEN`.
4. **Bivariate correlation heatmap** of the continuous features —
   flag any pairs with |r| > 0.8 for the penalized-regression stage.

**Deliverable:** `notebooks/EDA_distress.Rmd` with 4–6 plots and a short
narrative.

---

## Phase 3 — Resampling strategy (don't get this wrong)

The panel has repeated observations per school, so naive random CV leaks
information between folds. Pick one of:

| Strategy | When to use | Implementation |
|---|---|---|
| **Single-cohort cross-section** | Simplest. Model is "what predicts distress *in year Y*." | Filter to one year (e.g. 2018), then standard k-fold. |
| **Grouped k-fold by UNITID** | Use the full panel but keep schools in one fold only. | `rsample::group_vfold_cv(group = UNITID)` |
| **Time-based split** | Tests generalization to future cohorts. | Train on 2011–2018, validate 2019, test 2020–2023. |

**Recommendation:** start with **grouped k-fold by UNITID** on the full
2011–2023 panel. Run the time-based split as a robustness check.

---

## Phase 4 — Baseline models

Fit these first so you have something to beat.

1. **Null model** — predict mean `CDR3` (or majority class). This is your
   floor; any real model must clear it.
2. **Linear/logistic regression** on a hand-picked set of 8–10 features
   (`CONTROL`, `PREDDEG`, `PCTPELL`, `ADM_RATE`, `C150`, `NPT4`,
   `UGDS`, `FIRST_GEN`, `MD_FAMINC`). Gives you a readable coefficient
   table for the writeup.
3. **Penalized regression (LASSO + ElasticNet)** via `glmnet` through
   `tidymodels`. Tune `penalty` (and `mixture` for ElasticNet) with the
   grouped-fold CV from Phase 3. LASSO does feature selection — note
   which predictors survive.

**Scaffold:**

```r
library(tidymodels)

rec <- recipe(CDR3 ~ ., data = train) |>
  update_role(UNITID, INSTNM, year, new_role = "id") |>
  step_impute_median(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

lasso_spec <- linear_reg(penalty = tune(), mixture = 1) |>
  set_engine("glmnet")

wf <- workflow() |> add_recipe(rec) |> add_model(lasso_spec)

folds <- group_vfold_cv(train, group = UNITID, v = 5)
tuned <- tune_grid(wf, resamples = folds, grid = 30)
```

**Deliverable:** metrics table (RMSE / R² for regression, or ROC-AUC /
log-loss for classification) across null → OLS → LASSO → ElasticNet.

---

## Phase 5 — Flexible models (gradient boosting)

Where the nonlinear interactions and feature importance come from.

1. **XGBoost** via `parsnip::boost_tree(engine = "xgboost")`.
2. Tune `trees`, `tree_depth`, `learn_rate`, `min_n`, `loss_reduction`,
   `sample_size`, `mtry`. Use Bayesian optimization via `tune_bayes()`
   to keep runtime reasonable.
3. No normalization, no dummies needed — XGBoost handles factors and NAs
   natively. **Simpler recipe** for this model: just cast to matrix.
4. Compare on the same folds as Phase 4. The gap between LASSO and
   XGBoost tells you how much nonlinearity matters.

**Deliverable:** tuned XGBoost model, CV metrics, comparison plot
vs. baselines.

---

## Phase 6 — Hierarchical model (the methodological distinctive)

This is the piece that makes the project statistically interesting
beyond a vanilla ML benchmark.

1. Fit **random-intercept model**:
   `lmer(CDR3 ~ CONTROL + PREDDEG + PCTPELL + ADM_RATE + C150 + NPT4 + (1 | UNITID), data = panel)`
2. Report **ICC** — what fraction of `CDR3` variance is *between-school*
   vs *within-school*? This directly answers the variance-decomposition
   side of the question.
3. If there's meaningful within-school variance, fit a
   **random-slopes model** with `(year | UNITID)` to ask "are some
   schools' default rates trending differently than others?"
4. For the binary classification version: use `glmer` with
   `family = binomial`.

**Deliverable:** ICC number, random-effects summary, story about
within-vs-between.

---

## Phase 7 — Interpretation

Make the XGBoost model readable.

1. **Global feature importance** via SHAP mean-|value|:
   ```r
   library(shapviz)
   sv <- shapviz(xgb_fit, X_pred = as.matrix(X_test))
   sv_importance(sv)
   sv_dependence(sv, v = "PCTPELL")  # marginal effect plots
   ```
2. **SHAP dependence plots** for the top 5 features — show nonlinearity
   and interactions.
3. **Partial dependence** plots (`DALEX` or `pdp`) as a complementary
   view.
4. Compare LASSO-selected features vs XGBoost top-importance features.
   Agreement = robust signal; disagreement = something interesting to
   discuss.

**Deliverable:** 2 SHAP plots (summary + top-feature dependence), a
table cross-referencing LASSO coefficients with XGBoost importance
ranks.

---

## Phase 8 — Robustness & subgroups

Show the result isn't an artifact of one modeling choice.

1. **Alternative outcomes:** rerun the pipeline with `RPY_3YR_RT`
   (2009–2016 overlap). Do the top predictors agree?
2. **Subgroup models:** fit separate models by `CONTROL` (public /
   private non-profit / for-profit). The for-profit sector often has
   qualitatively different distress drivers — worth showing.
3. **Temporal stability:** does a model trained on 2011–2016 still
   work on 2020–2023? Report the degradation.
4. **Coefficient signs:** any predictors whose direction contradicts
   the literature? Explain or flag.

**Deliverable:** short robustness appendix in the final report.

---

## Phase 9 — Writeup

Structure for the final report (maps onto the phases above):

1. Research question & motivation.
2. Data description (persistent panel, 3,790 schools × 13 cohorts
   after filtering).
3. Outcome definition (`CDR3`) and why.
4. EDA highlights (Phase 2).
5. Methods — resampling strategy, model ladder
   (null → OLS → LASSO → XGBoost → mixed model).
6. Results — metrics table, feature importance, SHAP story.
7. Variance decomposition from the hierarchical model.
8. Robustness checks.
9. Limitations (institution-level medians, Title IV only, selection
   bias).
10. Conclusions & what the findings imply for students / policy.

---

## Suggested file layout

```
src/
  build_distress_dataset.R      # Phase 1
  modeling_helpers.R             # shared recipe / metric functions
notebooks/
  EDA_distress.Rmd               # Phase 2
  modeling_distress.Rmd          # Phases 4-7
  robustness_distress.Rmd        # Phase 8
reports/
  roadmap_loan_distress.md       # this file
  research_questions.md
  final_report.Rmd               # Phase 9
data/processed/
  distress.csv                   # Phase 1 output
```

---

## Minimum-viable path (if time is tight)

If the group is crunched, the shortest defensible path is:
Phase 1 → 3 (grouped CV) → 4 (LASSO only) → 5 (XGBoost) → 6 (ICC only,
skip random slopes) → 7 (SHAP summary plot only) → 9.

Skip Phase 8 entirely; note the limitation.
