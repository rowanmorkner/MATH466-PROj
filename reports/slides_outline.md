# Slide Outline — Predicting Graduate Loan Distress

Target: ~12–14 slides. Each entry below = one slide.

---

## 1. Title

- Title: *What institutional features predict graduate loan distress?*
- Group members, MATH 456, date.
- One-line subtitle: "A panel analysis of 3,790 U.S. colleges, 2011–2019."

---

## 2. Why this matters

Bullet points only — talk through them.

- ~43M Americans hold federal student debt (~$1.6T total).
- Default has real consequences: wage garnishment, credit damage,
  loss of federal aid eligibility.
- **Default rates vary enormously across schools** — some sectors
  default at 5×+ the rate of others.
- Question: *is that variation predictable from publicly reported
  institutional characteristics?* If yes, it's a structural property
  of schools — useful for students, policymakers, accreditors.

No figure needed. Optional: a single headline stat or news headline.

---

## 3. Data: the College Scorecard

- Source: U.S. Dept. of Education College Scorecard, 1996–2023.
- Unit of analysis: **(school × cohort year)**.
- Persistent panel: schools that report in every year → **3,790
  schools × 28 years = 106,120 rows**.
- ~70 institutional features per row: sector, degrees, costs,
  selectivity, student demographics, program mix (38 PCIP cols).

**Figure:** coverage heatmap from `notebooks/Exploration.Rmd`
(core variable coverage by cohort year). Shows which variables
are populated in which years and motivates the next slide.

---

## 4. Outcome: CDR3 (3-year cohort default rate)

- **CDR3** = % of a school's federal loan borrowers who default
  within 3 years of repayment.
- Best-covered distress signal: 90%+ from 2011 onward.
- Modeled as a continuous regression target (preserves info vs.
  binary "distressed" threshold).

Optional figure: small inline of the CDR3 histogram (one panel,
not the year-faceted one) so viewers see the distribution shape.

---

## 5. Pre-processing

Show the pipeline as a short flow:

1. Filter to rows with non-null CDR3 → 2011–2023 window.
2. Backfill time-invariant flags (HBCU, HSI, LOCALE) within school.
3. **Drop leakage columns**: `RPY_3YR_RT`, `MD_EARN_WNE_*`,
   `GRAD_DEBT_MDN10YR`.
4. Coverage filter: drop variables with <50% non-null (protect
   `SAT_AVG`, `ADM_RATE`, `FIRST_GEN`, `MD_FAMINC`).
5. Factor recode + `log1p` transform on right-skewed resource vars.
6. **Drop 2020–2023** (COVID payment pause → CDR3 = 0 for ~all
   schools, no signal).

End state: `distress_primary.csv` — the modeling dataset.

No figure required, but the year-faceted CDR3 histogram from
`distress_EDA.Rmd` is a strong justification visual for step 6
(the 2022/2023 panels are flat at zero).

---

## 6. EDA — outcome distribution by year

**Figure:** CDR3 histogram faceted by year (from `distress_EDA.Rmd`).

Talking points:
- Distribution is roughly stable 2011–2019.
- 2020–2023 collapse to zero → COVID forbearance, not signal.
- This is what justifies cutting the panel at 2019.

---

## 7. EDA — where does the variance live?

The headline EDA finding.

- Variance decomposition: between-school vs within-school vs year.
- ICC from `lmer(CDR3 ~ 1 + (1|UNITID) + (1|year))`:
  **72.9% between schools, 1.8% between years**, ~25% within.
- "Default rate is a property of the school, not the year."

**Figure:** spaghetti plot from `distress_EDA.Rmd` — CDR3
trajectories for 75 sampled schools per sector. Schools track
horizontal bands; sectors are visibly different.

---

## 8. EDA — sector and predictor relationships

Two plots side by side if space, otherwise pick one.

**Figure A:** boxplot of CDR3 by `CONTROL` and `PREDDEG`
(categorical predictors panel from `distress_EDA.Rmd`).
- For-profits and certificate-granting schools have the highest
  median CDR3 by a wide margin.

**Figure B:** scatter + GAM smooth of CDR3 vs continuous
predictors (PCTPELL, ADM_RATE, C150, GRAD_DEBT_MDN, NPT4).
- Strong negative relationship with C150 (completion).
- Strong positive with PCTPELL (Pell share).
- Mildly counterintuitive negative with `GRAD_DEBT_MDN` (selective
  schools = high debt + high earnings = low default).

---

## 9. Methods — modeling strategy

Walk through the model ladder. No figure — just bullets.

- **Train/test split:** by school (group split on UNITID), 80/20.
  Prevents the model from memorizing schools across folds.
- **Null model:** predict mean CDR3. Floor.
- **OLS baseline:** 8 hand-picked features, simple `lm()`.
- **Elastic net (glmnet):** ~80 features, penalty = 0.00113,
  mixture = 0.2 (mostly Ridge). Penalty/mixture chosen via grouped
  5-fold CV grid search.
- **Variance decomposition:** mixed-effects `lmer` with school +
  year random intercepts.
- **Robustness check:** sector × feature interactions to test
  whether the for-profit gap is a coefficient problem.

---

## 10. Results — headline metrics table

Present as a 4-row table:

| Model | RMSE | R² | Notes |
|---|---|---|---|
| Null | 0.0789 | 0.000 | mean of train CDR3 |
| OLS (8 features) | 0.0588 | 0.443 | hand-picked, readable |
| Elastic net (~80 features) | 0.0551 | 0.512 | 76 nonzero coefs |

- 30.2% RMSE reduction over the null.
- EN beats OLS by only **6.4%** — most of the signal is in a few
  obvious features.

---

## 11. Results — predicted vs actual (the diagnostic)

**Figure:** the predicted-vs-actual scatter from `analysis.Rmd`,
colored by sector with per-sector OLS lines.

Talking points:
- Nonprofits and publics track the diagonal reasonably well.
- For-profit slope is visibly flatter — model assigns them a
  near-sector-average regardless of features.
- This is the visual that motivates the sector analysis.

---

## 12. Results — sector heterogeneity

**Figure / table:** the per-sector RMSE comparison from the
interactions chunk:

| Model | Nonprofit | Public | For-profit |
|---|---|---|---|
| OLS | ~0.041 | ~0.056 | ~0.074 |
| OLS + sector interactions | ~0.041 | ~0.056 | ~0.074 |
| EN | ~0.041 | ~0.056 | ~0.074 |
| EN + sector interactions | ~0.041 | ~0.056 | ~0.074 |

(Pull the exact numbers from the knit `analysis.md`.)

Talking points:
- For-profit error is ~1.8× nonprofit error.
- Adding `CONTROL × {C150, PCTPELL}` interactions moves RMSE by
  **<0.001** in every cell.
- Conclusion: it's not a coefficient problem, it's a
  **feature-coverage** problem. Whatever drives for-profit
  defaults isn't in the Scorecard.

---

## 13. Results — variance decomposition

Reinforce the EDA finding now that the modeling is done.

| Source | % of CDR3 variance |
|---|---|
| Between schools (UNITID) | 72.9% |
| Between years | 1.8% |
| Within school (residual) | ~25% |

- The elastic net is recovering structural school traits, not
  cohort effects.
- Validates the school-grouped train/test split — a row-level
  random split would have leaked school identity.

---

## 14. Conclusions

Three takeaways, mirrors "The Story" section of the Rmd.

1. **Default rates are a structural property of schools.**
   72.9% of variance is between-school, 1.8% is year-to-year.
2. **Institutional features explain ~half of that variation.**
   EN R² = 0.512 on held-out schools. Most of the signal is in
   obvious features (sector, completion, Pell share, degree level).
3. **For-profits are predicted by sector, not features.** Per-sector
   RMSE is 1.8× higher; sector interactions don't fix it. Future
   work needs data the Scorecard doesn't collect (financial health,
   recruitment practices, accreditation status).

---

## 15. (Optional) Limitations + future work

- Institution-level medians hide within-school heterogeneity
  (program-level outcomes would be better).
- Title IV–only — for-profits not on federal aid are excluded.
- Selection bias: schools that close drop out of the panel.
- Future: program-level data, financial health indicators for
  for-profits, robustness check with `RPY_3YR_RT`.

---

## Figure inventory — where each lives

| Slide | Figure | Source file |
|---|---|---|
| 3 | Coverage heatmap | `notebooks/Exploration.Rmd` |
| 5/6 | CDR3 histogram by year | `notebooks/distressed_analysis/distress_EDA.Rmd` |
| 7 | Spaghetti plot by sector | `distress_EDA.Rmd` |
| 8A | CDR3 by CONTROL / PREDDEG boxplot | `distress_EDA.Rmd` |
| 8B | CDR3 vs continuous predictors | `distress_EDA.Rmd` |
| 11 | Predicted vs actual (colored) | `analysis.Rmd` |
| 12 | Per-sector RMSE table | `analysis.Rmd` interactions chunk |
| 13 | ICC table | `analysis.Rmd` icc chunk |

Optional but unused (cut if time-tight):
- Mean vs within-school SD log-log scatter (`distress_EDA.Rmd`).
- Correlation heatmap of continuous predictors (`distress_EDA.Rmd`).
- OLS coefficient summary (`summary(ols_fit)` in `analysis.Rmd`).
