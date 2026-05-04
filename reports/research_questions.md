---
editor_options: 
  markdown: 
    wrap: 72
---

# Research Question Feasibility & Proposed Methods

Scoping notes from exploring `data/processed/panel.csv` (3,790 schools ×
28 years = 106,120 rows) on 2026-04-16.

## Data reality checks

Two findings drive everything downstream:

1.  **Earnings are nearly cross-sectional.** `MD_EARN_WNE_P10` is
    populated in only **7 cohort years** (2007, 2009, 2011, 2012, 2013,
    2014, 2020). Between-school variance accounts for **\~100%** of
    total earnings variance — within-school variance is effectively
    zero. Translation: once you pick a school, its earnings number
    barely moves over the panel. Panel / fixed-effects regressions on
    earnings buy you almost nothing.
2.  **PCIP program shares are the MVP column family.** 95%+ coverage in
    every one of the 28 years. `CDR3` (3-yr default rate) is the
    second-best — 90%+ from 2011 on. Those two are the real longitudinal
    signals.

Other coverage notes:

-   `NPT4` (net price): 92%+ from 2009 on.
-   `C150` (completion): 93%+ from 1997 on.
-   `GRAD_DEBT_MDN`: \~80% from 1997–2020.
-   `RPY_3YR_RT`: only 2009–2016 (\~84%).
-   `SAT_AVG` / `ADM_RATE`: \~30–48% (selective schools only).
-   `PREDDEG` / `CONTROL`: 100% always — great segmentation variables.

## Original five questions — feasibility

| \# | Question | Feasibility | Why |
|------------------|------------------|------------------|------------------|
| 1 | Predict median earnings | ✅ Strong | Cross-sectional on the populated cohort years; rich predictor set; ML-friendly. |
| 2 | School name vs. degree | ⚠️ Reformulable | "School name" = school identity / between-school variance. Variance decomposition answers this naturally. |
| 3 | What puts students in debt / unable to repay | ✅ Strongest | Best-covered outcome (`CDR3`, `RPY_3YR_RT`, `GRAD_DEBT_MDN`). Rich longitudinal panel. |
| 4 | Greatest ROI | ⚠️ Narrow window | Bottlenecked by the 7 earnings years. Feasible but shallow; mostly duplicates Q1 with a cost denominator. |
| 5 | Majors rising / falling | ✅ Easy but shallow | PCIP trends are trivial to compute. Risk of being *too* descriptive for a stats project unless paired with outcomes. |

## Proposed research questions (ranked)

### 1. "What institutional features predict graduate loan distress?" (strongest)

Matches the data's strengths, has a clear outcome, and is a natural home
for advanced methods.

-   **Outcome:** `CDR3` (3-yr default rate) or
    `GRAD_DEBT_MDN`-to-earnings ratio.
-   **Coverage:** 90%+, 2011–2023, a true panel.
-   **Methods:**
    -   Gradient boosting (XGBoost / LightGBM) + SHAP for interpretable
        feature importance.
    -   Hierarchical logistic regression with school random effects to
        separate within-school drift from cross-school type.
    -   Penalized regression (LASSO / ElasticNet) as a sparse-model
        baseline.
-   **Absorbs:** Q3 fully and most of Q4.

### 2. "A typology of American colleges and their outcome profiles."

Unsupervised structure discovery followed by outcome mapping — strong
visual storytelling for the report.

-   **Input:** the full \~70-variable feature space (PCIP mix + cost +
    selectivity + student body).
-   **Methods:**
    -   PCA / UMAP for 2-D visualization.
    -   Gaussian mixture models or k-means for discrete archetypes.
    -   **Functional clustering on PCIP trajectories** — each school is
        a curve over 28 years; cluster the curves to find schools that
        have *shifted* their program mix over time.
-   **Follow-up:** do archetypes differ on earnings / debt / completion?
-   **Absorbs:** Q5 with more depth than just "nursing up, humanities
    down."

### 3. "How much of earnings variation is school identity vs. program mix vs. student selectivity?"

Directly answers Q2 in its correct statistical framing.

-   **Model:**
    `earnings ~ PREDDEG + PCIP + SAT_AVG + ADM_RATE + PCTPELL + (1|school)`
-   **Reporting:** ICC and R² contributions by block.
-   **Methods:**
    -   Linear mixed models (`lme4`).
    -   Relative importance analysis (Shapley value regression).
    -   Ridge regression with grouped predictors.

## Recommendation

Lead with **#1** (debt / repayment prediction) — best coverage, clear
outcome, advanced-method-friendly. Use **#2** (typology) as a second
movement: the unsupervised step finds the structure, then show how
archetypes map to outcomes.
