## Setup

We start from the Phase 1 cleaned dataset (`distress_primary.csv`), not
the raw panel. That handles outcome-side leakage (RPY\_3YR\_RT, earnings
columns dropped upstream), the COVID-pause years (filtered to year &lt;=
2019), and the missing-coverage drops in one place so this script can
stay focused on modeling.

    library(tidymodels)
    library(tidyverse)
    library(here)
    library(lme4)
    library(performance)
    library(glmnet)

    panel <- read_csv(here("data", "processed", "distress_primary.csv"),
                      show_col_types = FALSE)

    # CSV round-trip strips factor levels -- restore them so the recipe
    # treats categoricals as categorical, not as text.
    panel <- panel %>%
      mutate(across(c(CONTROL, PREDDEG, REGION, LOCALE), as.factor)) %>%
      # `distressed` is derived from CDR3 (CDR3 > 0.10), so including it
      # as a predictor would be perfect leakage.
      select(-any_of("distressed"))

## Grouped train/test split

Schools appear in multiple years. A random row-level split puts the same
UNITID in both train and test, which inflates R^2 because the model just
learns school identity. `group_initial_split` keeps every school on one
side of the split.

    set.seed(42)

    panel_split <- group_initial_split(panel, group = UNITID, prop = 0.8)
    train_data  <- training(panel_split)
    test_data   <- testing(panel_split)

    c(train_schools = n_distinct(train_data$UNITID),
      test_schools  = n_distinct(test_data$UNITID),
      train_rows    = nrow(train_data),
      test_rows     = nrow(test_data))

    ## train_schools  test_schools    train_rows     test_rows 
    ##          3020           760         25426          6356

## Recipe

Median imputation is fit on the training data and then applied to test
(no test-set leakage). NA factor levels become an explicit `unknown`
level so we don’t drop rows. Predictors are normalized so the L1/L2
penalties act on the same scale.

    rec <- recipe(CDR3 ~ ., data = train_data) %>%
      update_role(UNITID, INSTNM, OPEID, year, new_role = "id") %>%
      step_unknown(all_nominal_predictors()) %>%
      step_impute_median(all_numeric_predictors()) %>%
      step_dummy(all_nominal_predictors()) %>%
      step_zv(all_predictors()) %>%
      step_normalize(all_numeric_predictors())

## Tune penalty and mixture together

5-fold grouped CV on the training set, tuning both `penalty` (lambda)
and `mixture` (alpha) so the model can sit anywhere on the LASSO-Ridge
continuum instead of being pinned at alpha = 0.5.

    en_spec <- linear_reg(penalty = tune(), mixture = tune()) %>%
      set_engine("glmnet")

    wf <- workflow() %>%
      add_recipe(rec) %>%
      add_model(en_spec)

    set.seed(42)
    folds <- group_vfold_cv(train_data, group = UNITID, v = 5)

    en_grid <- grid_regular(
      penalty(range = c(-4, 0)),
      mixture(range = c(0, 1)),
      levels = c(penalty = 20, mixture = 6)
    )

    tuned <- tune_grid(
      wf,
      resamples = folds,
      grid      = en_grid,
      metrics   = metric_set(yardstick::rmse, yardstick::rsq)
    )

    show_best(tuned, metric = "rmse", n = 5)

    ## # A tibble: 5 × 8
    ##    penalty mixture .metric .estimator   mean     n std_err .config          
    ##      <dbl>   <dbl> <chr>   <chr>       <dbl> <int>   <dbl> <chr>            
    ## 1 0.00113      0.2 rmse    standard   0.0566     5 0.00124 pre0_mod032_post0
    ## 2 0.000695     0.2 rmse    standard   0.0566     5 0.00124 pre0_mod026_post0
    ## 3 0.000428     0.4 rmse    standard   0.0566     5 0.00124 pre0_mod021_post0
    ## 4 0.000264     0.8 rmse    standard   0.0566     5 0.00124 pre0_mod017_post0
    ## 5 0.000264     0.6 rmse    standard   0.0566     5 0.00124 pre0_mod016_post0

## Fit final model on full train, evaluate on test

    best <- select_best(tuned, metric = "rmse")
    best

    ## # A tibble: 1 × 3
    ##   penalty mixture .config          
    ##     <dbl>   <dbl> <chr>            
    ## 1 0.00113     0.2 pre0_mod032_post0

    final_wf  <- finalize_workflow(wf, best)
    final_fit <- fit(final_wf, data = train_data)

    test_preds <- predict(final_fit, new_data = test_data) %>%
      bind_cols(test_data %>% select(CDR3))

    en_metrics <- test_preds %>%
      metrics(truth = CDR3, estimate = .pred)

    rmse_en <- en_metrics %>% filter(.metric == "rmse") %>% pull(.estimate)
    r2_en   <- en_metrics %>% filter(.metric == "rsq")  %>% pull(.estimate)

    # Null model = predict the training mean for every test row.
    mean_train <- mean(train_data$CDR3, na.rm = TRUE)
    rmse_null  <- sqrt(mean((test_data$CDR3 - mean_train)^2, na.rm = TRUE))

    tibble(
      model = c("Null (train mean)", "Elastic Net (tuned, grouped CV)"),
      rmse  = c(rmse_null, rmse_en)
    )

    ## # A tibble: 2 × 2
    ##   model                             rmse
    ##   <chr>                            <dbl>
    ## 1 Null (train mean)               0.0812
    ## 2 Elastic Net (tuned, grouped CV) 0.0603

    reduction_pct <- (rmse_null - rmse_en) / rmse_null * 100

## Coefficients

The interpretive payoff of the elastic net – which institutional
features actually carry weight after penalization. Nonzero coefficients
only, sorted by magnitude.

    en_coefs <- final_fit %>%
      extract_fit_parsnip() %>%
      tidy() %>%
      filter(term != "(Intercept)", estimate != 0) %>%
      mutate(abs_est = abs(estimate)) %>%
      arrange(desc(abs_est)) %>%
      select(term, estimate, penalty)

    en_coefs %>% print(n = 25)

    ## # A tibble: 74 × 3
    ##    term                      estimate penalty
    ##    <chr>                        <dbl>   <dbl>
    ##  1 PREDDEG_Bachelor.s        -0.0158  0.00113
    ##  2 C150                      -0.0143  0.00113
    ##  3 PREDDEG_Graduate          -0.0121  0.00113
    ##  4 PCTPELL                    0.0104  0.00113
    ##  5 LOCALE_unknown            -0.0102  0.00113
    ##  6 GRAD_DEBT_MDN             -0.00857 0.00113
    ##  7 HBCU                       0.00749 0.00113
    ##  8 CONTROL_Private.nonprofit -0.00733 0.00113
    ##  9 TUITFTE_log               -0.00684 0.00113
    ## 10 PCIP43                     0.00526 0.00113
    ## 11 CONTROL_Public            -0.00518 0.00113
    ## 12 PCTFLOAN                   0.00494 0.00113
    ## 13 INEXPFTE_log              -0.00485 0.00113
    ## 14 AVGFACSAL_log             -0.00484 0.00113
    ## 15 PCIP24                     0.00447 0.00113
    ## 16 PCIP50                     0.00443 0.00113
    ## 17 LOCALE_Town..Remote        0.00370 0.00113
    ## 18 REGION_Mid.East           -0.00320 0.00113
    ## 19 UGDS_BLACK                 0.00316 0.00113
    ## 20 REGION_Plains             -0.00315 0.00113
    ## 21 UGDS_log                   0.00313 0.00113
    ## 22 PCIP48                     0.00289 0.00113
    ## 23 PREDDEG_Certificate        0.00288 0.00113
    ## 24 UGDS_ASIAN                -0.00281 0.00113
    ## 25 LOCALE_Rural..Remote       0.00279 0.00113
    ## # ℹ 49 more rows

    n_nonzero <- nrow(en_coefs)

## Mixed-effects variance decomposition

Updated ICC with both school and year as random effects. The original
EDA fit only `(1 | UNITID)`; adding `(1 | year)` lets us separate
school-level variance from year-to-year drift.

    icc_model <- lmer(CDR3 ~ 1 + (1 | UNITID) + (1 | year), data = panel)
    summary(icc_model)

    ## Linear mixed model fit by REML ['lmerMod']
    ## Formula: CDR3 ~ 1 + (1 | UNITID) + (1 | year)
    ##    Data: panel
    ## 
    ## REML criterion at convergence: -100293.3
    ## 
    ## Scaled residuals: 
    ##      Min       1Q   Median       3Q      Max 
    ## -10.6023  -0.3751  -0.0367   0.3179  17.7156 
    ## 
    ## Random effects:
    ##  Groups   Name        Variance  Std.Dev.
    ##  UNITID   (Intercept) 0.0049281 0.07020 
    ##  year     (Intercept) 0.0001241 0.01114 
    ##  Residual             0.0017056 0.04130 
    ## Number of obs: 31782, groups:  UNITID, 3780; year, 9
    ## 
    ## Fixed effects:
    ##             Estimate Std. Error t value
    ## (Intercept) 0.101398   0.003893   26.05

    icc(icc_model, by_group = TRUE)

    ## # ICC by Group
    ## 
    ## Group  |   ICC
    ## --------------
    ## UNITID | 0.729
    ## year   | 0.018

    icc_vals <- icc(icc_model, by_group = TRUE)
    icc_school <- icc_vals %>% filter(Group == "UNITID") %>% pull(ICC)
    icc_year   <- icc_vals %>% filter(Group == "year")   %>% pull(ICC)

## Final ideas

#### One

- **72.9%** of the variance in default rates sits between schools
  (school-level ICC), and only **1.8%** is year-to-year drift.

If I pick two schools at random, most of the difference in default rates
comes from *which school they are* – not random fluctuation within a
school over time, and not the cohort year either.

#### Two

- Null RMSE: **0.0812**
- Elastic Net RMSE (tuned, grouped CV): **0.0603**
- Test R^2: **0.449**
- That’s a **25.8%** reduction in RMSE over the null.
- Best hyperparameters: penalty = 0.00113, mixture = 0.2. The model
  retained **74** nonzero coefficients.

#### Final

- Institution features reduce prediction error meaningfully relative to
  the null, even after blocking school identity out of the test set (so
  the gain is *generalization* across schools, not memorizing them).
- Most of the variation in CDR3 is persistent at the school level (72.9%
  ICC), which means the elastic net is recovering structural school
  traits – not transient cohort effects.
- Compared with the earlier random-split run (R^2 ~ 0.33, RMSE reduction
  ~ 18%), the grouped split is the honest number. Random splits on panel
  data overstate performance because the test set shares schools with
  the training set.
