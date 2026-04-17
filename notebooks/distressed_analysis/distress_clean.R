suppressPackageStartupMessages({
  library(here)
  library(readr)
  library(dplyr)
  library(tidyr)
})


PROCESSED_DIR <- here("data", "processed")

panel <- read_csv(here("data", "processed", "panel.csv"),
                  show_col_types = FALSE)

# Filter to rows where CDR3 is non-null. CDR3 is populated 2011-2023,
# so this also applies the analysis year window.
panel <- panel %>%
  filter(!is.na(CDR3))

# Backfill HBCU / HSI / LOCALE across years per school. HBCU is truly
# time-invariant; HSI and LOCALE change rarely. fill("updown") preserves any
# real year-to-year variation while filling gaps.
panel <- panel %>%
  arrange(UNITID, year) %>%
  group_by(UNITID) %>%
  fill(HBCU, HSI, LOCALE, .direction = "updown") %>%
  ungroup()

# Drop outcome-side columns that shouldn't be predictors of contemporaneous
# distress:
#   MD_EARN_WNE_P10 / P6 -- 6/10-yr post-entry earnings; sparse + circular.
#   RPY_3YR_RT           -- 3-yr repayment rate; alternative distress metric
#                           reserved for robustness comparison.
#   GRAD_DEBT_MDN10YR    -- 10-yr monthly payment derived from GRAD_DEBT_MDN.
outcome_side_cols <- c("MD_EARN_WNE_P10", "MD_EARN_WNE_P6",
                       "RPY_3YR_RT", "GRAD_DEBT_MDN10YR")
panel <- panel %>% select(-any_of(outcome_side_cols))

# Coverage check after the year filter. Columns we protect even if coverage
# < 50%:
#   SAT_AVG / ADM_RATE   -- selectivity signal; NA is informative (open
#                           enrollment / non-test-requiring schools) and the
#                           XGBoost stage handles NA natively.
#   FIRST_GEN / MD_FAMINC -- populated 2011-2016 only within the CDR3 window;
#                           handled by the two-dataset split at the end.
protected <- c("SAT_AVG", "ADM_RATE", "FIRST_GEN", "MD_FAMINC")

coverage <- panel %>%
  select(-INSTNM, -OPEID) %>%
  pivot_longer(-c(UNITID, year), names_to = "variable", values_to = "value") %>%
  group_by(variable) %>%
  summarise(pct_nonnull = mean(!is.na(value)) * 100, .groups = "drop")

cols_to_drop <- coverage %>%
  filter(pct_nonnull < 50, !variable %in% protected) %>%
  pull(variable)

panel_clean <- panel %>%
  select(-all_of(cols_to_drop))

message(sprintf("Dropped %d columns below 50%% coverage: %s",
                length(cols_to_drop),
                paste(cols_to_drop, collapse = ", ")))


# Feature engineering: integer-coded categoricals -> factors.
panel_clean <- panel_clean %>%
  mutate(
    CONTROL = factor(CONTROL, levels = c(1, 2, 3),
                     labels = c("Public", "Private nonprofit", "Private for-profit")),

    PREDDEG = factor(PREDDEG, levels = c(0, 1, 2, 3, 4),
                      labels = c("Not classified", "Certificate", "Associate's",
                                 "Bachelor's", "Graduate")),

    REGION = factor(REGION, levels = c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                    labels = c("U.S. Service Schools", "New England", "Mid East", "Great Lakes",
                               "Plains", "Southeast", "Southwest", "Rocky Mountains",
                               "Far West", "Outlying Areas")),

    LOCALE = factor(LOCALE, levels = c(11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43),
                    labels = c("City: Large", "City: Midsize", "City: Small",
                               "Suburb: Large", "Suburb: Midsize", "Suburb: Small",
                               "Town: Fringe", "Town: Distant", "Town: Remote",
                               "Rural: Fringe", "Rural: Distant", "Rural: Remote"))
  )

# Log-transform right-skewed institutional resource variables. log1p handles
# zeros safely. Raw columns are kept alongside the logged versions so
# downstream recipes can pick whichever works best.
panel_clean <- panel_clean %>%
  mutate(
    UGDS_log      = log1p(UGDS),
    AVGFACSAL_log = log1p(AVGFACSAL),
    INEXPFTE_log  = log1p(INEXPFTE),
    TUITFTE_log   = log1p(TUITFTE)
  )

# PCIP* columns left as-is (already proportions in [0, 1]).

# add  binary classification (e.g. `CDR3 > 0.10` = "distressed"; more interpretable, easier SHAP visuals)?

panel_clean <- panel_clean %>%
  mutate(
    distressed = factor(
      ifelse(CDR3 > 0.10, "distressed", "ok"),
      levels = c("ok", "distressed")
    )
  )
  
# Branch into the two analysis datasets.
#
# 1. distress_primary: full CDR3 window (2011-2023). Drops FIRST_GEN and
#    MD_FAMINC since they're retired after 2016 and would be >50% NA here.
distress_primary <- panel_clean %>%
  select(-any_of(c("FIRST_GEN", "MD_FAMINC"))) %>%
  filter(year <= 2019)

# 2. distress_firstgen: 2011-2016 overlap where FIRST_GEN and MD_FAMINC are
#    populated. Shorter panel, but lets the model use family-income and
#    first-generation signals.
distress_firstgen <- panel_clean %>%
  filter(year >= 2011, year <= 2016)

write_csv(distress_primary,  file.path(PROCESSED_DIR, "distress_primary.csv"))
write_csv(distress_firstgen, file.path(PROCESSED_DIR, "distress_firstgen.csv"))

message(sprintf(
  "Wrote distress_primary.csv:  %d rows x %d cols (%d schools, %d-%d)",
  nrow(distress_primary), ncol(distress_primary),
  n_distinct(distress_primary$UNITID),
  min(distress_primary$year), max(distress_primary$year)
))
message(sprintf(
  "Wrote distress_firstgen.csv: %d rows x %d cols (%d schools, %d-%d)",
  nrow(distress_firstgen), ncol(distress_firstgen),
  n_distinct(distress_firstgen$UNITID),
  min(distress_firstgen$year), max(distress_firstgen$year)
))




