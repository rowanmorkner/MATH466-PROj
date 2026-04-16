# Build the processed Scorecard datasets.
#
# 1. For each cohort year, load the MERGED*_PP.csv, reduce it to the
#    KEEP_COLS feature set via extract_df(), tag with the cohort year,
#    and write a per-year data/processed/scorecard_<year>.csv for
#    single-year analyses.
# 2. Keep every year's dataframe in memory, then filter to UNITIDs
#    that appear in every cohort (the "persistent panel") and stack
#    into one long tibble.
# 3. Write data/processed/panel.csv with one row per (school, year)
#    for the panel analyses.

suppressPackageStartupMessages({
  library(here)
  library(readr)
  library(dplyr)
})

# Anchor paths to the project root via the here package. here() finds
# the .Rproj file at the repo root, so the script works regardless of
# which directory it's invoked from (Rscript, RStudio, knitr, etc.).
source(here("src", "preprocessing_utils.R"))

RAW_DIR <- here("data", "raw")
PROCESSED_DIR <- here("data", "processed")

years <- 1996:2023

# Load and tag every cohort.
yearly <- vector("list", length(years))
names(yearly) <- as.character(years)

for (year in years) {
  # Filename pattern is MERGED<year>_<(year+1) last two digits>_PP.csv,
  # e.g. MERGED2011_12_PP.csv, MERGED1999_00_PP.csv. Using
  # (year + 1) %% 100 wraps correctly across the century boundary
  # (1999 -> 00, 2000 -> 01); %02d zero-pads the 2000-2009 range.
  filepath <- file.path(
    RAW_DIR,
    sprintf("MERGED%d_%02d_PP.csv", year, (year + 1) %% 100)
  )
  df <- extract_df(filepath) |>
    mutate(year = year)
  yearly[[as.character(year)]] <- df

  # Per-year CSV for single-year analyses.
  year_out <- file.path(PROCESSED_DIR, sprintf("scorecard_%d.csv", year))
  write_csv(df, year_out)
  message(sprintf("Processed %s -> %s (%d rows)", filepath, year_out, nrow(df)))
}

# Intersection of UNITIDs across every cohort = schools that report
# data in all 28 years. These are the persistent panel.
persistent_unitids <- Reduce(intersect, lapply(yearly, `[[`, "UNITID"))

# Filter each year to persistent schools and stack into one long tibble.
panel <- lapply(
  yearly,
  function(df) filter(df, UNITID %in% persistent_unitids)
) |>
  bind_rows()

panel_out <- file.path(PROCESSED_DIR, "panel.csv")
write_csv(panel, panel_out)

message(sprintf(
  "Wrote %s: %d persistent schools x %d years = %d rows",
  panel_out,
  length(persistent_unitids),
  length(years),
  nrow(panel)
))
