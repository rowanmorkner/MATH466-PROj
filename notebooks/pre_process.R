# Loop over MERGED cohort years, reduce each to the KEEP_COLS feature
# set via extract_df(), and write the cleaned CSVs to data/processed/.
# Mirror of notebooks/pre_process.py.

suppressPackageStartupMessages({
  library(here)
  library(readr)
})

# Anchor paths to the project root via the here package. here() finds
# the .Rproj file at the repo root, so the script works regardless of
# which directory it's invoked from (Rscript, RStudio, knitr, etc.).
source(here("src", "preprocessing_utils.R"))

RAW_DIR <- here("data", "raw")
PROCESSED_DIR <- here("data", "processed")

years <- c(1996:2023)

for (year in years) {
  # Filename pattern is MERGED<year>_<(year+1) last two digits>_PP.csv,
  # e.g. MERGED2011_12_PP.csv, MERGED1999_00_PP.csv. Using (year + 1) %% 100
  # wraps correctly across the century boundary (1999 -> 00, 2000 -> 01),
  # and %02d zero-pads the 2000-2009 range.
  filepath <- file.path(
    RAW_DIR,
    sprintf("MERGED%d_%02d_PP.csv", year, (year + 1) %% 100)
  )
  df <- extract_df(filepath)
  out <- file.path(PROCESSED_DIR, sprintf("scorecard_%d.csv", year))
  write_csv(df, out)
  message(sprintf("Processed %s -> %s", filepath, out))
}
