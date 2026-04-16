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

years <- c(2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
           2021, 2022, 2023)

for (year in years) {
  # Filename pattern is MERGED<year>_<(year+1) last two digits>_PP.csv,
  # e.g. MERGED2011_12_PP.csv. %02d zero-pads so years like 2000 would
  # still format correctly if the list ever grew backward.
  filepath <- file.path(
    RAW_DIR,
    sprintf("MERGED%d_%02d_PP.csv", year, year - 1999)
  )
  df <- extract_df(filepath)
  out <- file.path(PROCESSED_DIR, sprintf("scorecard_%d.csv", year))
  write_csv(df, out)
  message(sprintf("Processed %s -> %s", filepath, out))
}
