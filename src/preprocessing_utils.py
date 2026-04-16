"""Utilities for reducing a College Scorecard MERGED*_PP.csv cohort file
down to the ~70-variable institution-level feature set the project uses."""

import pandas as pd

# Null sentinel tokens declared in data/raw/data.yaml. These get mapped
# to NaN on read so downstream numeric coercion works cleanly.
NULL_TOKENS = ["NULL", "PrivacySuppressed", "NA", "PS"]

# CIP-2 program share columns (PCIP01..PCIP54). The gaps (e.g. no PCIP02,
# PCIP17, PCIP18) are intentional -- those CIP-2 codes don't exist in the
# Scorecard schema.
PCIP_COLS = [
    "PCIP01", "PCIP03", "PCIP04", "PCIP05", "PCIP09", "PCIP10",
    "PCIP11", "PCIP12", "PCIP13", "PCIP14", "PCIP15", "PCIP16",
    "PCIP19", "PCIP22", "PCIP23", "PCIP24", "PCIP25", "PCIP26",
    "PCIP27", "PCIP29", "PCIP30", "PCIP31", "PCIP38", "PCIP39",
    "PCIP40", "PCIP41", "PCIP42", "PCIP43", "PCIP44", "PCIP45",
    "PCIP46", "PCIP47", "PCIP48", "PCIP49", "PCIP50", "PCIP51",
    "PCIP52", "PCIP54",
]

# Grouped so a block can be added or dropped without hunting through one
# flat list. The guiding rule: one measure per concept, one time horizon,
# overall population -- subgroup cuts only when a question needs them.
KEEP_COLS = {
    # Identifiers + name, for joining across years and labeling.
    "id": ["UNITID", "OPEID", "INSTNM"],
    # Structural and resource descriptors of the institution itself.
    # CONTROL/PREDDEG/REGION/LOCALE are integer-coded categoricals;
    # HBCU/HSI/DISTANCEONLY are 0/1 flags.
    "inst_chars": [
        "CONTROL", "PREDDEG", "REGION", "LOCALE",
        "HBCU", "HSI", "DISTANCEONLY",
        "INEXPFTE", "AVGFACSAL", "PFTFAC", "TUITFTE", "STUFACR",
    ],
    # Selectivity signals. Individual SAT/ACT percentile fields are
    # dropped -- SAT_AVG captures the same dimension in one number.
    "admissions": ["ADM_RATE", "SAT_AVG"],
    # Student body composition. Only the four largest race groups are
    # kept; the pre-2009 and 2000-era race fields are dropped.
    "student_body": [
        "UGDS", "PCTPELL", "PCTFLOAN", "MD_FAMINC", "FIRST_GEN",
        "UGDS_WHITE", "UGDS_BLACK", "UGDS_HISP", "UGDS_ASIAN",
        "FEMALE", "UG25ABV",
    ],
    # Academic profile as 38 program-share percentages.
    "academics": PCIP_COLS,
    # Net price -- public and private are mutually exclusive by CONTROL,
    # so they get coalesced into a single NPT4 column below.
    "cost": ["NPT4_PUB", "NPT4_PRIV"],
    # 150%-time completion rate, 4yr and <4yr. Mutually exclusive by
    # institution type; coalesced into a single C150 column below.
    # Using the plain (not _POOLED_SUPP) variants because the pooled/
    # suppressed versions are only filled in the most-recent cohort
    # file, while C150_4 / C150_L4 are populated across most cohorts.
    "completion": ["C150_4", "C150_L4"],
    # Median completer debt (principal + 10-yr monthly payment).
    "debt": ["GRAD_DEBT_MDN", "GRAD_DEBT_MDN10YR"],
    # Repayment signals. CDR3 (3-yr cohort default rate) is populated
    # across nearly every cohort 1997-2023 and serves as the long-range
    # panel metric. RPY_3YR_RT (3-yr repayment rate) only covers
    # ~2009-2016 but gives a different-methodology check in overlap
    # years; keeping both lets the research question pick.
    "repayment": ["RPY_3YR_RT", "CDR3"],
    # Median earnings of working-not-enrolled graduates at 6 and 10 yrs.
    # Kept both so the 6-yr value is available as a robustness check
    # without a second pass over the raw file.
    "earnings": ["MD_EARN_WNE_P10", "MD_EARN_WNE_P6"],
}


def _flat_keep_cols() -> list[str]:
    """Flatten KEEP_COLS into the list of column names to request."""
    return [c for group in KEEP_COLS.values() for c in group]


def extract_df(filepath: str) -> pd.DataFrame:
    """Load a College Scorecard MERGED*_PP.csv and return the reduced,
    cleaned institution-level dataframe defined by KEEP_COLS.

    Coalesces NPT4_PUB/NPT4_PRIV into a single NPT4 column and
    C150_4/C150_L4 into a single C150. Raises KeyError if any requested
    column is missing from the file, so silent schema drift across
    cohort years is caught early.
    """
    requested = _flat_keep_cols()

    # Peek at the header first. MERGED files are multi-hundred-MB, so
    # validating column presence before the full read avoids a long
    # load just to discover a missing column.
    header = pd.read_csv(filepath, nrows=0).columns
    missing = [c for c in requested if c not in header]
    if missing:
        raise KeyError(
            f"Columns missing from {filepath}: {missing}"
        )

    df = pd.read_csv(
        filepath,
        usecols=requested,
        na_values=NULL_TOKENS,
        keep_default_na=True,
        # Forces pandas to scan the whole column before picking a dtype;
        # otherwise mixed-token columns get inferred as object.
        low_memory=False,
    )

    # Coalesce net price. Exactly one of NPT4_PUB / NPT4_PRIV is populated
    # per row (driven by CONTROL), so fillna gives a single comparable
    # net-price column regardless of institution type.
    df["NPT4"] = df["NPT4_PUB"].fillna(df["NPT4_PRIV"])
    df = df.drop(columns=["NPT4_PUB", "NPT4_PRIV"])

    # Same coalesce for 150%-time completion: the 4yr and <4yr rates
    # are mutually exclusive by institution type.
    df["C150"] = df["C150_4"].fillna(df["C150_L4"])
    df = df.drop(columns=["C150_4", "C150_L4"])

    return df
