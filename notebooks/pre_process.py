import sys, pathlib 
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.preprocessing_utils import extract_df 

RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]


for year in years: 
    filepath = RAW_DIR / f"MERGED{year}_{year-1999}_PP.csv"
    df = extract_df(filepath)

    out = PROCESSED_DIR / f"scorecard_{year}.csv"
    df.to_csv(out, index=False)
    print(f"Processed {filepath} -> {out}")

