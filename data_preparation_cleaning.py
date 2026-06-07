from __future__ import annotations
import re
from pathlib import Path
import pandas as pd


DATA_URL = (
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/"
    "titanic.csv"
)
OUTPUT_PATH = Path("cleaned_dataset.csv")


def normalize_column_name(col: str) -> str:
    col = col.strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_")


def coerce_numeric_series(s: pd.Series) -> pd.Series:
   
    if s.dtype == "object":
       
        s2 = s.astype(str).str.replace(r"[,\s]", "", regex=True)
       
        s2 = s2.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        return pd.to_numeric(s2, errors="coerce")
    return s


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {}

    report["initial_rows"] = len(df)
    report["initial_cols"] = len(df.columns)

   
    df = df.copy()
    df.columns = [normalize_column_name(str(c)) for c in df.columns]

   
    before_dupes = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before_dupes - len(df)

   
    for col in df.columns:
        df[col] = coerce_numeric_series(df[col])


    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "string", "boolean"]).columns.tolist()

   
    for col in numeric_cols:
        if df[col].isna().any():
            median = df[col].median(skipna=True)
            df[col] = df[col].fillna(median)

    for col in object_cols:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "unknown"
            
            if df[col].dtype == "object" or df[col].dtype.name.startswith("string"):
                df[col] = df[col].astype("string").str.strip()
                df[col] = df[col].replace({"": pd.NA})
            df[col] = df[col].fillna(fill_value)

    
    datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    for col in datetime_cols:
        if df[col].isna().any():
            df[col] = df[col].ffill().bfill()

   
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA})

    report["final_rows"] = len(df)
    report["final_cols"] = len(df.columns)
    report["missing_values_total_after"] = int(df.isna().sum().sum())

    return df, report

    if __name__ == "__main__":

       main() 



