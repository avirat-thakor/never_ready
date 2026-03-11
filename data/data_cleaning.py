import pandas as pd


def load_car_data(path, target_name):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Year, Month"], format="%Y, %b")
    value_col = "Sales" if "Sales" in df.columns else "Value"
    df = df[["date", value_col]].rename(columns={value_col: target_name})
    df[target_name] = df[target_name].round().astype("Int64")
    return df

def load_macro_monthly(path, series_name):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["observation_date"])
    return df[["date", series_name]]

def load_tdsp_quarterly_to_monthly(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["observation_date"])
    df = df[["date", "TDSP"]].set_index("date")
    df = df.resample("MS").ffill().reset_index()
    return df


civic   = load_car_data("data/raw_data/CivicData.csv",    "civic_sales")
corolla = load_car_data("data/raw_data/CorrollaData.csv", "corolla_sales")
sentra  = load_car_data("data/raw_data/SentraData.csv",   "sentra_sales")
cpi     = load_macro_monthly("data/raw_data/CPILFESL.csv", "CPILFESL")
fedfund = load_macro_monthly("data/raw_data/FEDFUNDS.csv", "FEDFUNDS")
gas     = load_macro_monthly("data/raw_data/GASREGW.csv",  "GASREGW")
unrate  = load_macro_monthly("data/raw_data/UNRATE.csv",   "UNRATE")
csi     = load_macro_monthly('data/raw_data/UMCSENT.csv', "UMCSENT")
tdsp    = load_tdsp_quarterly_to_monthly("data/raw_data/TDSP.csv")

combined = civic.merge(corolla, on="date", how="left")
combined = combined.merge(sentra,  on="date", how="left")
combined = combined.merge(cpi,     on="date", how="left")
combined = combined.merge(fedfund, on="date", how="left")
combined = combined.merge(gas,     on="date", how="left")
combined = combined.merge(unrate,  on="date", how="left")
combined = combined.merge(csi,  on="date", how="left")
combined = combined.merge(tdsp,    on="date", how="left")

combined = combined.rename(columns={
    "CPILFESL": "cpi",
    "UNRATE": "unemploy",
    "FEDFUNDS": "fedfunds",
    "GASREGW": "gas",
    "TDSP": "tdsp",
    "UMCSENT": "csi"
})
combined = combined.sort_values("date").reset_index(drop=True)
combined = combined.ffill() 


combined.to_csv("data/combined_table.csv", index=False)

