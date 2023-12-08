import pandas as pd
import numpy as np
from os.path import isfile

MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5,
                    "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10,
                    "NOV": 11, "DEC": 12}

def show_col_dtypes(df):
    for col in df.columns.values:
        dtype = type(df[col].values[0])
        print(f"{col}: {dtype}")

def height_to_inches(in_df, cname="Height"):
    df = in_df.copy()
    df[["ft","in"]] = df[cname].str.split("\' ", n=1, expand=True)
    df["in"] = df["in"].str[:-1]
    df["in"] = df["in"].astype(np.float32)
    df["ft"] = df["ft"].astype(np.float32)
    df[cname] = (df["ft"] * 12.0) + df["in"]
    return df[cname]

def remove_last(df, cname, n=1):
    rem = -1 * n
    return df[cname].str[:rem]

def decode_date(df, cname="DOB"):
    df[["birth_month", "birth_day", "birth_year"]] = df[cname].str.split(" ", n=2, expand=True)
    df["birth_day"] = remove_last(df, "birth_day").astype(np.float32)
    df["birth_month"] = df["birth_month"].str[:3]
    df["birth_month"] = df["birth_month"].str.upper().map(MONTHS).astype(np.float32)
    df["birth_year"] = df["birth_year"].astype(np.float32)
    return df.drop([cname], axis=1)

if __name__ == "__main__":
    # data from: https://www.kaggle.com/datasets/rajeevw/ufcdata?select=preprocessed_data.csv
    fighters_fname = "fighters.csv"
    bouts_fname = "bouts.csv"
    if not isfile(fighters_fname):
        fighters = pd.read_csv("raw_fighter_details.csv")
        # cleanup fighters df
        # show_col_dtypes(fighters)
        fighters["Stance"] = fighters["Stance"].fillna("unkown_stance")
        fighters["Stance"] = fighters["Stance"].astype(str)
        stance = pd.get_dummies(fighters["Stance"])
        stance.columns = ["open_stance", "orthodox_stance", "sideways_stance",
                        "southpaw_stance", "switch_stance", "unkown_stance"]
        stance = stance.astype(np.float32)
        fighters.drop(["Stance"], axis=1, inplace=True)
        fighters = fighters.join(stance)
        fighters["Weight"] = fighters["Weight"].str.split(" ", n=1, expand=True)[0].astype(np.float32)
        fighters["Height"] = height_to_inches(fighters)
        rem_last = ["Reach", "Str_Acc", "Str_Def", "TD_Acc", "TD_Def"]
        for col in rem_last:
            fighters[col] = remove_last(fighters, col).astype(np.float32)
        fighters = decode_date(fighters)
        fighters.insert(0, "fighter_id", range(fighters.shape[0]))
        print(fighters)
        fighters.to_csv(fighters_fname)
    else:
        fighters = pd.read_csv(fighters_fname)
    if not isfile(bouts_fname):
        bouts = pd.read_csv("raw_total_fight_data.csv", sep=';')

        # cleanup bouts dataframe
        print("Bout Details:")
        show_col_dtypes(bouts)