import pandas as pd
import numpy as np
from os.path import isfile
pd.options.mode.chained_assignment = None

MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5,
                    "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10,
                    "NOV": 11, "DEC": 12}
WEIGHT_CLASSES_LBS = {"Strawweight": 115.0, "Flyweight": 125.0, "Bantamweight": 135.0,
                       "Featherweight": 145.0, "Lightweight": 155.0, "Welterweight": 170.0,
                       "Middleweight": 185.0, "Light Heavyweight": 205.0, "Heavyweight": 265.0}
WEIGHT_CLASSES_KGS = {"Strawweight": 52.2, "Flyweight": 56.7, "Bantamweight": 61.2,
                       "Featherweight": 65.8, "Lightweight": 70.3, "Welterweight": 77.1,
                       "Middleweight": 83.9, "Light Heavyweight": 93.0, "Heavyweight": 120.2}

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

# removes the last n characters from every entry in a particular column
def remove_last(df, cname, n=1):
    rem = -1 * n
    return df[cname].str[:rem]

def decode_date(df, cname="DOB", prefix=""):
    new_cols = [f"{prefix}month", f"{prefix}day", f"{prefix}year"]
    df[new_cols] = df[cname].str.split(" ", n=2, expand=True)
    df[new_cols[1]] = remove_last(df, new_cols[1]).astype(np.float32)
    df[new_cols[0]] = df[new_cols[0]].str[:3]
    df[new_cols[0]] = df[new_cols[0]].str.upper().map(MONTHS).astype(np.float32)
    df[new_cols[2]] = df[new_cols[2]].astype(np.float32)
    return df.drop([cname], axis=1)

def decode_format(df, cname="Format"):
    df["rounds"] = df[cname].str[0]
    df["rounds"] = df["rounds"].str.replace("N", "0")
    df["rounds"] = df["rounds"].astype(np.float32)
    return df.drop([cname], axis=1)

def decode_location(df, cname="location"):
    df[["city", "country"]] = df[cname].str.split(", ", n=1, expand=True)
    df["country"] = df["country"].str.split(", ").str[-1]
    return df.drop([cname], axis=1)

def decode_winner(df):
    print(df["win_by"].unique())
    df["Draw/NC"] = ((df["Winner"] != df["R_fighter"]) & 
                        (df["Winner"] != df["B_fighter"]))
    df["R"] = df["Winner"] == df["R_fighter"]
    df["B"] = df["Winner"] == df["B_fighter"]
    df["winner"] = df[["Draw/NC","R","B"]].idxmax(1)
    return df.drop(["R","B","Draw/NC","Winner"], axis=1)

def decode_weight_class(df, cname="Fight_type"):
    df[cname] = df[cname].str.replace("UFC ", "")
    df[cname] = df[cname].str.replace(" Bout", "")
    df[cname] = df[cname].str.replace(" Title", "")
    df[cname] = df[cname].str.replace("Interim ", "")
    keep = ["Bantamweight", "Middleweight", "Heavyweight", "Women's Strawweight",
            "Women's Bantamweight", "Lightweight", "Welterweight", "Flyweight", 
            "Light Heavyweight", "Featherweight", "Women's Flyweight", 
            "Women's Featherweight"]
    df = df[df[cname].isin(keep)]
    df["F"] = df[cname].str.contains("Women")
    df["M"] = df["F"] == False
    df["gender"] = df[["M","F"]].idxmax(1)
    df[cname] = df[cname].str.replace("Women's ", "")
    df[cname] = df[cname].replace(WEIGHT_CLASSES_LBS)
    df = df.drop(["M","F"], axis=1)
    return df.rename(columns={cname: "weight_class"})

def get_fids_and_ages(bouts, fighters):
    red_ages = []
    blue_ages = []
    for _, row in bouts.iterrows():
        # get fighter ids
        row["R_fighter"] = fighters[fighters["fighter_name"] == row["R_fighter"]]["fighter_id"].values[0]
        row["B_fighter"] = fighters[fighters["fighter_name"] == row["B_fighter"]]["fighter_id"].values[0]
        # get figher ages
        bout_year = row["bout_year"]
        red_birth_year = fighters[fighters["fighter_id"] == row["R_fighter"]]["birth_year"].values[0]
        blue_birth_year = fighters[fighters["fighter_id"] == row["B_fighter"]]["birth_year"].values[0]
        red_ages.append((bout_year-red_birth_year))
        blue_ages.append((bout_year-blue_birth_year))
    bouts["red_fighter_age"] = red_ages
    bouts["blue_fighter_age"] = blue_ages
    return bouts.rename(columns={"R_fighter": "red_fid", "B_fighter": "blue_fid"})

def calculate_records(df, bouts, cname="fighter_name"):
    df["wins"] = df.apply(lambda x: bouts[bouts["Winner"] == x[cname]].shape[0], axis=1)
    df["losses"] = df.apply(lambda x: bouts[bouts["Winner"] == x[cname]].shape[0], axis=1)
    df["draw/nc"] = df.apply(lambda x: bouts[((bouts["R_fighter"] == x[cname]) |
                                             (bouts["B_fighter"] == x[cname])) &
                                             (bouts["Winner"] != x[cname])].shape[0], axis=1)
    df["bouts"] = df.apply(lambda x: bouts[(bouts["R_fighter"] == x[cname]) |
                                           (bouts["B_fighter"] == x[cname])].shape[0], axis=1)
    return df

if __name__ == "__main__":
    # data from: https://www.kaggle.com/datasets/rajeevw/ufcdata?select=preprocessed_data.csv
    fighters_fname = "fighters.csv"
    bouts_fname = "bouts.csv"
    if not isfile(fighters_fname):
        fighters = pd.read_csv("raw_fighter_details.csv")
        # show_col_dtypes(fighters)
        fighters["Stance"] = fighters["Stance"].fillna("Unknown")
        fighters["Stance"] = fighters["Stance"].astype(str)
        # stance = pd.get_dummies(fighters["Stance"])
        # stance.columns = ["open_stance", "orthodox_stance", "sideways_stance",
        #                "southpaw_stance", "switch_stance", "unkown_stance"]
        # stance = stance.astype(np.float32)
        # fighters.drop(["Stance"], axis=1, inplace=True)
        # fighters = fighters.join(stance)
        fighters["Weight"] = fighters["Weight"].str.split(" ", n=1, expand=True)[0].astype(np.float32)
        fighters["Height"] = height_to_inches(fighters)
        rem_last = ["Reach", "Str_Acc", "Str_Def", "TD_Acc", "TD_Def"]
        for col in rem_last:
            fighters[col] = remove_last(fighters, col).astype(np.float32)
        fighters = decode_date(fighters, prefix="birth_")
        fighters.insert(0, "fighter_id", range(fighters.shape[0]))
        # print(fighters)
        fighters.to_csv(fighters_fname, index=False)
    else:
        fighters = pd.read_csv(fighters_fname)
    if not isfile(bouts_fname):
        bouts = pd.read_csv("raw_total_fight_data.csv", sep=';')

        keep_cols = ["R_fighter", "B_fighter", "Format", "Referee", "date", 
                     "location", "Fight_type", "Winner", "win_by", "last_round"]

        bouts = bouts[keep_cols]
        if not set(["wins", "losses", "draw/nc", "bouts"]).issubset(set(fighters.columns)):
            fighters = calculate_records(fighters, bouts)
            # remove fighters with 0 bouts
            fighters = fighters[fighters["bouts"] > 0]
            # fighters.to_csv(fighters_fname, axis=False)
        print(fighters)
        bouts = decode_format(bouts)
        bouts = decode_date(bouts, cname="date", prefix="bout_")
        bouts = decode_location(bouts)
        bouts["title_bout"] = bouts["Fight_type"].str.contains("Title").astype(np.float32)
        bouts = decode_winner(bouts)
        bouts = decode_weight_class(bouts)
    #     cols = ['gender', 'red_fid', 'weight_class', 'rounds', 'blue_fid', 'Referee',
    #             'bout_month', 'bout_day', 'bout_year', 'city', 'country', 'title_bout',
    #             'winner', 'win_by', 'last_round']
    #     bouts = bouts[cols]
    #     bouts.to_csv(bouts_fname)
    # else:
    #     bouts = pd.read_csv(bouts_fname)