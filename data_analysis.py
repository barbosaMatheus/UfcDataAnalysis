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
    df["none"] = ((df["Winner"] != df["R_fighter"]) & 
                        (df["Winner"] != df["B_fighter"]))
    df["red"] = df["Winner"] == df["R_fighter"]
    df["blue"] = df["Winner"] == df["B_fighter"]
    df["winner"] = df[["none","red","blue"]].idxmax(1)
    return df.drop(["red","blue","none","Winner"], axis=1)

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
    df["bouts"] = df.apply(lambda x: bouts[(bouts["R_fighter"] == x[cname]) |
                                           (bouts["B_fighter"] == x[cname])].shape[0], axis=1)
    df["wins"] = df.apply(lambda x: bouts[bouts["Winner"] == x[cname]].shape[0], axis=1)
    df["losses"] = df.apply(lambda x: bouts[((bouts["R_fighter"] == x[cname]) &
                                             (bouts["Winner"] == bouts["B_fighter"])) |
                                             ((bouts["B_fighter"] == x[cname]) &
                                             (bouts["Winner"] == bouts["R_fighter"]))].shape[0], 
                                             axis=1)
    df["draw_nc"] = df["bouts"] - (df["wins"] + df["losses"])
    return df

def get_fighter_ids(df, fighters):
    f = fighters.copy()
    df["red_fid"] = df["R_fighter"].map(f.set_index("fighter_name")["fighter_id"])
    df["blue_fid"] = df["B_fighter"].map(f.set_index("fighter_name")["fighter_id"])
    return df

def get_numeric_data_report(df):
    num_df = df.select_dtypes("number")
    rows = num_df.shape[0]
    qual_rep = num_df.describe()
    missing = {"n": [], "p": []}
    unique = {"n": [], "p": []}
    for col in num_df.columns.values:
        n_miss = num_df[col].isna().sum()
        p_miss = float(n_miss) / rows
        missing["n"].append(n_miss)
        missing["p"].append(p_miss)
        n_unique = num_df[col].nunique()
        p_unique = float(n_unique) / rows
        unique["n"].append(n_unique)
        unique["p"].append(p_unique)
    qual_rep.loc["n_missing"] = missing["n"]
    qual_rep.loc["missing_pct"] = missing["p"]
    qual_rep.loc["n_unique"] = unique["n"]
    qual_rep.loc["unique_pct"] = unique["p"]
    return qual_rep

def get_object_data_report(df):
    obj_df = df.select_dtypes("object")
    qual_rep = pd.DataFrame(columns=obj_df.columns.values)
    rows = obj_df.shape[0]
    modes1 = {"v": [], "f": []}
    modes2 = {"v": [], "f": []}
    missing = {"n": [], "p": []}
    unique = {"n": [], "p": []}
    antimodes = {"v": [], "f": []}
    for col in obj_df.columns.values:
        counts = obj_df[col].value_counts()
        mode1 = counts.index[0]
        mode2 = counts.index[1]
        antimode = counts.index[-1]
        m1_freq = float(counts.iloc[0]) / rows
        m2_freq = float(counts.iloc[1]) / rows
        am_freq = float(counts.iloc[-1]) / rows
        modes1["v"].append(mode1)
        modes1["f"].append(m1_freq)
        modes2["v"].append(mode2)
        modes2["f"].append(m2_freq)
        antimodes["v"].append(antimode)
        antimodes["f"].append(am_freq)
        n_miss = obj_df[col].isna().sum()
        p_miss = float(n_miss) / rows
        missing["n"].append(n_miss)
        missing["p"].append(p_miss)
        n_unique = obj_df[col].nunique()
        p_unique = float(n_unique) / rows
        unique["n"].append(n_unique)
        unique["p"].append(p_unique)
    qual_rep.loc["1-mode"] = modes1["v"]
    qual_rep.loc["1-mode_freq"] = modes1["f"]
    qual_rep.loc["2-mode"] = modes2["v"]
    qual_rep.loc["2-mode_freq"] = modes2["f"]
    qual_rep.loc["antimode"] = antimodes["v"]
    qual_rep.loc["a-mode_freq"] = antimodes["f"]
    qual_rep.loc["n_missing"] = missing["n"]
    qual_rep.loc["missing_pct"] = missing["p"]
    qual_rep.loc["n_unique"] = unique["n"]
    qual_rep.loc["unique_pct"] = unique["p"]
    return qual_rep

if __name__ == "__main__":
    # data from: https://www.kaggle.com/datasets/rajeevw/ufcdata
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
        if not set(["wins", "losses", "draw_nc", "bouts"]).issubset(set(fighters.columns)):
            fighters = calculate_records(fighters, bouts)
            # remove fighters with 0 bouts
            fighters = fighters[fighters["bouts"] > 0]
            fighters.to_csv(fighters_fname, index=False)
        bouts = decode_format(bouts)
        bouts = decode_date(bouts, cname="date", prefix="bout_")
        bouts = decode_location(bouts)
        bouts["is_title_bout"] = bouts["Fight_type"].str.contains("Title").astype(np.float32)
        bouts = decode_winner(bouts)
        bouts = decode_weight_class(bouts)
        bouts = get_fighter_ids(bouts, fighters)
        bouts = bouts.rename(columns={"R_fighter": "r_fighter", "B_fighter": "b_fighter"})
        cols = ["gender", "weight_class", "red_fid", "r_fighter", "blue_fid", "b_fighter", 
                "rounds", "Referee", "bout_month", "bout_day", "bout_year", "city", 
                "country", "is_title_bout", "winner", "win_by", "last_round"]
        bouts = bouts[cols]
        bouts.to_csv(bouts_fname, index=False)
    else:
        bouts = pd.read_csv(bouts_fname)

    # fighters_num_rep = get_numeric_data_report(fighters)
    # fighters_obj_rep = get_object_data_report(fighters)
    # print(fighters_num_rep)
    # print(fighters_obj_rep)
    # fighters_num_rep.to_csv("fighters_numeric_report.csv", index_label="Metric")
    # fighters_obj_rep.to_csv("fighters_object_report.csv", index_label="Metric")
    # bouts_num_rep = get_numeric_data_report(bouts)
    # bouts_obj_rep = get_object_data_report(bouts)
    # print(bouts_num_rep)
    # print(bouts_obj_rep)
    # bouts_num_rep.to_csv("bouts_numeric_report.csv", index_label="Metric")
    # bouts_obj_rep.to_csv("bouts_object_report.csv", index_label="Metric")

    """
    Imputation and cleanup notes:
    - drop 32 bouts w/ missing refs
    - drop fighers w/ missing height
    - drop fighters w/ missing weight
    - analysis for best predictors of reach
    - drop fighters w/ missing birthdays
    """
    bouts_preprocessed = bouts.copy()
    bouts_preprocessed = bouts_preprocessed.dropna(subset=["Referee"])

    fighters_preprocessed = fighters.copy()
    fighters_preprocessed = fighters_preprocessed.dropna(subset=["Height", "Weight", "birth_month",
                                                                 "birth_day", "birth_year"])
    corr = fighters_preprocessed.select_dtypes("number").corr()
    print(corr["Reach"])

    reach_preds = ["Height", "Weight"]