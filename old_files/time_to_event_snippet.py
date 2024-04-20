import numpy as np
import pandas as pd
from sksurv.util import Surv # pip install scikit-survival
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest

DIR_BASE      = "/mnt/c/Users/u0131222/PhD/"
DIR_MSOAC     = DIR_BASE + "Projects/MSOAC/Data/"
DIR_PROCESSED = DIR_BASE + "Supervision/2023_MultiTaskMS_Lennert-Sofia/thesis_repo/new_data/"

# Your preprocessed dataframe
df = pd.read_csv(DIR_PROCESSED + "merged_data.csv")
df = df.set_index("USUBJID")

# A preprocessed dataframe with EDSS scores (since I need some stuff from it)
# - Load the QS table and extract only the EDSS scores from it
edss = pd.read_sas(DIR_MSOAC + "qs.xpt", format="xport", encoding="utf-8")
edss = edss.loc[edss.QSCAT == "EDSS"]
edss = edss[["USUBJID", "QSORRES", "VISIT", "QSDY"]] # columns of interest
# - Convert the scores to floats
edss.loc[edss.QSORRES == "", "QSORRES"] = np.nan
edss.QSORRES = edss.QSORRES.astype(float)
# - Fix the date thing (starts counting from 1)
edss.loc[edss.QSDY > 0, "QSDY"] = edss.loc[edss.QSDY > 0, "QSDY"] - 1
edss.QSDY = edss.QSDY / 365.25 # convert to years
edss = edss.sort_values(by=["USUBJID", "QSDY"])

# Defining the survival target
# - I assume df["SMSTDY"] is the time to first confirmed relapse if it exists
y = pd.DataFrame(np.nan, index=df.index, columns=["cens", "time"])
y["time"] = df["SMSTDY"] / 365.25
y["cens"] = (~y["time"].isna()) # censoring status: 0 = censored, 1 = observed
# - If it does not exist, I put a time to censoring at the last EDSS visit
to_fill = y.index[(~y["cens"]).values.flatten()]
y.loc[to_fill,"time"] = edss.groupby("USUBJID").QSDY.max().loc[to_fill]
y = y.loc[y.time > 0] # TODO 78 anomaly cases with non-positive time to event
y = y.dropna() # NOTE Still some missing values, drop for now
y_surv = Surv().from_arrays(y.cens, y.time) # Convert to recarray (required for sksurv)

# Defining the input data
X = df[
    ["AGE","SEX","RACE","MHDIAGN","CARDIO","URINARY","MUSCKELET","FATIGUE"] + 
    [col for col in df.columns if col.endswith("before")]
].copy()
X = X.loc[y.index]

# Modeling
rsf = RandomSurvivalForest(n_jobs=-1, random_state=42)
X_train = X # TODO use the CV split here + first preprocess (impute missing values etc)
X_test  = X # TODO use the CV split here + first preprocess (impute missing values etc)
y_train = y_surv # TODO use the CV split here
y_test  = y_surv # TODO use the CV split here
rsf.fit(X_train, y_train)
y_pred = rsf.predict(X_test)
hci = concordance_index_censored(y_test["event"], y_test["time"], y_pred)[0]
print(f"Harrell's C-index = {hci:.3f}")
