import pandas as pd
import numpy as np

df = pd.read_csv("2022.csv")
df = df.drop(columns=["REG.", "ESTD"], errors="ignore")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

district_map = {
    "EG": "EAST GODAVARI",
    "GTR": "GUNTUR",
    "KRI": "KRISHNA",
    "PKS": "PRAKASAM",
    "SKL": "SRIKAKULAM",
    "VSP": "VISAKHAPATNAM",
    "VZM": "VIZIANAGARAM",
    "WG": "WEST GODAVARI",
    "ATP": "ANANTAPUR",
    "CTR": "CHITTOOR",
    "KDP": "KADAPA",
    "KNL": "KURNOOL",
    "NLR": "NELLORE"
}

df["DISTRICT"] = df["DISTRICT"].replace(district_map)

df = df.replace([' ', '--', 'NA', '-', '_'], np.nan)

# ===============================
# CUT-OFF CLEANING
# ===============================

cutoff_pairs = [
("OC_BOYS","OC_GIRLS"),
("OC_EWS_BOYS","OC_EWS_GIRLS"),
("BCA_BOYS","BCA_GIRLS"),
("BCB_BOYS","BCB_GIRLS"),
("BCC_BOYS","BCC_GIRLS"),
("BCD_BOYS","BCD_GIRLS"),
("BCE_BOYS","BCE_GIRLS"),
("SC_BOYS","SC_GIRLS"),
("ST_BOYS","ST_GIRLS")
]

# Fill within same category
for boys, girls in cutoff_pairs:
    if boys in df.columns and girls in df.columns:
        df[boys] = df[boys].fillna(df[girls])
        df[girls] = df[girls].fillna(df[boys])

# Convert to numeric
all_cutoff_cols = [c for pair in cutoff_pairs for c in pair]
df[all_cutoff_cols] = df[all_cutoff_cols].apply(pd.to_numeric, errors='coerce')

# Replace remaining missing
df[all_cutoff_cols] = df[all_cutoff_cols].fillna(999999)

# ===============================
# COLLEGE FEE CLEANING
# ===============================

df["COLLEGE_FEE"] = pd.to_numeric(df["COLLEGE_FEE"], errors="coerce")
df["COLLEGE_FEE"] = df.groupby("INST_CODE")["COLLEGE_FEE"].transform(
    lambda x: x.fillna(x.mean())
)
# ===============================
# CLEAN COLLEGE NAMES
# ===============================

df["INST_NAME"] = df["INST_NAME"].astype(str)

# Remove newline characters
df["INST_NAME"] = df["INST_NAME"].str.replace('\n', ' ', regex=True)

# Remove extra spaces
df["INST_NAME"] = df["INST_NAME"].str.strip()

# Replace multiple spaces with single space
df["INST_NAME"] = df["INST_NAME"].str.replace(r'\s+', ' ', regex=True)

# Optional: convert to uppercase for consistency
df["INST_NAME"] = df["INST_NAME"].str.upper()

# Save cleaned
df.to_csv("cleaned_2022.csv", index=False)
print("2022 cleaned correctly ✅")

# ===============================
# LONG FORMAT CONVERSION
# ===============================

df_long = df.melt(
    id_vars=[
        "INST_CODE", "INST_NAME", "BRANCH_CODE",
        "DISTRICT","COED","AFFLIATED", "COLLEGE_FEE"
    ],
    value_vars=all_cutoff_cols,
    var_name="CATEGORY_GENDER",
    value_name="CUTOFF"
)

# ✅ FIXED CATEGORY + GENDER SPLIT
df_long["CATEGORY"] = df_long["CATEGORY_GENDER"].apply(
    lambda x: "_".join(x.split("_")[:-1])
)
df_long["GENDER"] = df_long["CATEGORY_GENDER"].apply(
    lambda x: x.split("_")[-1]
)

# Add year
df_long["YEAR"] = 2022

#  Remove invalid cutoff rows
df_long = df_long[df_long["CUTOFF"] != 999999]

# Drop helper column
df_long = df_long.drop(columns=["CATEGORY_GENDER"])

# Reset index
df_long = df_long.reset_index(drop=True)

# Save final long dataset
df_long.to_csv("long_2022.csv", index=False)

print("Long format 2022 dataset created ✅")