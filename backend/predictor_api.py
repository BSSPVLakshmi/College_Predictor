import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("final_dataset.csv")

# Clean text
df["CATEGORY"] = df["CATEGORY"].str.strip().str.upper()
df["GENDER"] = df["GENDER"].str.strip().str.upper()
df["INST_NAME"] = df["INST_NAME"].str.strip()
df["DISTRICT"] = df["DISTRICT"].str.strip().str.upper()
df["BRANCH_CODE"] = df["BRANCH_CODE"].str.strip().str.upper()

# =========================
# 2. ENCODING
# =========================
le_category = LabelEncoder()
le_gender = LabelEncoder()
le_college = LabelEncoder()
le_branch = LabelEncoder()

df["CAT_ENC"] = le_category.fit_transform(df["CATEGORY"])
df["GEN_ENC"] = le_gender.fit_transform(df["GENDER"])
df["COL_ENC"] = le_college.fit_transform(df["INST_NAME"])
df["BR_ENC"] = le_branch.fit_transform(df["BRANCH_CODE"])

# =========================
# 3. FEATURES
# =========================
X = df[["CAT_ENC", "GEN_ENC", "COL_ENC", "BR_ENC", "YEAR"]]
y = df["CUTOFF"]

# =========================
# 4. MODEL
# =========================
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)


def _normalize_gender(gender: str) -> str:
    gender = str(gender).strip().upper()
    if gender == "MALE":
        return "BOYS"
    if gender == "FEMALE":
        return "GIRLS"
    return gender


def _parse_selection(value):
    """
    Accepts:
    - "ALL"
    - "east godavari,west godavari"
    - ["east godavari", "west godavari"]
    - None
    """
    if value is None:
        return None

    if isinstance(value, list):
        items = [str(v).strip().upper() for v in value if str(v).strip()]
        return None if not items or "ALL" in items else items

    text = str(value).strip().upper()
    if text == "" or text == "ALL":
        return None

    return [item.strip() for item in text.split(",") if item.strip()]


def get_available_districts():
    return sorted(df["DISTRICT"].dropna().unique().tolist())


def get_available_branches():
    return sorted(df["BRANCH_CODE"].dropna().unique().tolist())


def predict_colleges(
    rank,
    category,
    gender,
    districts="ALL",
    branches="ALL",
    sort_choice="1",
    show_only_safe_target="no",
):
    """
    Returns a list of dictionaries:
    [
        {
            "College": "...",
            "Branch": "...",
            "District": "...",
            "Predicted_Cutoff": 12345,
            "Chance": "SAFE",
            "Confidence": 91.2
        },
        ...
    ]
    """

    # -------------------------
    # Normalize inputs
    # -------------------------
    rank = int(rank)
    category = str(category).strip().upper()
    gender = _normalize_gender(gender)
    sort_choice = str(sort_choice).strip()
    show_only_safe_target = str(show_only_safe_target).strip().lower()

    selected_districts = _parse_selection(districts)
    selected_branches = _parse_selection(branches)

    # -------------------------
    # Validate inputs
    # -------------------------
    if category not in le_category.classes_:
        raise ValueError(f"Invalid category. Allowed: {list(le_category.classes_)}")

    if gender not in le_gender.classes_:
        raise ValueError(f"Invalid gender. Allowed: {list(le_gender.classes_)}")

    cat_enc = le_category.transform([category])[0]
    gen_enc = le_gender.transform([gender])[0]

    # -------------------------
    # Prediction loop
    # -------------------------
    results = []
    latest_year = df["YEAR"].max()

    unique_college_branch = df[["INST_NAME", "BRANCH_CODE"]].drop_duplicates()

    for _, row in unique_college_branch.iterrows():
        col = row["INST_NAME"]
        br = row["BRANCH_CODE"]

        district_rows = df[df["INST_NAME"] == col]
        if district_rows.empty:
            continue

        district = district_rows["DISTRICT"].iloc[0]

        # Filter 1: District
        if selected_districts and district not in selected_districts:
            continue

        # Filter 2: Branch
        if selected_branches and br not in selected_branches:
            continue

        # Filter 3: Women college restriction for BOYS
        if gender == "BOYS" and "WOMEN" in col.upper():
            continue

        col_enc = le_college.transform([col])[0]
        br_enc = le_branch.transform([br])[0]

        input_df = pd.DataFrame(
            [
                {
                    "CAT_ENC": cat_enc,
                    "GEN_ENC": gen_enc,
                    "COL_ENC": col_enc,
                    "BR_ENC": br_enc,
                    "YEAR": latest_year,
                }
            ]
        )

        # Hybrid prediction
        ml_cutoff = model.predict(input_df)[0]

        last_3_years = df[
            (df["INST_NAME"] == col)
            & (df["BRANCH_CODE"] == br)
            & (df["CATEGORY"] == category)
            & (df["GENDER"] == gender)
        ].sort_values(by="YEAR", ascending=False).head(3)

        avg_cutoff = last_3_years["CUTOFF"].mean()

        if pd.notna(avg_cutoff):
            pred_cutoff = (0.8 * avg_cutoff) + (0.2 * ml_cutoff)
        else:
            pred_cutoff = ml_cutoff

        # Calibration
        pred_cutoff = pred_cutoff * 0.95

        # Realistic filtering
        if rank > pred_cutoff + 1500:
            continue

        # Chance labels
        if rank <= pred_cutoff - 1500:
            chance = "SAFE"
        elif rank <= pred_cutoff:
            chance = "TARGET"
        else:
            chance = "DREAM"

        confidence = round(100 - abs(rank - pred_cutoff) / pred_cutoff * 100, 2)

        results.append(
            {
                "College": col,
                "Branch": br,
                "District": district,
                "Predicted_Cutoff": int(pred_cutoff),
                "Chance": chance,
                "Confidence": confidence,
            }
        )

    if not results:
        return []

    result_df = pd.DataFrame(results)

    # Remove duplicates
    result_df = result_df.sort_values(by="Predicted_Cutoff")
    result_df = result_df.drop_duplicates(subset=["College", "Branch"], keep="first")

    # Sorting
    chance_order = {"SAFE": 1, "TARGET": 2, "DREAM": 3}

    if sort_choice == "1":
        result_df["sort"] = result_df["Chance"].map(chance_order)
        result_df = result_df.sort_values(by=["sort", "Predicted_Cutoff"])
    else:
        result_df = result_df.sort_values(by=["Predicted_Cutoff"])

    # Optional post-filter
    if show_only_safe_target == "yes":
        result_df = result_df[result_df["Chance"] != "DREAM"]

    return result_df.head(20).to_dict(orient="records")


if __name__ == "__main__":
    print("Available Districts:", get_available_districts())
    print("Available Branches:", get_available_branches())

    rank = int(input("Enter your rank: "))
    category = input("Enter category: ").strip().upper()
    gender = input("Enter gender: ").strip().upper()
    districts = input("Enter districts (comma-separated or ALL): ").strip()
    branches = input("Enter branches (comma-separated or ALL): ").strip()
    sort_choice = input("Sort by (1: Chance, 2: Cutoff): ").strip()
    show_only_safe_target = input("Show only SAFE/TARGET? (yes/no): ").strip().lower()

    try:
        results = predict_colleges(
            rank=rank,
            category=category,
            gender=gender,
            districts=districts,
            branches=branches,
            sort_choice=sort_choice,
            show_only_safe_target=show_only_safe_target,
        )

        if not results:
            print("\n⚠️ No colleges found. Try different filters.")
        else:
            print("\n🎯 FINAL COLLEGE PREDICTIONS:\n")
            print(pd.DataFrame(results))

    except ValueError as e:
        print(f"\n❌ {e}")