import streamlit as st
import pandas as pd
import joblib

# ======================
# LOAD MODEL & DATA
# ======================
model = joblib.load("cutoff_predictor_xgb.pkl")
encoders = joblib.load("cutoff_encoders.pkl")
base_df = pd.read_csv("cleaned_dataset.csv")

st.set_page_config(page_title="EAPCET College Predictor", layout="wide")

# ======================
# PROBABILITY LOGIC
# ======================
def admission_probability(user_rank, cutoff):
    diff = cutoff - user_rank
    if diff < -15000: return 2
    if diff < -5000:  return 10
    if diff < 0:      return 30
    if diff < 5000:   return 55
    if diff < 15000:  return 75
    return 90

# ======================
# PREDICT FUNCTION
# ======================
def predict_colleges(rank, gender, category, branch_code, year=2025):

    df = base_df.copy()

    if branch_code:
        df = df[df["BRANCH_CODE"] == branch_code]

    df["YEAR"] = year
    df["GENDER"] = gender
    df["CATEGORY"] = category

    if gender == "MALE":
        df = df[df["COED"] != "GIRLS"]

    FEATURES = ["YEAR","GENDER","CATEGORY","BRANCH_CODE","DIST","COED","TYPE"]

    for col, le in encoders.items():
        df[col] = df[col].map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    X = df[FEATURES]
    df["PREDICTED_CUTOFF"] = model.predict(X).astype(int)
    df["PROBABILITY"] = df["PREDICTED_CUTOFF"].apply(
        lambda x: admission_probability(rank, x)
    )

    df = df.drop_duplicates(["INST_CODE","BRANCH_CODE"])

    for col, le in encoders.items():
        df[col] = df[col].apply(
            lambda x: le.inverse_transform([int(x)])[0] if x != -1 else "UNKNOWN"
        )

    return df

# ======================
# UI
# ======================
st.title("🎓 AP EAPCET College Predictor")

with st.sidebar:
    st.header("🔢 Student Details")

    rank = st.number_input("EAPCET Rank", min_value=1, max_value=200000, value=25000)
    gender = st.selectbox("Gender", ["MALE", "FEMALE"])
    category = st.selectbox("Category", ["OC","BC","SC","ST"])
    branch_code = st.selectbox(
        "Branch Code",
        [""] + sorted(base_df["BRANCH_CODE"].unique())
    )
    branch_code = branch_code if branch_code else None

    predict_btn = st.button("🔍 Predict Colleges")

# ======================
# PREDICT
# ======================
if predict_btn:

    result = predict_colleges(rank, gender, category, branch_code)

    st.subheader("🎯 Filter Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        branches = st.multiselect(
            "Select Branch",
            sorted(result["BRANCH_NAME"].unique())
        )

    with col2:
        districts = st.multiselect(
            "Select District",
            sorted(result["DIST"].unique())
        )

    with col3:
        college_type = st.selectbox(
            "College Type",
            ["ALL","COED","GIRLS"]
        )

    with col4:
        min_prob = st.slider(
            "Minimum Probability %",
            0, 100, 60
        )

    filtered = result.copy()

    if branches:
        filtered = filtered[filtered["BRANCH_NAME"].isin(branches)]

    if districts:
        filtered = filtered[filtered["DIST"].isin(districts)]

    if college_type != "ALL":
        filtered = filtered[filtered["COED"] == college_type]

    filtered = filtered[filtered["PROBABILITY"] >= min_prob]

    filtered["PROBABILITY_%"] = filtered["PROBABILITY"].astype(str) + "%"

    st.subheader("📊 Predicted Colleges")

    if filtered.empty:
        st.warning("No colleges found. Try reducing filters.")
    else:
        st.dataframe(
            filtered[
                [
                    "INST_CODE","INST_NAME","TYPE",
                    "BRANCH_NAME","DIST","PLACE",
                    "COED","PREDICTED_CUTOFF","PROBABILITY_%"
                ]
            ].sort_values("PROBABILITY", ascending=False),
            use_container_width=True
        )
