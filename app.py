import streamlit as st
import pandas as pd
import joblib

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="AP EAPCET College Predictor",
    layout="wide"
)

st.title("🎓 AP EAPCET College Predictor")
st.caption("ML-based prediction using historical cutoff trends")

# ======================
# LOAD MODEL & DATA
# ======================
@st.cache_resource
def load_model():
    model = joblib.load("cutoff_predictor_xgb.pkl")
    encoders = joblib.load("cutoff_encoders.pkl")
    return model, encoders

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_dataset.csv")

model, encoders = load_model()
base_df = load_data()

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
# PREDICTION FUNCTION
# ======================
def predict_colleges(rank, gender, category, branch_code, year):

    df = base_df.copy()

    if branch_code != "ALL":
        df = df[df["BRANCH_CODE"] == branch_code]

    if gender == "MALE":
        df = df[df["COED"] != "GIRLS"]

    df["YEAR"] = year
    df["GENDER"] = gender
    df["CATEGORY"] = category

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
    df["PROBABILITY_%"] = df["PROBABILITY"].astype(str) + "%"

    df = df.drop_duplicates(["INST_CODE","BRANCH_CODE"])

    # Decode
    for col, le in encoders.items():
        df[col] = df[col].apply(
            lambda x: le.inverse_transform([int(x)])[0] if x != -1 else "UNKNOWN"
        )

    return df

# ======================
# USER INPUTS
# ======================
with st.sidebar:
    st.header("📝 Enter Details")

    rank = st.number_input("Rank", min_value=1, max_value=200000, value=50000)
    gender = st.selectbox("Gender", ["FEMALE","MALE"])
    category = st.selectbox("Category", ["OC","BC","SC","ST"])

    branch_options = ["ALL"] + sorted(base_df["BRANCH_CODE"].unique())
    branch_code = st.selectbox("Branch", branch_options)

    year = st.selectbox("Counselling Year", [2025])

    predict_btn = st.button("🔮 Predict Colleges")

# ======================
# PREDICT & FILTER
# ======================
if predict_btn:

    predicted_df = predict_colleges(
        rank, gender, category, branch_code, year
    )

    # ------------------
    # FILTER OPTIONS (FROM UNFILTERED predicted_df)
    # ------------------
    st.subheader("🎯 Apply Filters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        branches = st.multiselect(
            "Branch Name",
            sorted(predicted_df["BRANCH_NAME"].unique())
        )

    with col2:
        districts = st.multiselect(
            "District",
            sorted(predicted_df["DIST"].unique())
        )

    with col3:
        college_type = st.selectbox(
            "College Type",
            ["ALL","COED","GIRLS"]
        )

    with col4:
        min_prob = st.slider(
            "Minimum Probability %",
            0, 100, 50
        )

    # ------------------
    # APPLY FILTERS
    # ------------------
    filtered_df = predicted_df.copy()

    if branches:
        filtered_df = filtered_df[
            filtered_df["BRANCH_NAME"].isin(branches)
        ]

    if districts:
        filtered_df = filtered_df[
            filtered_df["DIST"].isin(districts)
        ]

    if college_type != "ALL":
        filtered_df = filtered_df[
            filtered_df["COED"] == college_type
        ]

    filtered_df = filtered_df[
        filtered_df["PROBABILITY"] >= min_prob
    ]

    # ------------------
    # DISPLAY RESULT
    # ------------------
    st.subheader("🏫 Predicted Colleges")

    if filtered_df.empty:
        st.warning("No colleges match the selected filters. Try relaxing filters.")
    else:
        st.dataframe(
            filtered_df[
                [
                    "INST_CODE","INST_NAME","TYPE",
                    "BRANCH_NAME","DIST","PLACE",
                    "COED","PREDICTED_CUTOFF","PROBABILITY_%"
                ]
            ].sort_values("PROBABILITY", ascending=False),
            use_container_width=True
        )


# import streamlit as st
# import pandas as pd
# import joblib

# # ======================
# # PAGE CONFIG
# # ======================
# st.set_page_config(
#     page_title="AP EAPCET College Predictor",
#     layout="wide"
# )

# st.title("🎓 AP EAPCET College Predictor")
# st.caption("ML-based prediction using historical cutoff trends")

# # ======================
# # LOAD MODEL & DATA
# # ======================
# @st.cache_resource
# def load_model():
#     model = joblib.load("cutoff_predictor_xgb.pkl")
#     encoders = joblib.load("cutoff_encoders.pkl")
#     return model, encoders

# @st.cache_data
# def load_data():
#     return pd.read_csv("cleaned_dataset.csv")

# model, encoders = load_model()
# base_df = load_data()

# # ======================
# # PROBABILITY LOGIC
# # ======================
# def admission_probability(user_rank, cutoff):
#     diff = cutoff - user_rank
#     if diff < -15000: return 2
#     if diff < -5000:  return 10
#     if diff < 0:      return 30
#     if diff < 5000:   return 55
#     if diff < 15000:  return 75
#     return 90

# # ======================
# # PREDICT FUNCTION
# # ======================
# def predict_colleges(rank, gender, category, branch_code=None, year=2025):

#     df = base_df.copy()

#     if branch_code:
#         df = df[df["BRANCH_CODE"] == branch_code]

#     df["YEAR"] = year
#     df["GENDER"] = gender
#     df["CATEGORY"] = category

#     if gender == "MALE":
#         df = df[df["COED"] != "GIRLS"]

#     FEATURES = [
#         "YEAR","GENDER","CATEGORY",
#         "BRANCH_CODE","DIST","COED","TYPE"
#     ]

#     for col, le in encoders.items():
#         df[col] = df[col].map(
#             lambda x: le.transform([x])[0] if x in le.classes_ else -1
#         )

#     X = df[FEATURES]

#     df["PREDICTED_CUTOFF"] = model.predict(X).astype(int)
#     df["PROBABILITY"] = df["PREDICTED_CUTOFF"].apply(
#         lambda x: admission_probability(rank, x)
#     )

#     df["PROBABILITY_%"] = df["PROBABILITY"].astype(str) + "%"


#     df = df.drop_duplicates(["INST_CODE","BRANCH_CODE"])

#     # Decode categorical columns
#     for col, le in encoders.items():
#         df[col] = df[col].apply(
#             lambda x: le.inverse_transform([int(x)])[0] if x != -1 else "UNKNOWN"
#         )

#     df["PROBABILITY_%"] = df["PROBABILITY"].astype(int).astype(str) + "%"

#     return df.sort_values(
#         "PROBABILITY", ascending=False
#     ).reset_index(drop=True)

# # ======================
# # USER INPUT UI
# # ======================
# st.sidebar.header("🧑 Candidate Details")

# rank = st.sidebar.number_input(
#     "Rank", min_value=1, max_value=200000, step=1
# )

# gender = st.sidebar.selectbox(
#     "Gender", ["MALE", "FEMALE"]
# )

# category = st.sidebar.selectbox(
#     "Category", ["OC", "BC", "SC", "ST"]
# )

# branch_code = st.sidebar.text_input(
#     "Branch Code (optional)", placeholder="CSE, IT, ECE"
# ).upper()

# branch_code = branch_code if branch_code else None

# predict_btn = st.sidebar.button("🔍 Predict Colleges")

# # ======================
# # PREDICTION OUTPUT
# # ======================
# if predict_btn:

#     with st.spinner("Predicting colleges..."):
#         result = predict_colleges(
#             rank, gender, category, branch_code
#         )

#     if result.empty:
#         st.warning("No colleges found.")
#         st.stop()

#     st.subheader("📊 Predicted Colleges")

#     # ======================
#     # FILTERS
#     # ======================
#     st.markdown("### 🔎 Filters")

#     col1, col2, col3, col4 = st.columns(4)

#     with col1:
#         selected_branches = st.multiselect(
#             "Branch",
#             sorted(result["BRANCH_NAME"].unique())
#         )

#     with col2:
#         selected_districts = st.multiselect(
#             "District",
#             sorted(result["DIST"].unique())
#         )

#     with col3:
#         college_type = st.selectbox(
#             "College Type",
#             ["ALL", "COED", "GIRLS"]
#         )

#     with col4:
#         min_prob = st.slider(
#             "Minimum Probability %",
#             0, 100, 0, step=5
#         )

#     filtered = result.copy()

# # Branch filter
#     if selected_branches:
#         filtered = filtered[
#             filtered["BRANCH_NAME"].isin(selected_branches)
#         ]

# # District filter
#     if selected_districts:
#         filtered = filtered[
#             filtered["DIST"].isin(selected_districts)
#         ]

# # College type filter
#     if college_type != "ALL":
#         filtered = filtered[
#             filtered["COED"] == college_type
#         ]

# # Probability filter (NUMERIC)
#     if min_prob > 0:
#         filtered = filtered[
#             filtered["PROBABILITY"] >= min_prob
#         ]


#     # ======================
#     # DISPLAY TABLE
#     # ======================
#     st.markdown("### ✅ Results")

#     if filtered.empty:
#         st.error("No colleges match your filters.")
#     else:
#         st.dataframe(
#             filtered[
#                 [
#                     "INST_CODE","INST_NAME","TYPE","BRANCH_NAME",
#                     "DIST","PLACE","COED",
#                     "PREDICTED_CUTOFF","PROBABILITY_%"
#                 ]
#             ],
#             use_container_width=True
#         )


