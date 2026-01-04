import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="College Predictor", layout="wide")

# ======================
# LOAD MODEL & DATA
# ======================
@st.cache_resource
def load_resources():
    model = joblib.load("cutoff_predictor_xgb.pkl")
    encoders = joblib.load("cutoff_encoders.pkl")
    df = pd.read_csv("cleaned_dataset.csv")
    return model, encoders, df

model, encoders, base_df = load_resources()

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
def predict_colleges(
    rank,
    gender,
    category,
    branch_code=None,
    districts=None,
    college_type=None,
    min_probability=0,
    year=2025
):

    df = base_df.copy()

    # ---------- BASE FILTERS ----------
    if branch_code:
        df = df[df["BRANCH_CODE"] == branch_code]

    if districts:
        df = df[df["DIST"].isin(districts)]

    if college_type:
        df = df[df["COED"] == college_type]

    if gender == "MALE":
        df = df[df["COED"] != "GIRLS"]

    # ---------- ADD USER FEATURES ----------
    df["YEAR"] = year
    df["GENDER"] = gender
    df["CATEGORY"] = category

    FEATURES = ["YEAR","GENDER","CATEGORY","BRANCH_CODE","DIST","COED","TYPE"]

    # ---------- ENCODE ----------
    for col, le in encoders.items():
        df[col] = df[col].map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    # ---------- PREDICT ----------
    X = df[FEATURES]
    df["PREDICTED_CUTOFF"] = model.predict(X).astype(int)

    df["PROBABILITY"] = df["PREDICTED_CUTOFF"].apply(
        lambda x: admission_probability(rank, x)
    )

    # ---------- AUTO FILTER ----------
    df = df[df["PROBABILITY"] >= min_probability]

    # ---------- CLEAN ----------
    df = df.drop_duplicates(["INST_CODE","BRANCH_CODE"])

    # ---------- DECODE ----------
    for col, le in encoders.items():
        df[col] = df[col].apply(
            lambda x: le.inverse_transform([int(x)])[0] if x != -1 else "UNKNOWN"
        )

    return df[
        ["INST_CODE","INST_NAME","TYPE","BRANCH_NAME",
         "DIST","PLACE","COED",
         "PREDICTED_CUTOFF","PROBABILITY"]
    ].sort_values("PROBABILITY", ascending=False).reset_index(drop=True)

# ======================
# UI
# ======================
st.title("🎓 ML-Based College Predictor (AP EAPCET)")

st.sidebar.header("🔍 Student Details")

rank = st.sidebar.number_input("EAPCET Rank", min_value=1, step=1)
gender = st.sidebar.selectbox("Gender", ["MALE", "FEMALE"])
category = st.sidebar.selectbox("Category", ["OC", "BC", "SC", "ST"])
branch_code = st.sidebar.selectbox(
    "Branch",
    [""] + sorted(base_df["BRANCH_CODE"].unique())
)
branch_code = branch_code if branch_code else None

st.sidebar.header("🎯 Filters")

districts = st.sidebar.multiselect(
    "Districts",
    sorted(base_df["DIST"].unique())
)

college_type = st.sidebar.selectbox(
    "College Type",
    ["", "COED", "GIRLS"]
)
college_type = college_type if college_type else None

min_probability = st.sidebar.slider(
    "Minimum Admission Probability (%)",
    0, 90, 30
)

# ======================
# PREDICT BUTTON
# ======================
if st.button("🚀 Predict Colleges"):

    if rank == 0:
        st.warning("Please enter a valid rank.")
    else:
        with st.spinner("Predicting colleges..."):
            result = predict_colleges(
                rank=rank,
                gender=gender,
                category=category,
                branch_code=branch_code,
                districts=districts,
                college_type=college_type,
                min_probability=min_probability
            )

        if result.empty:
            st.error("No colleges found. Try relaxing filters.")
        else:
            st.success(f"Found {len(result)} colleges")
            st.dataframe(result, use_container_width=True)

            # Download
            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results",
                csv,
                "college_predictions.csv",
                "text/csv"
            )

# ======================
# FOOTER
# ======================
st.markdown("---")
st.caption("Built using Machine Learning (XGBoost) • Predicts college admission probability based on rank trends")


# import streamlit as st
# import pandas as pd
# import joblib

# # ======================
# # PAGE CONFIG
# # ======================
# st.set_page_config(page_title="AP EAPCET College Predictor", layout="wide")

# st.title("🎓 AP EAPCET College Predictor")
# st.caption("ML-based prediction using historical cutoff trends")

# # ======================
# # LOAD MODEL & DATA
# # ======================
# @st.cache_resource
# def load_model():
#     return (
#         joblib.load("cutoff_predictor_xgb.pkl"),
#         joblib.load("cutoff_encoders.pkl")
#     )

# @st.cache_data
# def load_data():
#     return pd.read_csv("cleaned_dataset.csv")

# model, encoders = load_model()
# base_df = load_data()

# # ======================
# # PROBABILITY LOGIC
# # ======================
# def admission_probability(rank, cutoff):
#     diff = cutoff - rank
#     if diff < -10000: return 0
#     if diff < -5000: return 5
#     if diff < 0: return 15
#     if diff < 3000: return 40
#     if diff < 8000: return 65
#     return 85

# # ======================
# # PREDICTION FUNCTION
# # ======================
# def predict_colleges(rank, gender, category, branch, year):

#     df = base_df.copy()

#     if branch != "ALL":
#         df = df[df["BRANCH_CODE"] == branch]

#     if gender == "MALE":
#         df = df[df["COED"] != "GIRLS"]

#     df["YEAR"] = year
#     df["GENDER"] = gender
#     df["CATEGORY"] = category

#     FEATURES = ["YEAR","GENDER","CATEGORY","BRANCH_CODE","DIST","COED","TYPE"]

#     encoded = df.copy()
#     for col, le in encoders.items():
#         encoded[col] = encoded[col].map(
#             lambda x: le.transform([x])[0] if x in le.classes_ else -1
#         )

#     df["PREDICTED_CUTOFF"] = model.predict(encoded[FEATURES]).astype(int)

#     # STRICT ELIGIBILITY
#     df = df[df["PREDICTED_CUTOFF"] >= rank]

#     df["PROBABILITY"] = df["PREDICTED_CUTOFF"].apply(
#         lambda x: admission_probability(rank, x)
#     )
#     df["PROBABILITY_%"] = df["PROBABILITY"].astype(str) + "%"

#     return df.drop_duplicates(
#         ["INST_CODE","BRANCH_CODE"]
#     ).reset_index(drop=True)

# # ======================
# # SIDEBAR INPUTS
# # ======================
# with st.sidebar:
#     st.header("📝 Student Details")

#     rank = st.number_input("Rank", 1, 200000, 19000)
#     gender = st.selectbox("Gender", ["FEMALE","MALE"])
#     category = st.selectbox("Category", ["OC","BC","SC","ST"])
#     branch = st.selectbox(
#         "Branch",
#         ["ALL"] + sorted(base_df["BRANCH_CODE"].unique())
#     )
#     year = 2025

#     predict_btn = st.button("🔮 Predict Colleges")

# # ======================
# # PREDICT
# # ======================
# if predict_btn:
#     st.session_state["predicted"] = predict_colleges(
#         rank, gender, category, branch, year
#     )

# # ======================
# # FILTER + DISPLAY (AUTO)
# # ======================
# if "predicted" in st.session_state:

#     df = st.session_state["predicted"]

#     st.subheader("🎯 Filters")

#     c1, c2, c3, c4 = st.columns(4)

#     with c1:
#         f_branch = st.multiselect(
#             "Branch Name",
#             sorted(df["BRANCH_NAME"].unique())
#         )

#     with c2:
#         f_dist = st.multiselect(
#             "District",
#             sorted(df["DIST"].unique())
#         )

#     with c3:
#         f_type = st.selectbox(
#             "College Type",
#             ["ALL","COED","GIRLS"]
#         )

#     with c4:
#         f_prob = st.slider(
#             "Minimum Probability %",
#             0, 100, 40
#         )

#     # APPLY FILTERS LIVE
#     filtered = df.copy()

#     if f_branch:
#         filtered = filtered[filtered["BRANCH_NAME"].isin(f_branch)]
#     if f_dist:
#         filtered = filtered[filtered["DIST"].isin(f_dist)]
#     if f_type != "ALL":
#         filtered = filtered[filtered["COED"] == f_type]

#     filtered = filtered[filtered["PROBABILITY"] >= f_prob]

#     st.subheader("🏫 Predicted Colleges")

#     if filtered.empty:
#         st.warning("No colleges match your filters.")
#     else:
#         st.dataframe(
#             filtered.sort_values("PROBABILITY", ascending=False)[
#                 ["INST_CODE","INST_NAME","TYPE",
#                  "BRANCH_NAME","DIST","PLACE",
#                  "COED","PREDICTED_CUTOFF","PROBABILITY_%"]
#             ],
#             use_container_width=True
#         )




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
 # ======================
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








