import pandas as pd

df1 = pd.read_csv("long_2022.csv")
df2 = pd.read_csv("long_2023.csv")
df3 = pd.read_csv("long_2024.csv")

# Combine
final_df = pd.concat([df1, df2, df3], ignore_index=True)

# Optional: remove duplicates
final_df = final_df.drop_duplicates()

# Save
final_df.to_csv("final_dataset.csv", index=False)

print("Final dataset created successfully ✅")
print(final_df.shape)

print(final_df.isnull().sum())
print(final_df["CUTOFF"].min(), final_df["CUTOFF"].max())
print(final_df["CATEGORY"].unique())
print(final_df["GENDER"].unique())
print(final_df["YEAR"].unique())