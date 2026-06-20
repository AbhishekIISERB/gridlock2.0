import pandas as pd

df = pd.read_csv("jan_to_may_police_violation_anonymized.csv")

print(df.shape)
print(df.dtypes)
print(df.head(3))
print(df.isnull().sum())
print(df.nunique())