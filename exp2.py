
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


df = pd.read_csv("train.csv")

print(df.shape)
print(pd.DataFrame({
    "Missing": df.isnull().sum(),
    "Percent": df.isnull().mean() * 100
}))

# Missing values
num = ["Age", "Fare", "SibSp", "Parch"]
cat = ["Sex", "Embarked"]

for col in num:
    df[col] = df[col].fillna(df[col].median())

for col in cat:
    df[col] = df[col].fillna(df[col].mode()[0])

print(df[num + cat].isnull().sum())

# Encoding
print(pd.get_dummies(df[cat], drop_first=True).head())

df["Sex_encoded"] = LabelEncoder().fit_transform(df["Sex"])
print(df[["Sex", "Sex_encoded"]].head())

# Scaling
std = pd.DataFrame(StandardScaler().fit_transform(df[num]), columns=num)
mm = pd.DataFrame(MinMaxScaler().fit_transform(df[num]), columns=num)

print(std.agg(["mean", "std"]))
print(mm.agg(["min", "max"]))

# Box plot
sns.boxplot(data=std)
plt.show()

# IQR table
rows = []
for col in num:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    count = ((df[col] < low) | (df[col] > high)).sum()
    rows.append([col, q1, q3, iqr, low, high, count])

table = pd.DataFrame(rows, columns=[
    "Column", "Q1", "Q3", "IQR", "Lower", "Upper", "Outliers"
])
print(table)

# Remove outliers from Fare
low = table.set_index("Column").loc["Fare", "Lower"]
high = table.set_index("Column").loc["Fare", "Upper"]

clean = df[df["Fare"].between(low, high)]
print("Before:", len(df), "After:", len(clean))

