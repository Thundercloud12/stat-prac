
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("train.csv")

N = len(df)
n = int(0.2 * N)
mean = df["Fare"].mean()

print("Population:", N, "Sample:", n, "Mean:", mean)

# 1. Simple random sampling
srs = df.sample(n=n, random_state=42)
print(srs.head())
print(df.sample(n=n, random_state=10).head())

# 2. Stratified sampling using Sex
sns.countplot(data=df, x="Sex")
plt.show()

strat = df.groupby("Sex", group_keys=False).sample(
    frac=0.2, random_state=42
)

print(pd.DataFrame({
    "Population %": df["Sex"].value_counts(normalize=True) * 100,
    "Sample %": strat["Sex"].value_counts(normalize=True) * 100
}))

# 3. Cluster sampling using Embarked
# Cluster sampling
clusters = df["Embarked"].dropna().unique()

selected = np.random.choice(clusters, size=1, replace=False)

cluster = df[df["Embarked"].isin(selected)]

print("Selected clusters:", selected)

# 4. Comparison table
samples = {"SRS": srs, "Stratified": strat, "Cluster": cluster}

result = pd.DataFrame({
    "Size": [len(s) for s in samples.values()],
    "Mean": [s["Fare"].mean() for s in samples.values()]
}, index=samples.keys())

result["Difference"] = result["Mean"] - mean
print(result)

# 5. Graph
result["Mean"].plot(kind="bar")
plt.axhline(mean, color="red", label="Population Mean")
plt.legend()
plt.show()


