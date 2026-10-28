
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("train.csv")

# Dataset information
print(df.head())
print(df.shape)
df.info()
print(df.isnull().sum())

# Descriptive statistics
num = df[["Age", "Fare", "SibSp", "Parch"]]
print(num.describe())

# Histograms
num.hist(bins=10, figsize=(10, 6))
plt.tight_layout()
plt.show()

# Correlation matrix and heatmap
corr = num.corr()
print(corr)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()

# Strongest correlations
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
print(pairs.sort_values(key=abs, ascending=False).head(10))
