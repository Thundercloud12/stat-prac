import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df = pd.read_csv("train.csv")

def decide(p):
    return "Reject H0" if p < 0.05 else "Fail to reject H0"

# 1. T-test: Fare by Sex
g1 = df[df["Sex"] == "male"]["Fare"].dropna()
g2 = df[df["Sex"] == "female"]["Fare"].dropna()

print("Male:", len(g1), g1.mean(), g1.std())
print("Female:", len(g2), g2.mean(), g2.std())

sns.boxplot(data=df, x="Sex", y="Fare")
plt.show()

t, p1 = ttest_ind(g1, g2, equal_var=False)
print("T-test:", t, p1, decide(p1))

# 2. Chi-square: Sex vs Survived
table = pd.crosstab(df["Sex"], df["Survived"])
print(table)

sns.heatmap(table, annot=True, fmt="d")
plt.show()

chi, p2, dof, expected = chi2_contingency(table)
print("Chi-square:", chi, "DOF:", dof, "p:", p2, decide(p2))
print("Expected frequencies:", expected)
# print("Expected cells >= 5:", (expected >= 5).mean() * 100, "%")

# 3. ANOVA: Fare by Pclass
groups = [g["Fare"].dropna() for _, g in df.groupby("Pclass")]

print(df.groupby("Pclass")["Fare"].agg(["count", "mean"]))
sns.boxplot(data=df, x="Pclass", y="Fare")
plt.show()

f, p3 = f_oneway(*groups)
print("ANOVA:", f, p3, decide(p3))

# 4. Tukey HSD
print(pairwise_tukeyhsd(df["Fare"], df["Pclass"]))

# 5. Summary
result = pd.DataFrame({
    "Test": ["T-test", "Chi-square", "ANOVA"],
    "Statistic": [t, chi, f],
    "P-value": [p1, p2, p3],
    "Decision": [decide(p) for p in [p1, p2, p3]]
})
print(result)

