
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)


df = pd.read_csv("train.csv")

# 1. Same data as Experiment 8
cols = ["Age", "Fare", "Pclass", "SibSp", "Parch"]
data = df[cols + ["Survived"]].dropna()

X = data[cols]
y = data["Survived"]

print("Class mapping: 0 = Died, 1 = Survived")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train:", len(X_train), "Test:", len(X_test))

# 2. Models
models = {
    "Full Tree": DecisionTreeClassifier(random_state=42),
    "Small Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_split=10,
        min_samples_leaf=5, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, oob_score=True, random_state=42
    )
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    results.append([
        name,
        accuracy_score(y_train, model.predict(X_train)),
        accuracy_score(y_test, pred),
        precision_score(y_test, pred),
        recall_score(y_test, pred),
        f1_score(y_test, pred),
        roc_auc_score(y_test, prob)
    ])

    print("\n", name)
    print(classification_report(y_test, pred))

    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d")
    plt.title(name)
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=name)


result = pd.DataFrame(results, columns=[
    "Model", "Train Accuracy", "Accuracy",
    "Precision", "Recall", "F1", "AUC"
]).set_index("Model")

print(result)
# 5. Random Forest OOB score
rf = models["Random Forest"]
print("OOB Score:", rf.oob_score_)

# 6. Feature importance
perm = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42
)

importance = pd.DataFrame({
    "Tree": models["Full Tree"].feature_importances_,
    "Forest": rf.feature_importances_,
    "Permutation": perm.importances_mean,
    "Permutation SD": perm.importances_std
}, index=cols)

print(importance)

# importance[["Tree", "Forest"]].plot(kind="barh")
# plt.show()

# importance["Permutation"].plot(
#     kind="barh", xerr=importance["Permutation SD"]
# )
# plt.show()


