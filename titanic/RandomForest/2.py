import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")

# Combine train and test data
passenger_ids = test["PassengerId"]
combined = pd.concat([train.drop("Survived", axis=1), test], axis=0, ignore_index=True)

combined = combined.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Handle Missing Values
imputer = SimpleImputer(strategy="median")
combined["Age"] = imputer.fit_transform(combined[["Age"]])
combined["Fare"] = imputer.fit_transform(combined[["Fare"]])

combined["Embarked"] = combined["Embarked"].fillna(combined["Embarked"].mode()[0])

# Extract Title from Name
train_data = pd.read_csv("../train.csv")
train_data["Title"] = train_data["Name"].apply(
    lambda x: x.split(",")[1].split(".")[0].strip()
)
train_data["Title"] = train_data["Title"].replace(["Mlle", "Ms", "Mme"], "Miss")
train_data["Title"] = train_data["Title"].replace(
    ["Lady", "Countess", "Dona"], "Royalty"
)
train_data["Title"] = train_data["Title"].replace(
    ["Capt", "Col", "Major", "Dr", "Rev"], "Officer"
)
train_data["Title"] = train_data["Title"].replace(["Don", "Sir", "Jonkheer"], "Noble")

combined["Title"] = train_data["Title"]

# FamilySize Feature
combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1

# IsAlone Feature
combined["IsAlone"] = (combined["FamilySize"] == 1).astype(int)

# FarePerPerson Feature
combined["FarePerPerson"] = combined["Fare"] / combined["FamilySize"]
combined["FarePerPerson"].fillna(combined["FarePerPerson"].median(), inplace=True)

# Age*Class Interaction Feature
combined["Age*Class"] = combined["Age"] * combined["Pclass"]

# FareBin Feature
combined["FareBin"] = pd.qcut(combined["Fare"], 4, labels=[1, 2, 3, 4])

# AgeBin Feature (Binned Age into categories)
combined["AgeBin"] = pd.cut(
    combined["Age"], bins=[0, 12, 20, 40, 60, 80], labels=[0, 1, 2, 3, 4]
)

# interaction terms
combined["Pclass*Sex"] = (
    combined["Pclass"].astype(str) + "_" + combined["Sex"].astype(str)
)
combined["AgeBin*Pclass"] = (
    combined["AgeBin"].astype(str) + "_" + combined["Pclass"].astype(str)
)

# Encode Categorical Variables
le = LabelEncoder()
for col in ["Sex", "Title", "Pclass*Sex", "AgeBin*Pclass"]:
    combined[col] = le.fit_transform(combined[col])

# One-Hot Encoding 'Embarked'
combined = pd.get_dummies(combined, columns=["Embarked"], drop_first=True)

# Scale numerical features
scaler = StandardScaler()
combined[["Age", "Fare", "FarePerPerson", "Age*Class"]] = scaler.fit_transform(
    combined[["Age", "Fare", "FarePerPerson", "Age*Class"]]
)

# Split the combined dataset back into train and test sets
train_cleaned = combined[: len(train)]
test_cleaned = combined[len(train) :]

train_cleaned["Survived"] = train["Survived"]

X = train_cleaned.drop("Survived", axis=1)
y = train_cleaned["Survived"]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


param_grid_rf = {
    "n_estimators": range(60, 370, 50),
    "max_features": [
        "sqrt",
        "log2",
        None,
    ],
    "max_depth": [5, 10, 15],
    "min_samples_split": [8, 9, 10],
    "min_samples_leaf": [
        1,
        2,
        4,
    ],
    "max_leaf_nodes": [78, 84, 90],
    "class_weight": [{0: 1, 1: 1.5}, {0: 1, 1: 1.6}],
    "ccp_alpha": [0.00003, 0.00005, 0.00007],
}

rfc = RandomForestClassifier(
    criterion="gini",
    max_depth=10,
    min_samples_split=8,
    min_samples_leaf=1,
    max_leaf_nodes=84,
    class_weight={0: 1, 1: 1.5},
    ccp_alpha=4e-05,
    random_state=1212,
)

grid_search = GridSearchCV(
    estimator=rfc,
    param_grid=param_grid_rf,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=10,
)
grid_search.fit(X_train, y_train)
print(f"Best Parameters: {grid_search.best_params_}")

# Predict
best_rfc = grid_search.best_estimator_
y_val_pred = best_rfc.predict(X_val)

accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

test_predictions = best_rfc.predict(test_cleaned)

submission = pd.DataFrame({"PassengerId": passenger_ids, "Survived": test_predictions})

submission.to_csv("titanic_submission.csv", index=False)
print("Submission file created!")
