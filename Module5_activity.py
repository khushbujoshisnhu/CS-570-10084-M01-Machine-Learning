# CS 570 - Module 5 Activity
# Completed from the professor-provided starter code.
# The program performs data exploration, encoding, correlation analysis,
# pre-imputation model testing, four imputation comparisons, and visualizations.

import os
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import OrdinalEncoder

# Imputation
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.impute import IterativeImputer

# Model selection and pipeline
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# Regression
from sklearn.linear_model import LinearRegression


# ----------------------------------------------------------------
# Function to test dataset on a linear regression model
# Use this function to compare how different imputation methods
# affect model performance.
# ----------------------------------------------------------------
def get_score(df, imputer=None):
    # Define features (X) and target (y)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    regressor = LinearRegression()

    if imputer is not None:
        estimator = make_pipeline(imputer, regressor)
    else:
        estimator = regressor

    scores = cross_val_score(
        estimator,
        X,
        y,
        scoring="neg_mean_squared_error",
        cv=4
    )

    return scores.mean(), scores.std()


# ----------------------------------------------------------------
# Step 1: Load the CSV file into a pandas DataFrame
# ----------------------------------------------------------------
# Use Module5Dataset.csv when running in Codio.
# This also works with the uploaded file name used for verification.
if os.path.exists("Module5Dataset.csv"):
    data_file = "Module5Dataset.csv"
elif os.path.exists("Module5Dataset.csv.csv"):
    data_file = "Module5Dataset.csv.csv"
else:
    raise FileNotFoundError(
        "Module5Dataset.csv was not found. Place the dataset in the same folder as this script."
    )

df = pd.read_csv(data_file)


# ----------------------------------------------------------------
# Step 2: Data Exploration and Visualization
# ----------------------------------------------------------------
print("\n================ DATA EXPLORATION ================\n")

print("Dataset shape:", df.shape)
print("\nHeader and first 4 samples:\n")
print(df.head(4))

print("\nData types:\n")
print(df.dtypes)

print("\nMissing values by feature:\n")
print(df.isna().sum())

print("\nCount, unique, top, and freq:\n")
print(df.describe(include="all").T)

print("\nUnique values in each feature:\n")
for column in df.columns:
    print(f"{column}: {df[column].dropna().unique()}")

# Visualize the distribution / missingness of the original features.
# Because the original features are categorical, count plots are used.
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, column in zip(axes.ravel(), df.columns[:-1]):
    counts = df[column].value_counts(dropna=False)
    counts.plot(kind="bar", ax=ax)
    ax.set_title(f"{column} Distribution")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------
# Step 3: Transform the dataset for linear regression
# ----------------------------------------------------------------
# The six predictor features are categorical but have an ordinal
# ordering in the dataset. OrdinalEncoder converts them to numeric
# values while preserving that ordering.
#
# Feature1: low < med < high < vhigh
# Feature2: low < med < high < vhigh
# Feature3: 2 < 3 < 4 < 5more
# Feature4: 2 < 4 < more
# Feature5: small < med < big
# Feature6: low < med < high
#
# Class is encoded as:
# unacc=0, acc=1, good=2, vgood=3
# ----------------------------------------------------------------

feature_columns = [
    "Feature1", "Feature2", "Feature3",
    "Feature4", "Feature5", "Feature6"
]

feature_categories = [
    ["low", "med", "high", "vhigh"],
    ["low", "med", "high", "vhigh"],
    ["2", "3", "4", "5more"],
    ["2", "4", "more"],
    ["small", "med", "big"],
    ["low", "med", "high"]
]

class_categories = [["unacc", "acc", "good", "vgood"]]

feature_encoder = OrdinalEncoder(
    categories=feature_categories,
    handle_unknown="use_encoded_value",
    unknown_value=np.nan
)

class_encoder = OrdinalEncoder(categories=class_categories)

transformed_features = feature_encoder.fit_transform(df[feature_columns])
transformed_class = class_encoder.fit_transform(df[["Class"]]).ravel()

transformed_df = pd.DataFrame(
    transformed_features,
    columns=feature_columns,
    index=df.index
)
transformed_df["Class"] = transformed_class

print("\n================ TRANSFORMED DATA ================\n")
print(transformed_df.head(4))
print("\nClass encoding:")
print("unacc=0, acc=1, good=2, vgood=3")


# ----------------------------------------------------------------
# Step 3 continued: Correlation matrix
# ----------------------------------------------------------------
print("\n================ CORRELATION MATRIX ================\n")
print(transformed_df.corr())

plt.figure(figsize=(9, 7))
sns.heatmap(
    transformed_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Matrix - Ordinally Encoded Dataset")
plt.tight_layout()
plt.show()


# ----------------------------------------------------------------
# Step 4: Score the model before imputation
# ----------------------------------------------------------------
print("\n================ PRE-IMPUTATION MODEL SCORE ================\n")
print(
    "The raw dataset cannot be used directly by LinearRegression because "
    "the predictors are categorical strings."
)
print(
    "After ordinal encoding, the dataset is numeric, but it still contains "
    "NaN values. Therefore, LinearRegression cannot produce a valid "
    "pre-imputation cross-validation score."
)

try:
    raw_mse, raw_std = get_score(transformed_df)
    print("RAW Dataset negative MSE:", raw_mse)
    print("RAW Dataset STD:", raw_std)
except ValueError as error:
    raw_mse = np.nan
    raw_std = np.nan
    print("RAW Dataset score: NOT AVAILABLE before imputation.")
    print("Reason:", error)


# ----------------------------------------------------------------
# Step 5: Impute missing values
# The professor's starter code specifies:
#   1. SimpleImputer(strategy='constant', fill_value=0)
#   2. SimpleImputer(strategy='mean')
#   3. KNNImputer()
#   4. IterativeImputer(max_iter=10, random_state=42)
# ----------------------------------------------------------------

x_labels = []
mses = np.zeros(5)
stds = np.zeros(5)
mses[:] = np.nan
stds[:] = np.nan

# Raw dataset placeholder
mses[0] = raw_mse
stds[0] = raw_std
x_labels.append("RAW Dataset")


# Zero Imputation
zero_imputer = SimpleImputer(strategy="constant", fill_value=0)
mses[1], stds[1] = get_score(transformed_df, zero_imputer)
x_labels.append("Zero Imputation")


# Mean Imputation
mean_imputer = SimpleImputer(strategy="mean")
mses[2], stds[2] = get_score(transformed_df, mean_imputer)
x_labels.append("Mean Imputation")


# KNN Imputation
knn_imputer = KNNImputer()
mses[3], stds[3] = get_score(transformed_df, knn_imputer)
x_labels.append("KNN Imputation")


# Iterative Imputation
iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
mses[4], stds[4] = get_score(transformed_df, iterative_imputer)
x_labels.append("Iterative Imputation")


# ----------------------------------------------------------------
# Step 6: Collect metrics and visualize imputed datasets
# ----------------------------------------------------------------
print("\n================ CROSS-VALIDATION RESULTS ================\n")

results = pd.DataFrame({
    "Imputation Method": x_labels,
    "Negative MSE": mses,
    "STD": stds
})

print(results.to_string(index=False))

# Lower MSE means better performance. Since the scorer is
# negative MSE, the value closer to zero is better.
valid_results = results.dropna(subset=["Negative MSE"])
best_row = valid_results.loc[valid_results["Negative MSE"].idxmax()]

print("\nBest method based on negative MSE:")
print(best_row.to_string())


# Create fully imputed dataframes for visualization.
zero_data = pd.DataFrame(
    zero_imputer.fit_transform(transformed_df),
    columns=transformed_df.columns,
    index=transformed_df.index
)

mean_data = pd.DataFrame(
    mean_imputer.fit_transform(transformed_df),
    columns=transformed_df.columns,
    index=transformed_df.index
)

knn_data = pd.DataFrame(
    knn_imputer.fit_transform(transformed_df),
    columns=transformed_df.columns,
    index=transformed_df.index
)

iterative_data = pd.DataFrame(
    iterative_imputer.fit_transform(transformed_df),
    columns=transformed_df.columns,
    index=transformed_df.index
)

print("\nMissing values after imputation:")
print("Zero:", int(zero_data.isna().sum().sum()))
print("Mean:", int(mean_data.isna().sum().sum()))
print("KNN:", int(knn_data.isna().sum().sum()))
print("Iterative:", int(iterative_data.isna().sum().sum()))


# Correlation matrix after mean imputation.
plt.figure(figsize=(9, 7))
sns.heatmap(
    mean_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Matrix After Mean Imputation")
plt.tight_layout()
plt.show()


# Visualize distributions after mean imputation.
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, column in zip(axes.ravel(), feature_columns):
    mean_data[column].plot(kind="hist", bins=10, ax=ax)
    ax.set_title(f"{column} After Mean Imputation")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")

plt.tight_layout()
plt.show()


# Plot cross-validation results for the four imputation methods.
plot_results = results.dropna(subset=["Negative MSE"]).copy()

plt.figure(figsize=(10, 6))
plt.bar(
    plot_results["Imputation Method"],
    plot_results["Negative MSE"],
    yerr=plot_results["STD"],
    capsize=5
)
plt.xlabel("Imputation Method")
plt.ylabel("Negative Mean Squared Error")
plt.title("Comparison of Imputation Methods")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

print("\n================ ACTIVITY COMPLETE ================\n")
