import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# ============================================================
# PART 1: CS-570 Machine Learning Project
# ============================================================

# ------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------

file_path = "ProjectPartOneDataSet.csv.csv"

df = pd.read_csv(file_path)

print("=" * 70)
print("1. DATASET LOADED")
print("=" * 70)

print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\nDataset:")
print(df.head())

# ------------------------------------------------------------
# 2. Display column names
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2. COLUMN NAMES")
print("=" * 70)

print(df.columns.tolist())

# ------------------------------------------------------------
# 3. Display data types
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. DATA TYPES")
print("=" * 70)

print(df.dtypes)

# ------------------------------------------------------------
# 4. Check for missing values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. MISSING VALUES")
print("=" * 70)

missing_values = df.isnull().sum()

print(missing_values)

print("\nTotal missing values:", df.isnull().sum().sum())

# ------------------------------------------------------------
# 5. Check for duplicate records
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("5. DUPLICATE RECORDS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Number of duplicate records:", duplicate_count)

# ------------------------------------------------------------
# 6. Display unique values for each column
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("6. UNIQUE VALUES")
print("=" * 70)

for column in df.columns:
    print(f"\n{column}:")
    print(df[column].unique())

# ------------------------------------------------------------
# 7. Number of unique values in each column
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("7. NUMBER OF UNIQUE VALUES")
print("=" * 70)

print(df.nunique())

# ------------------------------------------------------------
# 8. Target/class distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("8. CLASS DISTRIBUTION")
print("=" * 70)

class_counts = df["class"].value_counts()

print(class_counts)

print("\nClass percentages:")
print((df["class"].value_counts(normalize=True) * 100).round(2))

# ------------------------------------------------------------
# 9. Descriptive statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("9. DESCRIPTIVE STATISTICS")
print("=" * 70)

print(df.describe(include="all"))

# ------------------------------------------------------------
# 10. Histogram Visualization
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("10. GENERATING HISTOGRAM")
print("=" * 70)

# Convert categorical class labels to numeric values only for histogram visualization.
class_mapping = {
    "unacc": 1,
    "acc": 2,
    "good": 3,
    "vgood": 4
}

class_numeric = df["class"].map(class_mapping)

plt.figure(figsize=(8, 6))

plt.hist(
    class_numeric,
    bins=[0.5, 1.5, 2.5, 3.5, 4.5],
    edgecolor="black"
)

plt.title("Histogram of Target Class Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")

plt.xticks(
    [1, 2, 3, 4],
    ["unacc", "acc", "good", "vgood"]
)

plt.tight_layout()
plt.savefig("part1_class_distribution.png",dpi=300,bbox_inches="tight")
plt.show()
plt.close()

# ------------------------------------------------------------
# 11. Prepare categorical data for correlation analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("11. PREPARING DATA FOR CORRELATION ANALYSIS")
print("=" * 70)

# Explicit ordinal mappings based on the ordered category
# labels present in the dataset.

ordinal_mappings = {
    "feat1": {
        "low": 1,
        "med": 2,
        "high": 3,
        "vhigh": 4
    },

    "feat2": {
        "low": 1,
        "med": 2,
        "high": 3,
        "vhigh": 4
    },

    "feat3": {
        "2": 2,
        "3": 3,
        "4": 4,
        "5more": 5
    },

    "feat4": {
        "2": 2,
        "4": 4,
        "more": 5
    },

    "feat5": {
        "small": 1,
        "med": 2,
        "big": 3
    },

    "feat6": {
        "low": 1,
        "med": 2,
        "high": 3
    },

    "class": {
        "unacc": 1,
        "acc": 2,
        "good": 3,
        "vgood": 4
    }
}

correlation_df = df.copy()

for column, mapping in ordinal_mappings.items():
    correlation_df[column] = correlation_df[column].map(mapping)

print("\nEncoded data used ONLY for exploratory correlation analysis:")
print(correlation_df.head())

# ------------------------------------------------------------
# 12. Correlation matrix
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("12. CORRELATION MATRIX")
print("=" * 70)

correlation_matrix = correlation_df.corr(numeric_only=True)

print(correlation_matrix.round(3))

# ------------------------------------------------------------
# 13. Correlation matrix visualization
# ------------------------------------------------------------

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig("part1_correlation_matrix.png", dpi=300)

plt.show()

# ------------------------------------------------------------
# 14. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PART 1 DATA EXPLORATION COMPLETED")
print("=" * 70)

print(f"Dataset size: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate records: {df.duplicated().sum()}")
print(f"Target variable: class")
print(f"Number of target classes: {df['class'].nunique()}")

print("\nClass distribution:")
print(df["class"].value_counts())

print("\nPlots saved:")
print("1. part1_class_distribution.png")
print("2. part1_correlation_matrix.png")

# ============================================================
# PART 1: DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("PART 1: DATA PREPROCESSING")
print("=" * 70)

# ------------------------------------------------------------
# 15. Separate features and target
# ------------------------------------------------------------

X = df.drop("class", axis=1)
y = df["class"]

print("\n" + "=" * 70)
print("15. FEATURES AND TARGET")
print("=" * 70)

print("Feature columns:")
print(X.columns.tolist())

print("\nTarget column:")
print("class")

# ------------------------------------------------------------
# 16. Split data into training and testing sets
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 70)
print("16. TRAIN-TEST SPLIT")
print("=" * 70)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# ------------------------------------------------------------
# 17. Define the ordered categories
# ------------------------------------------------------------

ordered_categories = [
    ["low", "med", "high", "vhigh"],   # feat1
    ["low", "med", "high", "vhigh"],   # feat2
    ["2", "3", "4", "5more"],          # feat3
    ["2", "4", "more"],                # feat4
    ["small", "med", "big"],           # feat5
    ["low", "med", "high"]             # feat6
]

# ------------------------------------------------------------
# 18. Create the ordinal encoder
# ------------------------------------------------------------

ordinal_encoder = OrdinalEncoder(
    categories=ordered_categories,
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

# ------------------------------------------------------------
# 19. Fit encoder ONLY on training data
# ------------------------------------------------------------

X_train_encoded = ordinal_encoder.fit_transform(X_train)

X_test_encoded = ordinal_encoder.transform(X_test)

print("\n" + "=" * 70)
print("17. ORDINAL ENCODING")
print("=" * 70)

print("Original training data:")
print(X_train.head())

print("\nEncoded training data:")
print(
    pd.DataFrame(
        X_train_encoded,
        columns=X_train.columns
    ).head()
)

print("\nEncoded training data shape:")
print(X_train_encoded.shape)

print("\nEncoded testing data shape:")
print(X_test_encoded.shape)

# ------------------------------------------------------------
# 20. Standardization
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_encoded)

X_test_scaled = scaler.transform(X_test_encoded)

print("\n" + "=" * 70)
print("18. STANDARDIZATION")
print("=" * 70)

print("Scaled training data:")
print(
    pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns
    ).head()
)

print("\nScaled testing data shape:")
print(X_test_scaled.shape)

# ------------------------------------------------------------
# 21. Preprocessing summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("19. PREPROCESSING SUMMARY")
print("=" * 70)

print("1. Target variable separated: class")
print("2. Data split into training and testing sets")
print("3. Stratified split used to preserve class proportions")
print("4. Ordered categorical features encoded using OrdinalEncoder")
print("5. Encoder fitted only on training data")
print("6. Testing data transformed using the fitted encoder")
print("7. Encoded features standardized using StandardScaler")
print("8. No missing-value imputation required")
print("9. No duplicate-record removal required")

# ============================================================
# 19. MODEL EXPERIMENTATION AND COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("19. MODEL EXPERIMENTATION AND COMPARISON")
print("=" * 80)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
    classification_report
)

# ------------------------------------------------------------
# Create preprocessing pipelines
# ------------------------------------------------------------

# kNN:
# Ordinal encoding + Standardization
knn_pipeline = Pipeline([
    ("encoder", OrdinalEncoder(
        categories=ordered_categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )),
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(n_neighbors=5))
])

# Linear Discriminant Analysis:
# Ordinal encoding + Standardization
lda_pipeline = Pipeline([
    ("encoder", OrdinalEncoder(
        categories=ordered_categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )),
    ("scaler", StandardScaler()),
    ("model", LinearDiscriminantAnalysis())
])

# Gaussian Naive Bayes:
# Ordinal encoding
nb_pipeline = Pipeline([
    ("encoder", OrdinalEncoder(
        categories=ordered_categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )),
    ("model", GaussianNB())
])

# ------------------------------------------------------------
# Store the models
# ------------------------------------------------------------

models = {
    "kNN": knn_pipeline,
    "Discriminant Analysis": lda_pipeline,
    "Naive Bayes": nb_pipeline
}

# ------------------------------------------------------------
# Create 5-Fold Stratified Cross-Validation
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ------------------------------------------------------------
# Cross-validation and model evaluation
# ------------------------------------------------------------

results = []

for model_name, model in models.items():

    print("\n" + "-" * 80)
    print(model_name)
    print("-" * 80)

    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    print("5-Fold Cross-Validation Accuracy Scores:")

    for fold_number, score in enumerate(cv_scores, start=1):
        print(f"Fold {fold_number}: {score:.4f}")

    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
    print(f"CV Standard Deviation: {cv_scores.std():.4f}")

    # --------------------------------------------------------
    # Train model using complete training dataset
    # --------------------------------------------------------

    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # --------------------------------------------------------
    # Calculate evaluation metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    kappa = cohen_kappa_score(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # Display test metrics
    # --------------------------------------------------------

    print("\nTest Set Evaluation:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Kappa    : {kappa:.4f}")

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["unacc", "acc", "good", "vgood"]
    )

    print("Confusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # Save Confusion Matrix Plot
    # --------------------------------------------------------

    plt.figure(figsize=(7, 6))

    plt.imshow(cm)

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.xticks(
        range(4),
        ["unacc", "acc", "good", "vgood"]
    )

    plt.yticks(
        range(4),
        ["unacc", "acc", "good", "vgood"]
    )

    # Display values inside matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.colorbar()
    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "")
        + "_confusion_matrix.png"
    )

    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "Model": model_name,
        "Fold 1": cv_scores[0],
        "Fold 2": cv_scores[1],
        "Fold 3": cv_scores[2],
        "Fold 4": cv_scores[3],
        "Fold 5": cv_scores[4],
        "CV Mean": cv_scores.mean(),
        "CV Std": cv_scores.std(),
        "Test Accuracy": accuracy,
        "Test Precision": precision,
        "Test Recall": recall,
        "Test F1": f1,
        "Test Kappa": kappa
    })

# ------------------------------------------------------------
# Create final model comparison table
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("MODEL COMPARISON SUMMARY")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ------------------------------------------------------------
# Save results to CSV
# ------------------------------------------------------------

results_df.to_csv(
    "part1_model_comparison_results.csv",
    index=False
)

# ------------------------------------------------------------
# Determine the best model
# ------------------------------------------------------------

best_model_row = results_df.loc[
    results_df["CV Mean"].idxmax()
]

print("\n" + "=" * 80)
print("BEST MODEL BASED ON CROSS-VALIDATION")
print("=" * 80)

print(f"Best Model: {best_model_row['Model']}")
print(f"CV Mean Accuracy: {best_model_row['CV Mean']:.4f}")
print(f"CV Standard Deviation: {best_model_row['CV Std']:.4f}")
print(f"Test Accuracy: {best_model_row['Test Accuracy']:.4f}")
print(f"Test Precision: {best_model_row['Test Precision']:.4f}")
print(f"Test Recall: {best_model_row['Test Recall']:.4f}")
print(f"Test F1-Score: {best_model_row['Test F1']:.4f}")
print(f"Test Kappa: {best_model_row['Test Kappa']:.4f}")

# ------------------------------------------------------------
# Save comparison bar chart
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    results_df["Model"],
    results_df["CV Mean"]
)

plt.title("Model Comparison - Mean Cross-Validation Accuracy")
plt.xlabel("Model")
plt.ylabel("Mean Cross-Validation Accuracy")
plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "part1_model_comparison_cv_accuracy.png",
    dpi=300
)

plt.show()
plt.close()

print("\nModel experimentation completed successfully.")
print("Results saved to: part1_model_comparison_results.csv")

# ============================================================
# 21. kNN HYPERPARAMETER OPTIMIZATION USING GRIDSEARCHCV
# ============================================================

print("\n" + "=" * 80)
print("21. kNN HYPERPARAMETER OPTIMIZATION USING GRIDSEARCHCV")
print("=" * 80)

# ------------------------------------------------------------
# Create kNN pipeline
# ------------------------------------------------------------

knn_grid_pipeline = Pipeline([
    ("encoder", OrdinalEncoder(
        categories=ordered_categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )),
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())
])

# ------------------------------------------------------------
# Define hyperparameter grid
# ------------------------------------------------------------

param_grid = {
    "model__n_neighbors": [3, 5, 7, 9, 11, 15],
    "model__weights": ["uniform", "distance"],
    "model__metric": ["euclidean", "manhattan", "minkowski"]
}

# ------------------------------------------------------------
# Create GridSearchCV
# ------------------------------------------------------------

grid_search = GridSearchCV(
    estimator=knn_grid_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    return_train_score=True
)

# ------------------------------------------------------------
# Run GridSearchCV
# ------------------------------------------------------------

print("\nStarting GridSearchCV...")
print("Total parameter combinations: 36")
print("Cross-validation folds: 5")
print("Scoring metric: Accuracy")

grid_search.fit(X_train, y_train)

# ------------------------------------------------------------
# Display best parameters
# ------------------------------------------------------------

print("\n" + "-" * 80)
print("BEST HYPERPARAMETERS")
print("-" * 80)

print("Best Parameters:")
print(grid_search.best_params_)

print(f"\nBest Cross-Validation Accuracy: "
      f"{grid_search.best_score_:.4f}")

# ------------------------------------------------------------
# Display all parameter combinations
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("ALL GRIDSEARCHCV RESULTS")
print("=" * 80)

grid_results = pd.DataFrame(
    grid_search.cv_results_
)

grid_results_display = grid_results[
    [
        "rank_test_score",
        "param_model__n_neighbors",
        "param_model__weights",
        "param_model__metric",
        "mean_test_score",
        "std_test_score",
        "mean_train_score"
    ]
].sort_values("rank_test_score")

print(
    grid_results_display.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ------------------------------------------------------------
# Save ALL GridSearchCV results
# ------------------------------------------------------------

grid_results_display.to_csv(
    "part1_knn_gridsearch_all_results.csv",
    index=False
)

print(
    "\nAll GridSearchCV results saved to: "
    "part1_knn_gridsearch_all_results.csv"
)

# ------------------------------------------------------------
# Best model
# ------------------------------------------------------------

best_knn = grid_search.best_estimator_

# ------------------------------------------------------------
# Evaluate optimized model on test data
# ------------------------------------------------------------

y_pred_tuned = best_knn.predict(X_test)

# ------------------------------------------------------------
# Calculate metrics
# ------------------------------------------------------------

tuned_accuracy = accuracy_score(
    y_test,
    y_pred_tuned
)

tuned_precision = precision_score(
    y_test,
    y_pred_tuned,
    average="weighted",
    zero_division=0
)

tuned_recall = recall_score(
    y_test,
    y_pred_tuned,
    average="weighted",
    zero_division=0
)

tuned_f1 = f1_score(
    y_test,
    y_pred_tuned,
    average="weighted",
    zero_division=0
)

tuned_kappa = cohen_kappa_score(
    y_test,
    y_pred_tuned
)

# ------------------------------------------------------------
# Display optimized model results
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("OPTIMIZED kNN TEST SET EVALUATION")
print("=" * 80)

print(f"Accuracy : {tuned_accuracy:.4f}")
print(f"Precision: {tuned_precision:.4f}")
print(f"Recall   : {tuned_recall:.4f}")
print(f"F1-Score : {tuned_f1:.4f}")
print(f"Kappa    : {tuned_kappa:.4f}")

# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_tuned,
        zero_division=0
    )
)

# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------

tuned_cm = confusion_matrix(
    y_test,
    y_pred_tuned,
    labels=["unacc", "acc", "good", "vgood"]
)

print("Confusion Matrix:")
print(tuned_cm)

# ------------------------------------------------------------
# Plot optimized confusion matrix
# ------------------------------------------------------------

plt.figure(figsize=(7, 6))

plt.imshow(tuned_cm)

plt.title("Confusion Matrix - Optimized kNN")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(4),
    ["unacc", "acc", "good", "vgood"]
)

plt.yticks(
    range(4),
    ["unacc", "acc", "good", "vgood"]
)

for i in range(tuned_cm.shape[0]):
    for j in range(tuned_cm.shape[1]):
        plt.text(
            j,
            i,
            tuned_cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()

plt.savefig(
    "part1_optimized_knn_confusion_matrix.png",
    dpi=300
)

plt.show()
plt.close()

# ------------------------------------------------------------
# Compare baseline and optimized model
# ------------------------------------------------------------

baseline_row = results_df[
    results_df["Model"] == "kNN"
].iloc[0]

baseline_accuracy = baseline_row["Test Accuracy"]
baseline_precision = baseline_row["Test Precision"]
baseline_recall = baseline_row["Test Recall"]
baseline_f1 = baseline_row["Test F1"]
baseline_kappa = baseline_row["Test Kappa"]

print("\n" + "=" * 80)
print("BASELINE vs OPTIMIZED kNN")
print("=" * 80)

print(
    f"{'Metric':<20}"
    f"{'Baseline':>15}"
    f"{'Optimized':>15}"
    f"{'Change':>15}"
)

print("-" * 65)

print(
    f"{'Accuracy':<20}"
    f"{baseline_accuracy:>15.4f}"
    f"{tuned_accuracy:>15.4f}"
    f"{tuned_accuracy - baseline_accuracy:>15.4f}"
)

print(
    f"{'Precision':<20}"
    f"{baseline_precision:>15.4f}"
    f"{tuned_precision:>15.4f}"
    f"{tuned_precision - baseline_precision:>15.4f}"
)

print(
    f"{'Recall':<20}"
    f"{baseline_recall:>15.4f}"
    f"{tuned_recall:>15.4f}"
    f"{tuned_recall - baseline_recall:>15.4f}"
)

print(
    f"{'F1-Score':<20}"
    f"{baseline_f1:>15.4f}"
    f"{tuned_f1:>15.4f}"
    f"{tuned_f1 - baseline_f1:>15.4f}"
)

print(
    f"{'Kappa':<20}"
    f"{baseline_kappa:>15.4f}"
    f"{tuned_kappa:>15.4f}"
    f"{tuned_kappa - baseline_kappa:>15.4f}"
)

# ------------------------------------------------------------
# Save baseline vs optimized results
# ------------------------------------------------------------

optimization_results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "Kappa"
    ],
    "Baseline": [
        baseline_accuracy,
        baseline_precision,
        baseline_recall,
        baseline_f1,
        baseline_kappa
    ],
    "Optimized": [
        tuned_accuracy,
        tuned_precision,
        tuned_recall,
        tuned_f1,
        tuned_kappa
    ]
})

optimization_results["Change"] = (
    optimization_results["Optimized"]
    - optimization_results["Baseline"]
)

optimization_results.to_csv(
    "part1_knn_baseline_vs_optimized.csv",
    index=False
)

print("\nOptimization completed successfully.")
print("Best model parameters identified using GridSearchCV.")

# ============================================================
# 22. BEST kNN CONFIGURATION - INDIVIDUAL FOLD RESULTS
# ============================================================

print("\n" + "=" * 80)
print("22. BEST kNN CONFIGURATION - INDIVIDUAL FOLD RESULTS")
print("=" * 80)

# Find the row corresponding to the best GridSearchCV result
best_index = grid_search.best_index_

best_result = grid_search.cv_results_

print("\nBest Configuration:")
print(grid_search.best_params_)

print("\nIndividual Cross-Validation Fold Scores:")

fold_scores = []

for fold_number in range(5):

    score = best_result[
        f"split{fold_number}_test_score"
    ][best_index]

    fold_scores.append(score)

    print(
        f"Fold {fold_number + 1}: {score:.4f}"
    )

# ------------------------------------------------------------
# Calculate mean and standard deviation
# ------------------------------------------------------------

fold_mean = np.mean(fold_scores)
fold_std = np.std(fold_scores)

print(f"\nMean CV Accuracy: {fold_mean:.4f}")
print(f"CV Standard Deviation: {fold_std:.4f}")

# ------------------------------------------------------------
# Compare with GridSearchCV reported values
# ------------------------------------------------------------

print("\nGridSearchCV Reported Values:")
print(
    f"Mean Test Score: "
    f"{best_result['mean_test_score'][best_index]:.4f}"
)

print(
    f"Standard Deviation: "
    f"{best_result['std_test_score'][best_index]:.4f}"
)

# ------------------------------------------------------------
# Verify results
# ------------------------------------------------------------

print("\nVerification:")

if np.isclose(
    fold_mean,
    best_result["mean_test_score"][best_index]
):
    print("Mean CV accuracy verified successfully.")

if np.isclose(
    fold_std,
    best_result["std_test_score"][best_index]
):
    print("CV standard deviation verified successfully.")

# ------------------------------------------------------------
# Save fold results
# ------------------------------------------------------------

best_fold_results = pd.DataFrame({
    "Fold": [
        "Fold 1",
        "Fold 2",
        "Fold 3",
        "Fold 4",
        "Fold 5"
    ],
    "Accuracy": fold_scores
})

best_fold_results.loc[len(best_fold_results)] = [
    "Mean",
    fold_mean
]

best_fold_results.loc[len(best_fold_results)] = [
    "Standard Deviation",
    fold_std
]

best_fold_results.to_csv(
    "part1_best_knn_fold_results.csv",
    index=False
)

print(
    "\nBest model fold results saved to: "
    "part1_best_knn_fold_results.csv"
)

# ============================================================
# 23. BASELINE VS OPTIMIZED kNN PERFORMANCE VISUALIZATION
# ============================================================

print("\n" + "=" * 80)
print("23. BASELINE VS OPTIMIZED kNN PERFORMANCE VISUALIZATION")
print("=" * 80)

# ------------------------------------------------------------
# Create performance comparison data
# ------------------------------------------------------------

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "Kappa"
]

baseline_values = [
    baseline_accuracy,
    baseline_precision,
    baseline_recall,
    baseline_f1,
    baseline_kappa
]

optimized_values = [
    tuned_accuracy,
    tuned_precision,
    tuned_recall,
    tuned_f1,
    tuned_kappa
]

# ------------------------------------------------------------
# Create DataFrame
# ------------------------------------------------------------

performance_comparison = pd.DataFrame({
    "Metric": metrics,
    "Baseline kNN": baseline_values,
    "Optimized kNN": optimized_values
})

print("\nPerformance Comparison:")
print(
    performance_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ------------------------------------------------------------
# Create bar chart
# ------------------------------------------------------------

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    baseline_values,
    width,
    label="Baseline kNN"
)

plt.bar(
    x + width / 2,
    optimized_values,
    width,
    label="Optimized kNN"
)

plt.title("Baseline vs Optimized kNN Performance")
plt.xlabel("Evaluation Metric")
plt.ylabel("Score")

plt.xticks(
    x,
    metrics
)

plt.ylim(0, 1.05)

plt.legend()

plt.tight_layout()

plt.savefig(
    "part1_baseline_vs_optimized_knn.png",
    dpi=300
)

plt.show()
plt.close()

# ------------------------------------------------------------
# Save comparison data
# ------------------------------------------------------------

performance_comparison.to_csv(
    "part1_baseline_vs_optimized_performance.csv",
    index=False
)

print(
    "\nPerformance comparison saved to: "
    "part1_baseline_vs_optimized_performance.csv"
)

print(
    "Performance plot saved to: "
    "part1_baseline_vs_optimized_knn.png"
)


