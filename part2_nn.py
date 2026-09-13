# ============================================================
# PART 2 - MULTILAYER PERCEPTRON NEURAL NETWORK
# ============================================================
# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

# ------------------------------------------------------------
# 2. LOAD DIGITS DATASET
# ------------------------------------------------------------

digits = load_digits()

X = digits.data
y = digits.target

print("=" * 70)
print("PART 2 - DIGITS DATASET")
print("=" * 70)
print("\nDataset loaded successfully.")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Target classes: {np.unique(y)}")
print(f"\nFeature data type: {X.dtype}")
print(f"Target data type: {y.dtype}")

# ------------------------------------------------------------
# 3. DATA EXPLORATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA EXPLORATION")
print("=" * 70)

print("\nDataset shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nFirst five feature rows:")
print(X[:5])

print("\nFirst five target values:")
print(y[:5])

print("\nUnique target classes:")
print(np.unique(y))

print("\nClass distribution:")
class_distribution = pd.Series(y).value_counts().sort_index()
print(class_distribution)

print("\nDescriptive statistics:")
print(pd.DataFrame(X).describe())

# ------------------------------------------------------------
# 4. CHECK FOR MISSING VALUES
# ------------------------------------------------------------
print("\nMissing values:")
print(np.isnan(X).sum())

print("\nDuplicate rows:")
print(pd.DataFrame(X).duplicated().sum())

# ------------------------------------------------------------
# 5. VISUALIZE SAMPLE DIGITS
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.ravel()):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(f"Digit: {y[i]}")
    ax.axis("off")

plt.suptitle("Sample Images from the Digits Dataset")
plt.tight_layout()
plt.savefig("part2_sample_digits.png", dpi=300)
plt.show()
plt.close()

# ------------------------------------------------------------
# 6. VISUALIZE CLASS DISTRIBUTION
# -----------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.hist(y,bins=np.arange(-0.5, 10.5, 1),edgecolor="black")

plt.title("Distribution of Digit Classes")
plt.xlabel("Digit")
plt.ylabel("Frequency")
plt.xticks(range(10))

plt.tight_layout()
plt.savefig("part2_class_distribution.png", dpi=300)
plt.show()
plt.close()

# ------------------------------------------------------------
# 7. TRAINING AND VALIDATION SPLIT
# ------------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

print("\n" + "=" * 70)
print("TRAINING AND VALIDATION SPLIT")
print("=" * 70)
print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Validation samples: {X_test.shape[0]}")
print(f"Training features: {X_train.shape[1]}")
print(f"Validation features: {X_test.shape[1]}")
print("\nTraining class distribution:")
print(pd.Series(y_train).value_counts().sort_index())
print("\nValidation class distribution:")
print(pd.Series(y_test).value_counts().sort_index())

# ------------------------------------------------------------
# 8. DATA PREPROCESSING
# ------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 70)
print("DATA PREPROCESSING")
print("=" * 70)
print("\nStandardScaler applied successfully.")
print(f"Scaled training shape: {X_train_scaled.shape}")
print(f"Scaled validation shape: {X_test_scaled.shape}")
print(f"\nScaled training mean: {X_train_scaled.mean():.4f}")
print(f"Scaled training standard deviation: {X_train_scaled.std():.4f}")

# ------------------------------------------------------------
# 9. BASELINE MLP CLASSIFIER
# ------------------------------------------------------------
from sklearn.neural_network import MLPClassifier
mlp_baseline = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    learning_rate="constant",
    max_iter=300,
    batch_size=32,
    random_state=42
)

print("\n" + "=" * 70)
print("BASELINE MLP CLASSIFIER")
print("=" * 70)
print("\nInitial Hyperparameters:")
print(f"Hidden layers: {mlp_baseline.hidden_layer_sizes}")
print(f"Activation: {mlp_baseline.activation}")
print(f"Solver: {mlp_baseline.solver}")
print(f"Learning rate: {mlp_baseline.learning_rate}")
print(f"Maximum iterations: {mlp_baseline.max_iter}")
print(f"Batch size: {mlp_baseline.batch_size}")

# ------------------------------------------------------------
# 10. TRAIN BASELINE MLP
# ------------------------------------------------------------
print("\nTraining baseline MLP...")
mlp_baseline.fit(X_train_scaled,y_train)
print("Baseline MLP training completed.")
print(f"Number of iterations used: {mlp_baseline.n_iter_}")
print(f"Final loss: {mlp_baseline.loss_:.6f}")

# ------------------------------------------------------------
# 11. BASELINE MODEL EVALUATION
# ------------------------------------------------------------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
    classification_report
)

y_pred_baseline = mlp_baseline.predict(X_test_scaled)

baseline_accuracy = accuracy_score(
    y_test,
    y_pred_baseline
)

baseline_precision = precision_score(
    y_test,
    y_pred_baseline,
    average="weighted",
    zero_division=0
)

baseline_recall = recall_score(
    y_test,
    y_pred_baseline,
    average="weighted",
    zero_division=0
)

baseline_f1 = f1_score(
    y_test,
    y_pred_baseline,
    average="weighted",
    zero_division=0
)

baseline_kappa = cohen_kappa_score(
    y_test,
    y_pred_baseline
)

baseline_cm = confusion_matrix(
    y_test,
    y_pred_baseline
)


print("\n" + "=" * 70)
print("BASELINE MLP MODEL EVALUATION")
print("=" * 70)

print(f"\nAccuracy:  {baseline_accuracy:.4f}")
print(f"Precision: {baseline_precision:.4f}")
print(f"Recall:    {baseline_recall:.4f}")
print(f"F1 Score:  {baseline_f1:.4f}")
print(f"Kappa:     {baseline_kappa:.4f}")

print("\nConfusion Matrix:")
print(baseline_cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_baseline,
        digits=4
    )
)

# ------------------------------------------------------------
# 12. BASELINE CONFUSION MATRIX VISUALIZATION
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.imshow(baseline_cm,interpolation="nearest")
plt.title("Baseline MLP Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(range(10))
plt.yticks(range(10))
plt.colorbar()
plt.tight_layout()
plt.savefig("part2_baseline_confusion_matrix.png",dpi=300)
plt.show()
plt.close()

# ------------------------------------------------------------
# 13. K-FOLD CROSS-VALIDATION
# ------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

mlp_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        learning_rate="constant",
        max_iter=300,
        batch_size=32,
        random_state=42
    ))
])

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

cv_scores = cross_val_score(
    mlp_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print("\nIndividual Fold Accuracy Scores:")

for fold_number, score in enumerate(cv_scores, start=1):
    print(f"Fold {fold_number}: {score:.4f}")

print(f"\nMean CV Accuracy: {cv_scores.mean():.4f}")
print(f"CV Standard Deviation: {cv_scores.std():.4f}")

# ------------------------------------------------------------
# 14. MLP HYPERPARAMETER OPTIMIZATION USING GRIDSEARCHCV
# ------------------------------------------------------------

from sklearn.model_selection import GridSearchCV
mlp_grid_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        random_state=42
    ))
])

param_grid = [
    {
        "mlp__hidden_layer_sizes": [(50,), (100,)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__solver": ["adam"],
        "mlp__learning_rate": ["constant", "adaptive"],
        "mlp__max_iter": [300, 500],
        "mlp__batch_size": [32]
    },
    {
        "mlp__hidden_layer_sizes": [(50,), (100,)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__solver": ["sgd"],
        "mlp__learning_rate": ["constant", "adaptive"],
        "mlp__max_iter": [300, 500],
        "mlp__batch_size": [32]
    }
]

print("\n" + "=" * 70)
print("MLP HYPERPARAMETER OPTIMIZATION")
print("=" * 70)

print("\nHyperparameters being tuned:")
print("Hidden layer sizes: (50,), (100,)")
print("Activation functions: relu, tanh")
print("Solvers: adam, sgd")
print("Learning rate strategies: constant, adaptive")
print("Maximum iterations: 300, 500")
print("Batch sizes: 32")
print("Cross-validation folds: 5")
print("Scoring metric: Accuracy")

total_combinations = 32

print(f"\nTotal hyperparameter combinations: {total_combinations}")
print(f"Total model fits: {total_combinations * 5}")

grid_search = GridSearchCV(
    estimator=mlp_grid_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    return_train_score=True
)

print("\nStarting GridSearchCV...")

grid_search.fit(
    X_train,
    y_train
)

print("\nGridSearchCV completed successfully.")

print("\n" + "-" * 70)
print("BEST HYPERPARAMETERS")
print("-" * 70)

print("\nBest Parameters:")
print(grid_search.best_params_)

print(
    f"\nBest Cross-Validation Accuracy: "
    f"{grid_search.best_score_:.4f}"
)

# ============================================================
# 15. OPTIMIZED MLP MODEL
# ============================================================

best_mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    learning_rate="constant",
    max_iter=300,
    batch_size=32,
    random_state=42
)

# Train the optimized model
best_mlp.fit(X_train_scaled, y_train)

print("\n" + "-" * 70)
print("OPTIMIZED MLP MODEL")
print("-" * 70)

print("Hidden layer sizes:", best_mlp.hidden_layer_sizes)
print("Activation:", best_mlp.activation)
print("Solver:", best_mlp.solver)
print("Learning rate:", best_mlp.learning_rate)
print("Maximum iterations:", best_mlp.max_iter)
print("Batch size:", best_mlp.batch_size)
print("Actual iterations:", best_mlp.n_iter_)
print("Final loss:", round(best_mlp.loss_, 6))

# ============================================================
# 16. OPTIMIZED MLP EVALUATION
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix
)

# Make predictions on the validation set
y_pred_optimized = best_mlp.predict(X_test_scaled)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred_optimized)
precision = precision_score(
    y_test,
    y_pred_optimized,
    average="weighted"
)
recall = recall_score(
    y_test,
    y_pred_optimized,
    average="weighted"
)
f1 = f1_score(
    y_test,
    y_pred_optimized,
    average="weighted"
)
kappa = cohen_kappa_score(y_test, y_pred_optimized)

print("\n" + "-" * 70)
print("OPTIMIZED MLP EVALUATION")
print("-" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"Kappa    : {kappa:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_optimized,
        digits=4
    )
)

# Confusion matrix
cm_optimized = confusion_matrix(
    y_test,
    y_pred_optimized
)

print("\nConfusion Matrix:")
print(cm_optimized)

# ============================================================
# STEP 17 - OPTIMIZED MLP CONFUSION MATRIX PLOT
# ============================================================

import matplotlib.pyplot as plt

# Create confusion matrix
cm_optimized = confusion_matrix(
    y_test,
    y_pred_optimized
)

# Plot confusion matrix
plt.figure(figsize=(8, 6))

plt.imshow(cm_optimized)

plt.title("Confusion Matrix - Optimized MLP")
plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")

plt.xticks(range(10), range(10))
plt.yticks(range(10), range(10))

# Display values inside the matrix
for i in range(cm_optimized.shape[0]):
    for j in range(cm_optimized.shape[1]):
        plt.text(
            j,
            i,
            cm_optimized[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()

# Save the plot
plt.savefig(
    "part2_optimized_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()