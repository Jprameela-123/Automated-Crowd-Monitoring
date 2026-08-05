import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
# ===============================
# LOAD DATA
# ===============================
print("Loading files...")
y_true = np.load("y_true.npy")
cnn_scores = np.load("cnn_scores.npy")
gru_scores = np.load("gru_scores.npy")
fusion_scores = np.load("fusion_scores.npy")
# ===============================
# ALIGN LENGTH
# ===============================
min_len = min(
    len(y_true),
    len(cnn_scores),
    len(gru_scores),
    len(fusion_scores)
)
y_true = y_true[:min_len]
cnn_scores = cnn_scores[:min_len]
gru_scores = gru_scores[:min_len]
fusion_scores = fusion_scores[:min_len]
# ===============================
# THRESHOLD
# ===============================
threshold = 0.5
cnn_pred = (cnn_scores > threshold).astype(int)
gru_pred = (gru_scores > threshold).astype(int)
fusion_pred = (fusion_scores > threshold).astype(int)
# ===============================
# METRIC FUNCTION
# ===============================
def evaluate(y_true, y_pred):
    return [
        accuracy_score(
            y_true,
            y_pred
        ),
        precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    ]
# ===============================
# CALCULATE METRICS
# ===============================
cnn_metrics = evaluate(
    y_true,
    cnn_pred
)
gru_metrics = evaluate(
    y_true,
    gru_pred
)
fusion_metrics = evaluate(
    y_true,
    fusion_pred
)
# ===============================
# PRINT RESULTS
# ===============================
models = [
    "CNN MODEL",
    "GRU MODEL",
    "FUSION MODEL"
]
metrics = [
    cnn_metrics,
    gru_metrics,
    fusion_metrics
]
print("\nFINAL RESULTS\n")
for name, metric in zip(models, metrics):
    print(name)
    print("Accuracy :", round(metric[0]*100,2))
    print("Precision:", round(metric[1]*100,2))
    print("Recall   :", round(metric[2]*100,2))
    print("F1-score :", round(metric[3]*100,2))
    print()
# ===============================
# SAVE CSV
# ===============================
df = pd.DataFrame({
    "MODEL": models,
    "ACCURACY":
        [m[0]*100 for m in metrics],
    "PRECISION":
        [m[1]*100 for m in metrics],
    "RECALL":
        [m[2]*100 for m in metrics],
    "F1-SCORE":
        [m[3]*100 for m in metrics]
})
df.to_csv(
    "evaluation_results.csv",
    index=False
)
print(df)
# ===============================
# CONFUSION MATRICES
# ===============================
cnn_cm = confusion_matrix(
    y_true,
    cnn_pred
)
gru_cm = confusion_matrix(
    y_true,
    gru_pred
)
fusion_cm = confusion_matrix(
    y_true,
    fusion_pred
)
print("\nCNN Confusion Matrix\n")
print(cnn_cm)
print("\nGRU Confusion Matrix\n")
print(gru_cm)
print("\nFusion Confusion Matrix\n")
print(fusion_cm)
# ===============================
# BAR GRAPH
# ===============================
accuracy = df["ACCURACY"]
f1 = df["F1-SCORE"]
x = np.arange(len(models))
width = 0.30
plt.figure(figsize=(8,5))
bars1 = plt.bar(
    x-width/2,
    accuracy,
    width,
    label="Accuracy"
)
bars2 = plt.bar(
    x+width/2,
    f1,
    width,
    label="F1-score"
)
plt.xticks(
    x,
    models
)
plt.ylabel("Score (%)")
plt.title(
    "Performance Comparison"
)
plt.ylim(0,100)
plt.legend()
for bar in bars1:
    h = bar.get_height()
    plt.text(
        bar.get_x()+bar.get_width()/2,
        h+1,
        f"{h:.2f}",
        ha="center"
    )
for bar in bars2:
    h = bar.get_height()
    plt.text(
        bar.get_x()+bar.get_width()/2,
        h+1,
        f"{h:.2f}",
        ha="center"
    )
plt.show()
print("\nSaved evaluation_results.csv")
