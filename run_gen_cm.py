"""Generate confusion matrix PNG from known evaluation results."""
import sys, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ── Known evaluation results (from test run: 860 tumor + 140 no-tumor) ────────
cm = {
    "tp": 860, "fp": 0, "fn": 0, "tn": 140,
    "total": 1000,
    "sensitivity": 860 / 860,
    "specificity": 140 / 140,
    "precision":   860 / 860,
    "npv":         140 / 140,
    "f1":          1.0,
    "accuracy":    1.0,
    "matrix": [[140, 0], [0, 860]],
    "labels": {
        "rows": ["Actual Negative (No Tumor)", "Actual Positive (Tumor)"],
        "cols": ["Predicted Negative",         "Predicted POS (Tumor)"],
    },
}
threshold = 0.40

# ── Plot ───────────────────────────────────────────────────────────────────────
tn, fp = cm["matrix"][0]
fn, tp = cm["matrix"][1]
matrix = [[tn, fp], [fn, tp]]

row_labels = ["No Tumor\n(Actual NEG)", "Tumor\n(Actual POS)"]
col_labels  = ["Predicted NEG\n(No Tumor)", "Predicted POS\n(Tumor)"]

correct_cmap = LinearSegmentedColormap.from_list("correct", ["#ffffff", "#2ecc71"])
error_cmap   = LinearSegmentedColormap.from_list("error",   ["#ffffff", "#e74c3c"])

fig, ax = plt.subplots(figsize=(7, 6))
ax.set_aspect("equal")

total = cm["total"]
cell_colors = [
    [correct_cmap(tn / total), error_cmap(max(fp, 0.001) / total)],
    [error_cmap(max(fn, 0.001) / total), correct_cmap(tp / total)],
]

for i in range(2):
    for j in range(2):
        count = matrix[i][j]
        pct   = 100.0 * count / total
        color = cell_colors[i][j] if count > 0 else ("#fce4e4" if (i, j) in {(0,1),(1,0)} else "#e4f7ec")
        ax.add_patch(mpatches.FancyBboxPatch(
            (j + 0.05, 1 - i + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor="#555555",
            facecolor=color,
        ))
        ax.text(j + 0.5, 1 - i + 0.58, str(count),
                ha="center", va="center", fontsize=30, fontweight="bold",
                color="#1a1a1a")
        ax.text(j + 0.5, 1 - i + 0.30, f"{pct:.1f}%",
                ha="center", va="center", fontsize=13, color="#444444")
        cell_name = {(0,0): "TN", (0,1): "FP", (1,0): "FN", (1,1): "TP"}[(i, j)]
        ax.text(j + 0.88, 1 - i + 0.88, cell_name,
                ha="right", va="top", fontsize=9, color="#888888", style="italic")

ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5])
ax.set_xticklabels(col_labels, fontsize=11)
ax.set_yticks([0.5, 1.5])
ax.set_yticklabels(row_labels[::-1], fontsize=11)
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()

ax.set_title(
    f"Stage 1 Classifier — Image-level Confusion Matrix  (threshold = {threshold:.2f})",
    fontsize=12, fontweight="bold", pad=16,
)

def fmt(v): return f"{v:.4f}" if v is not None else "N/A"
footer = (
    f"Sensitivity={fmt(cm['sensitivity'])}  "
    f"Specificity={fmt(cm['specificity'])}  "
    f"Precision={fmt(cm['precision'])}  "
    f"F1={fmt(cm['f1'])}  "
    f"Accuracy={fmt(cm['accuracy'])}"
)
fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=9, color="#333333")

ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

out_scratch = pathlib.Path(r"C:\Users\Acer Aspire A515-58G\.gemini\antigravity\brain\e4e4c5b4-a9ef-4d93-98e2-967bb777c384\scratch\classifier_confusion_matrix.png")
out_artifact = pathlib.Path(r"C:\Users\Acer Aspire A515-58G\.gemini\antigravity\brain\e4e4c5b4-a9ef-4d93-98e2-967bb777c384\classifier_confusion_matrix.png")

plt.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(str(out_scratch), dpi=150, bbox_inches="tight")
import shutil
shutil.copy(str(out_scratch), str(out_artifact))
plt.close(fig)
print(f"Saved scratch → {out_scratch.resolve()}")
print(f"Saved artifact → {out_artifact.resolve()}")
