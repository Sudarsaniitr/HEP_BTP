import matplotlib.pyplot as plt
import numpy as np

stages = ["W", "N1", "N2", "N3", "REM"]

hrv_auc = [0.776, 0.65, 0.565, 0.716, 0.335]
hep_auc = [0.575, 0.59, 0.608, 0.675, 0.837]
comb_auc = [0.755, 0.651, 0.643, 0.727, 0.824]

x = np.arange(len(stages))
width = 0.25

plt.figure(figsize=(10,6))

plt.bar(x - width, hrv_auc, width, label="HRV")
plt.bar(x, hep_auc, width, label="HEP")
plt.bar(x + width, comb_auc, width, label="Combined")

plt.xticks(x, stages)
plt.ylabel("Accuracy")
plt.title("Stage-wise Model Performance")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.show()