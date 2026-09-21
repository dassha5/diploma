import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

# створення папки для результатів
os.makedirs("analysis_results", exist_ok=True)

print("ЗАПУСК ПОВНОГО АНАЛІЗУ ДАТАСЕТУ...")

# -------------------------------------------------
# ЗАВАНТАЖЕННЯ ДАНИХ
# -------------------------------------------------
df = pd.read_csv("gestures_data.csv", header=None)

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

print("ОСНОВНА СТАТИСТИКА ДАТАСЕТУ")
print(f"Кількість записів: {df.shape[0]}")
print(f"Кількість ознак: {X.shape[1]}")
print(f"Кількість класів: {y.nunique()}")

print("\nПРИКЛАД ДАНИХ:")
print(df.head())

# -------------------------------------------------
# ПЕРЕВІРКА ПРОПУСКІВ
# -------------------------------------------------
missing_total = df.isnull().sum().sum()

if missing_total > 0:
    print(f"Знайдено пропущених значень: {missing_total}")
    df.fillna(0, inplace=True)
    print("Пропуски заповнені нулями")
else:
    print("Пропусків немає")

# -------------------------------------------------
# РОЗПОДІЛ КЛАСІВ
# -------------------------------------------------
gesture_counts = y.value_counts()
print("\nРОЗПОДІЛ ЖЕСТІВ:")
print(gesture_counts)

balance_ratio = gesture_counts.min() / gesture_counts.max()
print(f"\nБаланс датасету: {balance_ratio:.2f}")

# графік
plt.figure(figsize=(12, 6))
sns.barplot(x=gesture_counts.index, y=gesture_counts.values)
plt.title("Розподіл жестів")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("analysis_results/class_distribution.png")
plt.close()

# -------------------------------------------------
# СТАТИСТИКА ОЗНАК
# -------------------------------------------------
feature_stats = X.describe()
feature_stats.to_csv("analysis_results/feature_statistics.csv")

# -------------------------------------------------
# КОРЕЛЯЦІЙНА МАТРИЦЯ
# -------------------------------------------------
plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), cmap="coolwarm", center=0)
plt.title("Матриця кореляції")
plt.tight_layout()
plt.savefig("analysis_results/correlation_matrix.png")
plt.close()

# -------------------------------------------------
# BOXPLOT
# -------------------------------------------------
plt.figure(figsize=(15, 6))
sns.boxplot(data=X)
plt.title("Розподіл координат")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("analysis_results/coordinate_boxplot.png")
plt.close()

# -------------------------------------------------
# PCA АНАЛІЗ
# -------------------------------------------------
print("\n=== PCA АНАЛІЗ ===")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Gesture"] = y.values

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="Gesture",
    data=pca_df,
    alpha=0.5
)
plt.title("PCA візуалізація жестів")
plt.grid(True)
plt.tight_layout()
plt.savefig("analysis_results/pca_visualization.png")
plt.close()

explained = pca.explained_variance_ratio_
print(f"PC1: {explained[0]*100:.2f}%")
print(f"PC2: {explained[1]*100:.2f}%")

# -------------------------------------------------
# ЗАВАНТАЖЕННЯ МОДЕЛІ
# -------------------------------------------------
with open("gesture_model.pkl", "rb") as f:
    saved_data = pickle.load(f)

if isinstance(saved_data, dict):
    model = saved_data.get("model") or saved_data.get("classifier")
else:
    model = saved_data

# -------------------------------------------------
# ПЕРЕВІРКА
# -------------------------------------------------
if not hasattr(model, "feature_importances_"):
    raise ValueError("Модель не підтримує feature_importances_")

importances = model.feature_importances_

# -------------------------------------------------
# НОРМАЛЬНІ НАЗВИ ОЗНАК 🔥
# -------------------------------------------------
landmark_names = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

coords = ["x", "y", "z"]

feature_names = []
for lm in landmark_names:
    for c in coords:
        feature_names.append(f"{lm}_{c}")

# -------------------------------------------------
# ВАЖЛИВІСТЬ ОЗНАК
# -------------------------------------------------
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

top = importance_df.sort_values(by="importance", ascending=False).head(15)

print("\nТОП-15 ОЗНАК:")
print(top)

# графік
plt.figure(figsize=(10, 6))
plt.barh(top["feature"], top["importance"])
plt.gca().invert_yaxis()
plt.title("Важливість ознак (Random Forest)")
plt.tight_layout()
plt.savefig("analysis_results/feature_importance.png")
plt.close()

print("\nГОТОВО 🚀 Результати в папці analysis_results")
