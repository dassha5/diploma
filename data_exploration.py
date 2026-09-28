import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



os.makedirs("analysis_results", exist_ok=True)

print("ЗАПУСК ПОВНОГО АНАЛІЗУ ДАТАСЕТУ...")


df = pd.read_csv("gestures_data.csv", header=None)

missing_total = df.isnull().sum().sum()

if missing_total > 0:
    print(f"Знайдено пропущених значень: {missing_total}")

    df.fillna(0, inplace=True)

    print("Пропуски заповнені нулями")

else:
    print("Пропусків немає")

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

print("\nОСНОВНА СТАТИСТИКА ДАТАСЕТУ")

print(f"Кількість записів: {df.shape[0]}")
print(f"Кількість ознак: {X.shape[1]}")
print(f"Кількість класів: {y.nunique()}")
print("\nПРИКЛАД ДАНИХ:")
print(df.head())

gesture_counts = y.value_counts()

print("\nРОЗПОДІЛ ЖЕСТІВ:")
print(gesture_counts)

balance_ratio = gesture_counts.min() / gesture_counts.max()

print(f"\nБаланс датасету: {balance_ratio:.2f}")

plt.figure(figsize=(12, 6))

sns.barplot(
    x=gesture_counts.index,
    y=gesture_counts.values
)

plt.title("Розподіл жестів")
plt.xlabel("Жест")
plt.ylabel("Кількість записів")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "analysis_results/class_distribution.png",
    dpi=300
)

plt.close()


feature_stats = X.describe()

feature_stats.to_csv(
    "analysis_results/feature_statistics.csv"
)

plt.figure(figsize=(12, 10))

sns.heatmap(
    X.corr(),
    cmap="coolwarm",
    center=0
)

plt.title("Матриця кореляції")

plt.tight_layout()

plt.savefig(
    "analysis_results/correlation_matrix.png",
    dpi=300
)

plt.close()

plt.figure(figsize=(15, 6))

sns.boxplot(data=X)

plt.title("Розподіл координат")
plt.xlabel("Ознаки")
plt.ylabel("Значення")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    "analysis_results/coordinate_boxplot.png",
    dpi=300
)

plt.close()

print("\nPCA АНАЛІЗ")


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)


pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

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
plt.xlabel("Головна компонента PC1")
plt.ylabel("Головна компонента PC2")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "analysis_results/pca_visualization.png",
    dpi=300
)

plt.close()

explained = pca.explained_variance_ratio_

print(f"PC1: {explained[0] * 100:.2f}%")
print(f"PC2: {explained[1] * 100:.2f}%")

print(
    f"Разом: {explained.sum() * 100:.2f}%"
)


