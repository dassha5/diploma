import os
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

os.makedirs("analysis_results", exist_ok=True)


print("Завантаження датасету для навчання...")

df = pd.read_csv("gestures_data.csv", header=None)


X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

print(f"Загальна кількість зразків: {len(X)}")
print(f"Кількість ознак: {X.shape[1]}")
print(f"Кількість класів: {len(set(y))}")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Навчальна вибірка: {len(X_train)} зразків")
print(f"Тестова вибірка: {len(X_test)} зразків")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("Навчання моделі Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nЗАГАЛЬНА ТОЧНІСТЬ: {accuracy * 100:.2f}%")

print("\nДЕТАЛЬНИЙ ЗВІТ:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Матриця помилок класифікації жестів")
plt.xlabel("Прогноз моделі")
plt.ylabel("Справжній жест")

plt.tight_layout()

plt.savefig(
    "analysis_results/confusion_matrix.png",
    dpi=300
)

plt.show()

model_data = {
    "model": model,
    "scaler": scaler,
    "classes": model.classes_
}

with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model_data, f)


print("\nФайл 'gesture_model.pkl' оновлено!")
print("Модель готова до використання у Streamlit-застосунку.")