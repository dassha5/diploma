import os
import time

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv("gestures_data.csv", header=None)

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

print("Загальна кількість записів:", len(X))
print("Кількість ознак:", X.shape[1])
print("Кількість класів:", len(set(y)))



X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nНавчальна вибірка:", len(X_train_full))
print("Тестова вибірка:", len(X_test))


train_percentages = [20, 40, 60, 80, 100]

results = []


for percentage in train_percentages:

    print("\n")
    print(f"Навчання на {percentage}% даних")
    

    train_size = int(len(X_train_full) * percentage / 100)

    if percentage == 100:

        X_train = X_train_full
        y_train = y_train_full

    else:

        X_train, _, y_train, _ = train_test_split(
            X_train_full,
            y_train_full,
            train_size=train_size,
            random_state=42,
            stratify=y_train_full
        )

    print("Кількість навчальних записів:", len(X_train))



    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)



    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


    start_time = time.time()

    model.fit(X_train_scaled, y_train)

    training_time = time.time() - start_time



    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(
        y_test,
        predictions
    )


    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy (%): {accuracy * 100:.2f}%")
    print(f"Час навчання: {training_time:.2f} секунд")



    results.append({
        "Training data (%)": percentage,
        "Training samples": len(X_train),
        "Accuracy": accuracy,
        "Accuracy (%)": accuracy * 100,
        "Training time (sec)": training_time
    })


results_df = pd.DataFrame(results)

print("\n")
print("РЕЗУЛЬТАТИ ЕКСПЕРИМЕНТУ")
print("\n")

print(
    results_df.to_string(index=False)
)


os.makedirs("analysis_results", exist_ok=True)

results_df.to_csv(
    "analysis_results/training_data_experiment.csv",
    index=False
)

fig, axes = plt.subplots(2, 1, figsize=(10, 11))

axes[0].plot(
    results_df["Training data (%)"],
    results_df["Accuracy (%)"],
    marker="o"
)


axes[0].set_ylabel("Точність класифікації (%)")

axes[0].set_title(
    "Залежність точності Random Forest від обсягу навчальної вибірки"
)

axes[0].set_xticks(train_percentages)
axes[0].grid(True)

axes[1].plot(
    results_df["Training data (%)"],
    results_df["Training time (sec)"],
    marker="o"
)

axes[1].set_xlabel("Обсяг навчальної вибірки (%)")
axes[1].set_ylabel("Час навчання (с)")

axes[1].set_title(
    "Залежність часу навчання Random Forest від обсягу даних"
)

axes[1].set_xticks(train_percentages)
axes[1].grid(True)


plt.tight_layout()

plt.savefig(
    "analysis_results/training_data_experiment.png",
    dpi=300
)

plt.show()

print("\nЕксперимент завершено.")
print("Результати збережено у папці analysis_results/")
