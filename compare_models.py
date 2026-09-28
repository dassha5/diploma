import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

os.makedirs(
    "analysis_results",
    exist_ok=True
)


print(
    "Завантаження даних та підготовка до тестування..."
)

df = pd.read_csv(
    "gestures_data.csv",
    header=None
)

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values


print(
    f"Загальна кількість записів: {len(X)}"
)

print(
    f"Кількість ознак: {X.shape[1]}"
)

print(
    f"Кількість класів: {len(np.unique(y))}"
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.7,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=y
)


print(
    f"\nНавчальна вибірка: {len(X_train)} записів"
)

print(
    f"Тестова вибірка: {len(X_test)} записів"
)

noise_factor = 0.05

noise = np.random.normal(
    loc=0.0,
    scale=1.0,
    size=X_train.shape
)

X_train_noisy = (
    X_train +
    noise_factor * noise
)


print(
    f"Рівень шуму: {noise_factor}"
)

models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "k-NN": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]),

    "SVM": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            SVC(
                kernel="rbf",
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


summary_data = []


print("\n" + "=" * 65)

print(
    f"{'Алгоритм':<25} | "
    f"{'Середня точність (%)':<20} | "
    f"{'STD (%)':<10}"
)

print("-" * 65)


for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train_noisy,
        y_train,
        cv=kf,
        scoring="accuracy",
        n_jobs=-1
    )

    mean_acc = scores.mean() * 100

    std_acc = scores.std() * 100


    summary_data.append({
        "Model": name,
        "Accuracy": mean_acc,
        "STD": std_acc
    })


    print(
        f"{name:<25} | "
        f"{mean_acc:>18.2f}% | "
        f"{std_acc:>8.2f}%"
    )


print("=" * 65)


results_df = pd.DataFrame(
    summary_data
)


results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)


print(
    "\nПОРІВНЯННЯ АЛГОРИТМІВ:"
)

print(
    results_df.to_string(
        index=False
    )
)

results_df.to_csv(
    "analysis_results/model_comparison.csv",
    index=False
)

plt.figure(
    figsize=(12, 7)
)

sns.set_style(
    "whitegrid"
)


# Виділяємо Random Forest
# окремо на графіку.

palette = [
    "#2ecc71"
    if model_name == "Random Forest"
    else "#bdc3c7"
    for model_name in results_df["Model"]
]


ax = sns.barplot(
    x="Accuracy",
    y="Model",
    data=results_df,
    hue="Model",
    palette=palette,
    legend=False
)


for p in ax.patches:

    width = p.get_width()

    ax.text(
        width + 0.5,
        p.get_y() + p.get_height() / 2,
        f"{width:.2f}%",
        va="center",
        fontsize=11,
        fontweight="bold"
    )


plt.title(
    "Порівняльна характеристика точності алгоритмів",
    fontsize=14,
    pad=15
)

plt.xlabel(
    "Точність (Accuracy, %)",
    fontsize=12
)

plt.ylabel(
    "Метод класифікації",
    fontsize=12
)

plt.xlim(
    0,
    110
)

plt.tight_layout()


plt.savefig(
    "analysis_results/model_comparison_with_labels.png",
    dpi=300
)

plt.close()


best_model_name = "Random Forest"

final_model = models[
    best_model_name
]


print(
    f"\nДетальна оцінка моделі: "
    f"{best_model_name}"
)

final_model.fit(
    X_train_noisy,
    y_train
)

y_pred = final_model.predict(
    X_test
)

print(
    f"\nДетальний звіт для моделі "
    f"{best_model_name}:\n"
)


print(
    classification_report(
        y_test,
        y_pred
    )
)


print(
    "\nАналіз завершено!"
)

print(
    "Результати збережено у папці "
    "'analysis_results'."
)