import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt


with open("gesture_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]

landmark_names = [
    "WRIST",

    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",

    "INDEX_MCP",
    "INDEX_PIP",
    "INDEX_DIP",
    "INDEX_TIP",

    "MIDDLE_MCP",
    "MIDDLE_PIP",
    "MIDDLE_DIP",
    "MIDDLE_TIP",

    "RING_MCP",
    "RING_PIP",
    "RING_DIP",
    "RING_TIP",

    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP"
]

feature_names = []

for landmark in landmark_names:
    feature_names.append(f"{landmark}_x")
    feature_names.append(f"{landmark}_y")
    feature_names.append(f"{landmark}_z")


if not hasattr(model, "feature_importances_"):
    raise ValueError(
        "Завантажена модель не підтримує feature_importances_."
    )

importances = model.feature_importances_


if len(feature_names) != len(importances):
    raise ValueError(
        f"Кількість назв ознак ({len(feature_names)}) "
        f"не відповідає кількості ознак моделі ({len(importances)})."
    )


importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n15 найбільш важливих ознак:\n")

print(
    importance_df.head(15).to_string(index=False)
)

os.makedirs("analysis_results", exist_ok=True)

importance_df.to_csv(
    "analysis_results/feature_importance.csv",
    index=False
)

top_15 = importance_df.head(15).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_15["Feature"],
    top_15["Importance"]
)

plt.xlabel("Важливість ознаки")
plt.ylabel("Ознака")

plt.title(
    "15 найбільш важливих ознак для Random Forest"
)

plt.tight_layout()

plt.savefig(
    "analysis_results/feature_importance_top15.png",
    dpi=300
)

plt.close()

finger_groups = {
    "Thumb": [
        "THUMB_CMC",
        "THUMB_MCP",
        "THUMB_IP",
        "THUMB_TIP"
    ],

    "Index": [
        "INDEX_MCP",
        "INDEX_PIP",
        "INDEX_DIP",
        "INDEX_TIP"
    ],

    "Middle": [
        "MIDDLE_MCP",
        "MIDDLE_PIP",
        "MIDDLE_DIP",
        "MIDDLE_TIP"
    ],

    "Ring": [
        "RING_MCP",
        "RING_PIP",
        "RING_DIP",
        "RING_TIP"
    ],

    "Pinky": [
        "PINKY_MCP",
        "PINKY_PIP",
        "PINKY_DIP",
        "PINKY_TIP"
    ],

    "Wrist": [
        "WRIST"
    ]
}


group_importance = []


for group_name, landmarks in finger_groups.items():

    total_importance = 0

    for landmark in landmarks:

        mask = importance_df["Feature"].str.startswith(
            landmark + "_"
        )

        total_importance += importance_df.loc[
            mask,
            "Importance"
        ].sum()

    group_importance.append({
        "Part": group_name,
        "Importance": total_importance
    })


group_df = pd.DataFrame(group_importance)


group_df = group_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nВажливість частин руки:\n")

print(
    group_df.to_string(index=False)
)

group_df.to_csv(
    "analysis_results/hand_parts_importance.csv",
    index=False
)

plt.figure(figsize=(9, 6))

plt.bar(
    group_df["Part"],
    group_df["Importance"]
)

plt.xlabel("Частина руки")
plt.ylabel("Сумарна важливість")

plt.title(
    "Важливість частин руки для класифікації жестів"
)

plt.tight_layout()

plt.savefig(
    "analysis_results/hand_parts_importance.png",
    dpi=300
)

plt.close()


print("\nАналіз завершено.")
print("Результати збережено у папці analysis_results/")