import pandas as pd
import numpy as np
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Створюємо папку для результатів аналізу
os.makedirs("analysis_results", exist_ok=True)

print("🚀 Завантаження датасету для навчання...")
# Завантажуємо дані з CSV
df = pd.read_csv('gestures_data.csv', header=None)

# Розділяємо на ознаки (координати) та мітки (назви жестів)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# --- 1. МАСШТАБУВАННЯ (Scaler) ---
# Це критично важливо для точності
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. РОЗПОДІЛ ДАНИХ ---
# stratify=y гарантує, що в тест потраплять всі жести пропорційно
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"📊 Навчальна вибірка: {len(X_train)} зразків")
print(f"📊 Тестова вибірка: {len(X_test)} зразків")

# --- 3. НАВЧАННЯ МОДЕЛІ ---
print("🌲 Навчання моделі Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,      # Кількість дерев
    max_depth=None,        # Максимальна глибина
    class_weight='balanced', # ВИРІШУЄ ПРОБЛЕМУ 1000 VS 2000 зразків
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# --- 4. ОЦІНКА ТОЧНОСТІ ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ ЗАГАЛЬНА ТОЧНІСТЬ: {accuracy * 100:.2f}%")
print("\n📋 ДЕТАЛЬНИЙ ЗВІТ:")
print(classification_report(y_test, y_pred))

# --- 5. ВІЗУАЛІЗАЦІЯ (Матриця помилок) ---
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Матриця помилок класифікації жестів')
plt.xlabel('Прогноз моделі')
plt.ylabel('Справжній жест')
plt.tight_layout()
plt.savefig('analysis_results/confusion_matrix.png')

# --- 6. ЗБЕРЕЖЕННЯ МОДЕЛІ ТА СКЕЙЛЕРА ---
# Зберігаємо все в один файл .pkl
model_data = {
    'model': model,
    'scaler': scaler,
    'classes': model.classes_
}

with open('gesture_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\n💾 Файл 'gesture_model.pkl' оновлено! Тепер можна запускати камеру.")
