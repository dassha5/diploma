import cv2
import mediapipe as mp
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- 1. Словник перекладу ---
ukr_labels = {
    'HI': 'Привіт', 'LIKE': 'Добре', 'DISLIKE': 'Погано',
    'OK': 'Окей', 'STOP': 'Стоп', 'I': 'Я', 'YOU': 'Ти',
    'HE_SHE': 'Він/Вона', 'ONE': 'Один (1)', 'TWO': 'Два (2)',
    'ROCK': 'Рок', 'PEACE': 'Мир', 'PHONE': 'Телефон',
    'HEART': 'Серце', 'MONEY': 'Гроші'
}

# --- 2. Завантаження моделі та СКЕЙЛЕРА ---
print("📦 Завантаження інтелектуальної моделі...")
try:
    with open('gesture_model.pkl', 'rb') as f:
        data = pickle.load(f)
        # Тепер ми дістаємо і модель, і скейлер!
        model = data['model']
        scaler = data['scaler'] 
    print("✅ Система завантажена успішно.")
except Exception as e:
    print(f"❌ Помилка завантаження: {e}")
    exit()

# --- 3. Налаштування MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- 4. Шрифт ---
def get_font(size):
    font_paths = ["C:/Windows/Fonts/arial.ttf", "arial.ttf"]
    for path in font_paths:
        try: return ImageFont.truetype(path, size)
        except: continue
    return ImageFont.load_default()

font = get_font(40)

def draw_ukr_text(img, text, position):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- 5. Головний цикл ---
cap = cv2.VideoCapture(0)
last_prediction = None
prediction_count = 0
stable_label = "Очікування..."
FRAME_THRESHOLD = 5 

print("🚀 Система готова! (Q - вихід)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Формування ознак
        features = []
        base = hand_landmarks.landmark[0]
        for lm in hand_landmarks.landmark:
            features.extend([lm.x - base.x, lm.y - base.y, lm.z - base.z])

        # --- ВАЖЛИВА ПРАВКА: ТУТ ТЕПЕР Є SCALER ---
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]

        # Логіка стабілізації
        if prediction == last_prediction:
            prediction_count += 1
        else:
            prediction_count = 1
            last_prediction = prediction

        if prediction_count >= FRAME_THRESHOLD:
            stable_label = ukr_labels.get(prediction, prediction)

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        frame = draw_ukr_text(frame, f"Жест: {stable_label}", (20, 40))
    else:
        stable_label = "Руку не знайдено"
        frame = draw_ukr_text(frame, stable_label, (20, 40))
        prediction_count = 0
        last_prediction = None

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
