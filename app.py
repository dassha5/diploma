import os
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
import time

st.set_page_config(
    page_title="Sign Language Translator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .big-font { font-size: 80px !important; font-weight: bold; color: #FF4B4B; text-align: center; }
    .status-text { font-size: 24px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

ukr_labels = {
    'HI': 'Привіт 👋', 'LIKE': 'Добре 👍', 'DISLIKE': 'Погано 👎',
    'OK': 'Окей 👌', 'STOP': 'Стоп ✋', 'I': 'Я', 'YOU': 'Ти',
    'HE_SHE': 'Він/Вона', 'ONE': 'Один (1)', 'TWO': 'Два (2)',
    'ROCK': 'Рок 🤘', 'PEACE': 'Мир️', 'PHONE': 'Телефон 🤙',
    'HEART': 'Серце ❤️', 'MONEY': 'Гроші 💸'
}

@st.cache_resource
def load_resources():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'gesture_model.pkl')
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler'], mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, model_complexity=0
    ), mp.solutions.drawing_utils

model, scaler, hands, mp_drawing = load_resources()
mp_hands = mp.solutions.hands

with st.sidebar:
    st.title("Як користуватися")
    st.info("""
    1.  **Увімкніть камеру** галочкою під вікном.
    2.  **Покажіть руку** так, щоб її було повністю видно.
    3.  **Тримайте жест** протягом 1-2 секунд для стабілізації.
    4.  **Зміна теми:** Натисніть '⋮' (справа зверху) -> Settings -> Theme.
    """)
    st.divider()
    st.success("**Порада:** Для кращого розпізнавання використовуйте однотонний фон.")

st.title("Інтелектуальний перекладач жестової мови")

col1, col2 = st.columns([1.5, 1])

with col1:
    FRAME_WINDOW = st.image([]) 
    run = st.checkbox('Запустити камеру', value=True)

with col2:
    st.markdown("<p class='status-text'>Результат розпізнавання:</p>", unsafe_allow_html=True)
    result_placeholder = st.empty()
    
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0

last_prediction = None
prediction_count = 0
FRAME_THRESHOLD = 3  

while run:
    ret, frame = camera.read()
    if not ret: break

    frame_count += 1
    if frame_count % 2 != 0: continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    display_label = "Руку не знайдено"
    label_color = "grey"

    if results.multi_hand_landmarks:
        display_label = "Розпізнавання..." 
        label_color = "#FF4B4B"
        
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        features = []
        base = hand_landmarks.landmark[0]
        for lm in hand_landmarks.landmark:
            features.extend([lm.x - base.x, lm.y - base.y, lm.z - base.z])

        if len(features) == 63:
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0]
            

            if prediction == last_prediction:
                prediction_count += 1
            else:
                prediction_count = 1
                last_prediction = prediction

            if prediction_count >= FRAME_THRESHOLD:
                display_label = ukr_labels.get(prediction, prediction)

    if display_label == "Руку не знайдено":
        result_placeholder.markdown(f"<p class='big-font' style='color: grey; font-size: 40px;'>{display_label}</p>", unsafe_allow_html=True)
    elif display_label == "Розпізнавання...":
         result_placeholder.markdown(f"<p class='big-font' style='color: orange; font-size: 40px;'>{display_label}</p>", unsafe_allow_html=True)
    else:
        result_placeholder.markdown(f"<p class='big-font'>{display_label}</p>", unsafe_allow_html=True)

    FRAME_WINDOW.image(frame_rgb)
    time.sleep(0.01)
camera.release()
