import os
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
from gtts import gTTS
import base64
import io


st.set_page_config(
    page_title="Sign Language Translator",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .big-font {
        font-size: 80px !important;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0px;
    }

    .status-text {
        font-size: 24px;
        text-align: center;
        margin-top: 20px;
    }

    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        font-size: 18px;
        background-color: #FF4B4B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


ukr_labels = {
    'HI': 'Привіт 👋',
    'LIKE': 'Добре 👍',
    'DISLIKE': 'Погано 👎',
    'OK': 'Окей 👌',
    'STOP': 'Стоп ✋',
    'I': 'Я',
    'YOU': 'Ти',
    'HE_SHE': 'Він/Вона',
    'ONE': 'Один',
    'TWO': 'Два',
    'ROCK': 'Рок 🤘',
    'PEACE': 'Мир',
    'PHONE': 'Телефон 🤙',
    'HEART': 'Серце ❤️',
    'MONEY': 'Гроші 💸'
}


def speak_text(text):
    if text and text not in ["Руку не знайдено", "Розпізнавання..."]:
        try:
            clean_text = ''.join(
                c for c in text
                if c.isalnum() or c.isspace()
            )

            tts = gTTS(
                text=clean_text,
                lang='uk'
            )

            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)

            b64 = base64.b64encode(
                fp.read()
            ).decode()

            unique_id = int(time.time())

            audio_html = f'''
                <audio autoplay="true" key="{unique_id}">
                    <source
                        src="data:audio/mp3;base64,{b64}"
                        type="audio/mp3">
                </audio>
            '''

            st.components.v1.html(
                audio_html,
                height=0
            )

        except Exception as e:
            st.error(
                f"Помилка озвучки: {e}"
            )


@st.cache_resource
def load_resources():

    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    model_path = os.path.join(
        current_dir,
        'gesture_model.pkl'
    )

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    return (
        data['model'],
        data['scaler'],

        mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            model_complexity=0
        ),

        mp.solutions.drawing_utils
    )


model, scaler, hands, mp_drawing = load_resources()

mp_hands = mp.solutions.hands


st.title(
    "Інтелектуальна система розпізнавання жестів"
)


col1, col2 = st.columns([1.5, 1])


with st.sidebar:

    st.title("Керування")

    run = st.checkbox(
        'Запустити камеру',
        value=True
    )

    st.markdown(
        "### Інструкція користування"
    )

    st.markdown("""
    1. Увімкніть камеру.  
    2. Покажіть один жест перед камерою.  
    3. Тримайте руку нерухомо 1–2 секунди для точного розпізнавання.  
    4. Перегляньте результат у правій частині екрана.  
    5. Натисніть кнопку «Озвучити результат», щоб почути назву жесту.  
    """)

    st.markdown("### Поради")

    st.markdown("""
    - використовуйте достатнє освітлення;
    - тримайте кисть у межах кадру;
    - розташовуйте руку ближче до центру зображення;
    - не показуйте кілька жестів одночасно;
    - не рухайте рукою занадто швидко.
    """)

    st.markdown("### Додатково")

    st.info(
        "У налаштуваннях Streamlit можна змінити тему "
        "інтерфейсу на світлу або темну."
    )


with col1:
    FRAME_WINDOW = st.image([])


with col2:

    st.markdown(
        "<p class='status-text'>"
        "Результат розпізнавання:"
        "</p>",
        unsafe_allow_html=True
    )

    result_placeholder = st.empty()

    st.write("---")

    if st.button(
        "🔊 Озвучити результат"
    ):

        if 'last_detected' in st.session_state:

            speak_text(
                st.session_state.last_detected
            )

        else:

            st.warning(
                "Жест ще не розпізнано"
            )


if run:

    camera = cv2.VideoCapture(0)

    last_prediction = None
    prediction_count = 0

    FRAME_THRESHOLD = 5

    while run:

        ret, frame = camera.read()

        if not ret:
            break

        frame = cv2.flip(
            frame,
            1
        )

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(
            frame_rgb
        )

        current_display = "Руку не знайдено"


        if results.multi_hand_landmarks:

            current_display = "Розпізнавання..."

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )

            mp_drawing.draw_landmarks(
                frame_rgb,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


            features = []

            base = hand_landmarks.landmark[0]


            for lm in hand_landmarks.landmark:

                features.extend([
                    lm.x - base.x,
                    lm.y - base.y,
                    lm.z - base.z
                ])


            if len(features) == 63:

                features_scaled = scaler.transform(
                    [features]
                )

                prediction = model.predict(
                    features_scaled
                )[0]


                if prediction == last_prediction:

                    prediction_count += 1

                else:

                    prediction_count = 1
                    last_prediction = prediction


                if prediction_count >= FRAME_THRESHOLD:

                    current_display = ukr_labels.get(
                        prediction,
                        prediction
                    )

                    st.session_state.last_detected = (
                        current_display
                    )


        if current_display == "Руку не знайдено":

            result_placeholder.markdown(
                f"""
                <p class='big-font'
                   style='color: grey; font-size: 40px;'>
                    {current_display}
                </p>
                """,
                unsafe_allow_html=True
            )


        elif current_display == "Розпізнавання...":

            result_placeholder.markdown(
                f"""
                <p class='big-font'
                   style='color: orange; font-size: 40px;'>
                    {current_display}
                </p>
                """,
                unsafe_allow_html=True
            )


        else:

            result_placeholder.markdown(
                f"""
                <p class='big-font'>
                    {current_display}
                </p>
                """,
                unsafe_allow_html=True
            )


        FRAME_WINDOW.image(
            frame_rgb
        )


    camera.release()


else:

    st.warning(
        "Камеру вимкнено"
    )
