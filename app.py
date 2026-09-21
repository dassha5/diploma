import os
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
import base64
import io
import threading

from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av


# =========================================================
# НАЛАШТУВАННЯ СТОРІНКИ
# =========================================================

st.set_page_config(
    page_title="Sign Language Translator",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CSS
# =========================================================

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


# =========================================================
# НАЗВИ ЖЕСТІВ
# =========================================================

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
    'PEACE': 'Мир ✌️',
    'PHONE': 'Телефон 🤙',
    'HEART': 'Серце ❤️',
    'MONEY': 'Гроші 💸'
}


# =========================================================
# ОЗВУЧКА
# =========================================================

def speak_text(text):

    if text and text not in [
        "Руку не знайдено",
        "Розпізнавання..."
    ]:

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


# =========================================================
# ЗАВАНТАЖЕННЯ МОДЕЛІ
# =========================================================

@st.cache_resource
def load_resources():

    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    model_path = os.path.join(
        current_dir,
        "gesture_model.pkl"
    )

    with open(model_path, "rb") as f:

        data = pickle.load(f)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        model_complexity=0
    )

    return (
        data["model"],
        data["scaler"],
        hands,
        mp.solutions.drawing_utils
    )


model, scaler, hands, mp_drawing = load_resources()

mp_hands = mp.solutions.hands


# =========================================================
# ЗАГОЛОВОК
# =========================================================

st.title(
    "Інтелектуальна система розпізнавання жестів"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("Керування")

    st.markdown("### Інструкція користування")

    st.markdown("""
    1. Натисніть **START** під камерою.
    
    2. Дозвольте браузеру доступ до камери.
    
    3. Покажіть один жест перед камерою.
    
    4. Тримайте руку нерухомо 1–2 секунди.
    
    5. Перегляньте результат на екрані.
    
    6. Натисніть **STOP**, коли завершите роботу.
    """)

    st.markdown("### Поради")

    st.markdown("""
    - використовуйте достатнє освітлення;
    - тримайте кисть у межах кадру;
    - розташовуйте руку ближче до центру;
    - не показуйте кілька жестів одночасно;
    - не рухайте рукою занадто швидко.
    """)

    st.markdown("### Додатково")

    st.info(
        "Для роботи камери браузер повинен мати "
        "дозвіл на доступ до камери."
    )


# =========================================================
# СПІЛЬНИЙ СТАН ДЛЯ CALLBACK
# =========================================================

state_lock = threading.Lock()

recognition_state = {
    "result": "Руку не знайдено",
    "last_prediction": None,
    "prediction_count": 0
}

FRAME_THRESHOLD = 5


# =========================================================
# ОБРОБКА КАДРУ
# =========================================================

def video_frame_callback(frame):

    image = frame.to_ndarray(
        format="bgr24"
    )

    # Дзеркальне відображення
    image = cv2.flip(image, 1)

    # BGR → RGB
    frame_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe
    results = hands.process(
        frame_rgb
    )

    current_display = "Руку не знайдено"

    # =====================================================
    # ЯКЩО РУКУ ЗНАЙДЕНО
    # =====================================================

    if results.multi_hand_landmarks:

        current_display = "Розпізнавання..."

        hand_landmarks = (
            results.multi_hand_landmarks[0]
        )

        # Малюємо точки руки
        mp_drawing.draw_landmarks(
            frame_rgb,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # =================================================
        # ФОРМУВАННЯ 63 ОЗНАК
        # =================================================

        features = []

        base = hand_landmarks.landmark[0]

        for lm in hand_landmarks.landmark:

            features.extend([
                lm.x - base.x,
                lm.y - base.y,
                lm.z - base.z
            ])

        # =================================================
        # РОЗПІЗНАВАННЯ
        # =================================================

        if len(features) == 63:

            features_scaled = scaler.transform(
                [features]
            )

            prediction = model.predict(
                features_scaled
            )[0]

            # =================================================
            # СТАБІЛІЗАЦІЯ
            # =================================================

            with state_lock:

                if (
                    prediction
                    == recognition_state["last_prediction"]
                ):

                    recognition_state[
                        "prediction_count"
                    ] += 1

                else:

                    recognition_state[
                        "prediction_count"
                    ] = 1

                    recognition_state[
                        "last_prediction"
                    ] = prediction

                if (
                    recognition_state[
                        "prediction_count"
                    ] >= FRAME_THRESHOLD
                ):

                    current_display = ukr_labels.get(
                        prediction,
                        prediction
                    )

                    recognition_state[
                        "result"
                    ] = current_display

    # =====================================================
    # ТЕКСТ НА ВІДЕО
    # =====================================================

    if current_display == "Руку не знайдено":

        text_color = (
            180,
            180,
            180
        )

    elif current_display == "Розпізнавання...":

        text_color = (
            0,
            165,
            255
        )

    else:

        text_color = (
            0,
            255,
            0
        )

    # Чорний фон під текст
    cv2.rectangle(
        frame_rgb,
        (10, 10),
        (frame_rgb.shape[1] - 10, 80),
        (0, 0, 0),
        -1
    )

    # Текст результату
    cv2.putText(
        frame_rgb,
        current_display,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        text_color,
        3,
        cv2.LINE_AA
    )

    # RGB → BGR
    output_image = cv2.cvtColor(
        frame_rgb,
        cv2.COLOR_RGB2BGR
    )

    return av.VideoFrame.from_ndarray(
        output_image,
        format="bgr24"
    )


# =========================================================
# ОСНОВНА ЧАСТИНА
# =========================================================

col1, col2 = st.columns(
    [1.5, 1]
)


# =========================================================
# ЛІВА ЧАСТИНА — КАМЕРА
# =========================================================

with col1:

    st.markdown(
        "### 📷 Камера"
    )

    ctx = webrtc_streamer(
        key="gesture-recognition",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        },
        async_processing=True
    )


# =========================================================
# ПРАВА ЧАСТИНА — РЕЗУЛЬТАТ
# =========================================================

with col2:

    st.markdown(
        "<p class='status-text'>"
        "Результат розпізнавання:"
        "</p>",
        unsafe_allow_html=True
    )

    result_placeholder = st.empty()

    st.write("---")

    # Поточний результат
    with state_lock:

        current_result = recognition_state[
            "result"
        ]

    if current_result == "Руку не знайдено":

        result_placeholder.markdown(
            """
            <p class='big-font'
               style='color: grey; font-size: 40px;'>
               Руку не знайдено
            </p>
            """,
            unsafe_allow_html=True
        )

    elif current_result == "Розпізнавання...":

        result_placeholder.markdown(
            """
            <p class='big-font'
               style='color: orange; font-size: 40px;'>
               Розпізнавання...
            </p>
            """,
            unsafe_allow_html=True
        )

    else:

        result_placeholder.markdown(
            f"""
            <p class='big-font'>
            {current_result}
            </p>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # КНОПКА ОЗВУЧУВАННЯ
    # =====================================================

    if st.button(
        "🔊 Озвучити результат"
    ):

        with state_lock:

            result_to_speak = recognition_state[
                "result"
            ]

        if result_to_speak not in [
            "Руку не знайдено",
            "Розпізнавання..."
        ]:

            speak_text(
                result_to_speak
            )

        else:

            st.warning(
                "Жест ще не розпізнано."
            )


if ctx.state.playing:

    st.success(
        "🟢 Камера працює"
    )

else:

    st.info(
        "🔵 Натисніть START, щоб увімкнути камеру."
    )
