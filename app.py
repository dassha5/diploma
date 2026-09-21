import os
import streamlit as st
import cv2
import mediapipe as mp
import pickle
import time
import base64
import io
import threading

from gtts import gTTS
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoHTMLAttributes
)
import av


# =========================================================
# НАЛАШТУВАННЯ
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

.result-title {
    text-align: center;
    font-size: 24px;
    margin-top: 20px;
}

.result-text {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    margin-top: 50px;
}

.camera-status {
    text-align: center;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# ЖЕСТИ
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
        min_tracking_confidence=0.5,
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
# СТАН РОЗПІЗНАВАННЯ
# =========================================================

state_lock = threading.Lock()

recognition_state = {
    "result": "Руку не знайдено",
    "last_prediction": None,
    "prediction_count": 0
}

FRAME_THRESHOLD = 5


# =========================================================
# ОБРОБКА ВІДЕО
# =========================================================

def video_frame_callback(frame):

    image = frame.to_ndarray(
        format="bgr24"
    )

    # Дзеркальна камера
    image = cv2.flip(
        image,
        1
    )

    # BGR → RGB
    frame_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe
    results = hands.process(
        frame_rgb
    )

    # Якщо руки немає
    if not results.multi_hand_landmarks:

        with state_lock:

            recognition_state["result"] = (
                "Руку не знайдено"
            )

            recognition_state[
                "prediction_count"
            ] = 0

            recognition_state[
                "last_prediction"
            ] = None

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )

    # =====================================================
    # РУКУ ЗНАЙДЕНО
    # =====================================================

    hand_landmarks = (
        results.multi_hand_landmarks[0]
    )

    # Малюємо точки руки
    mp_drawing.draw_landmarks(
        frame_rgb,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS
    )

    # =====================================================
    # 63 ОЗНАКИ
    # =====================================================

    features = []

    base = hand_landmarks.landmark[0]

    for lm in hand_landmarks.landmark:

        features.extend([
            lm.x - base.x,
            lm.y - base.y,
            lm.z - base.z
        ])

    # =====================================================
    # РОЗПІЗНАВАННЯ
    # =====================================================

    if len(features) == 63:

        try:

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
                    ==
                    recognition_state[
                        "last_prediction"
                    ]
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
                    ]
                    >= FRAME_THRESHOLD
                ):

                    recognition_state[
                        "result"
                    ] = ukr_labels.get(
                        prediction,
                        prediction
                    )

        except Exception:

            pass

    # =====================================================
    # ПОВЕРТАЄМО КАДР
    # =====================================================

    output_image = cv2.cvtColor(
        frame_rgb,
        cv2.COLOR_RGB2BGR
    )

    return av.VideoFrame.from_ndarray(
        output_image,
        format="bgr24"
    )


# =========================================================
# ЗАГОЛОВОК
# =========================================================

st.title(
    "🖐️ Інтелектуальна система розпізнавання жестів"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("Керування")

    st.markdown("### Інструкція")

    st.markdown("""
    1. Натисніть **START**.
    2. Дозвольте доступ до камери.
    3. Покажіть один жест.
    4. Тримайте руку нерухомо 1–2 секунди.
    5. Перегляньте результат.
    6. Для завершення натисніть **STOP**.
    """)

    st.markdown("### Поради")

    st.markdown("""
    - достатнє освітлення;
    - рука повністю в кадрі;
    - рука ближче до центру;
    - показуйте один жест;
    - не рухайте рукою занадто швидко.
    """)


# =========================================================
# КОЛОНКИ
# =========================================================

col1, col2 = st.columns(
    [1.6, 1]
)


# =========================================================
# КАМЕРА
# =========================================================

with col1:

    st.markdown("### 📷 Камера")

    ctx = webrtc_streamer(

        key="gesture-recognition",

        mode=WebRtcMode.SENDRECV,

        video_frame_callback=video_frame_callback,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        # Без зайвих кнопок
        media_toggle_controls=False,

        # Без Play / Pause / таймера
        video_html_attrs=VideoHTMLAttributes(
            autoPlay=True,
            controls=False,
            muted=True
        ),

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
# ПРАВА ЧАСТИНА
# =========================================================

with col2:

    st.markdown(
        "<div class='result-title'>"
        "Результат розпізнавання:"
        "</div>",
        unsafe_allow_html=True
    )

    result_placeholder = st.empty()

    st.write("")

    # Кнопка озвучення
    speak_button = st.button(
        "🔊 Озвучити результат",
        use_container_width=True
    )


# =========================================================
# АВТОМАТИЧНЕ ОНОВЛЕННЯ РЕЗУЛЬТАТУ
# =========================================================

@st.fragment(run_every=0.2)
def show_result():

    with state_lock:

        current_result = recognition_state[
            "result"
        ]

    if current_result == "Руку не знайдено":

        result_placeholder.markdown(
            """
            <div class="result-text"
                 style="color: #888;">
                Руку не знайдено
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        result_placeholder.markdown(
            f"""
            <div class="result-text"
                 style="color: #FF4B4B;">
                {current_result}
            </div>
            """,
            unsafe_allow_html=True
        )


show_result()


# =========================================================
# ОЗВУЧЕННЯ
# =========================================================

if speak_button:

    with state_lock:

        text_to_speak = recognition_state[
            "result"
        ]

    if text_to_speak not in [
        "Руку не знайдено",
        "Розпізнавання..."
    ]:

        try:

            clean_text = ''.join(
                c for c in text_to_speak
                if c.isalnum() or c.isspace()
            )

            tts = gTTS(
                text=clean_text,
                lang="uk"
            )

            audio_buffer = io.BytesIO()

            tts.write_to_fp(
                audio_buffer
            )

            audio_buffer.seek(0)

            st.audio(
                audio_buffer,
                format="audio/mp3",
                autoplay=True
            )

        except Exception as e:

            st.error(
                f"Помилка озвучення: {e}"
            )

    else:

        st.warning(
            "Спочатку покажіть жест."
        )


# =========================================================
# СТАТУС
# =========================================================

if ctx.state.playing:

    st.markdown(
        """
        <div class="camera-status">
            🟢 Камера увімкнена
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <div class="camera-status">
            🔵 Натисніть START, щоб увімкнути камеру
        </div>
        """,
        unsafe_allow_html=True
    )
