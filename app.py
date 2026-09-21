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
    font-size: 52px;
    font-weight: bold;
    margin-top: 55px;
}

.info-text {
    text-align: center;
    font-size: 16px;
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
# СПІЛЬНИЙ СТАН
# =========================================================

state_lock = threading.Lock()

recognition_state = {
    "result": "Руку не знайдено",
    "last_prediction": None,
    "prediction_count": 0
}


# Скільки однакових кадрів потрібно
# для підтвердження жесту
FRAME_THRESHOLD = 5


# Лічильник кадрів
frame_counter = 0


# =========================================================
# ОБРОБКА КАДРУ
# =========================================================

def video_frame_callback(frame):

    global frame_counter

    frame_counter += 1

    # Отримуємо кадр
    image = frame.to_ndarray(
        format="bgr24"
    )

    # Дзеркальна камера
    image = cv2.flip(
        image,
        1
    )

    # -----------------------------------------------------
    # НЕ ОБРОБЛЯЄМО КОЖЕН КАДР
    #
    # Це зменшує навантаження на MediaPipe
    # і модель.
    # -----------------------------------------------------

    if frame_counter % 2 != 0:

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )

    # BGR → RGB
    frame_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # =====================================================
    # MEDIAPIPE
    # =====================================================

    results = hands.process(
        frame_rgb
    )

    # =====================================================
    # РУКИ НЕМАЄ
    # =====================================================

    if not results.multi_hand_landmarks:

        with state_lock:

            recognition_state[
                "result"
            ] = "Руку не знайдено"

            recognition_state[
                "last_prediction"
            ] = None

            recognition_state[
                "prediction_count"
            ] = 0

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

    # Малюємо точки MediaPipe
    mp_drawing.draw_landmarks(
        frame_rgb,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS
    )

    # =====================================================
    # ФОРМУЄМО 63 ОЗНАКИ
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
                        "last_prediction"
                    ] = prediction

                    recognition_state[
                        "prediction_count"
                    ] = 1

                # Підтверджуємо жест
                # після FRAME_THRESHOLD кадрів
                if (
                    recognition_state[
                        "prediction_count"
                    ] >= FRAME_THRESHOLD
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
    2. Дозвольте браузеру доступ до камери.
    3. Покажіть один жест.
    4. Тримайте руку нерухомо 1–2 секунди.
    5. Перегляньте результат.
    6. Для завершення натисніть **STOP**.
    """)

    st.markdown("### Поради")

    st.markdown("""
    - використовуйте достатнє освітлення;
    - тримайте всю кисть у кадрі;
    - розташовуйте руку по центру;
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
            "video": {
                "width": {
                    "ideal": 640
                },
                "height": {
                    "ideal": 480
                },
                "frameRate": {
                    "ideal": 15,
                    "max": 20
                }
            },
            "audio": False
        },

        media_toggle_controls=False,

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

        # ВАЖЛИВО:
        # синхронна обробка не дозволяє
        # накопичувати старі кадри
        async_processing=False
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

    # -----------------------------------------------------
    # ОТРИМУЄМО ПОТОЧНИЙ РЕЗУЛЬТАТ
    # -----------------------------------------------------

    with state_lock:

        current_result = recognition_state[
            "result"
        ]

    # -----------------------------------------------------
    # РЕЗУЛЬТАТ
    # -----------------------------------------------------

    if current_result == "Руку не знайдено":

        result_placeholder.markdown(
            """
            <div class="result-text"
                 style="color:#888;">
                Руку не знайдено
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        result_placeholder.markdown(
            f"""
            <div class="result-text"
                 style="color:#FF4B4B;">
                {current_result}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("---")

    st.markdown(
        "### 🔊 Озвучення"
    )

    st.markdown(
        "<div class='info-text'>"
        "Натисніть кнопку нижче, щоб створити "
        "українську озвучку результату."
        "</div>",
        unsafe_allow_html=True
    )

    speak_button = st.button(
        "🔊 Озвучити результат",
        use_container_width=True
    )


# =========================================================
# ОЗВУЧЕННЯ
# =========================================================

if speak_button:

    with state_lock:

        text_to_speak = recognition_state[
            "result"
        ]

    if text_to_speak == "Руку не знайдено":

        st.warning(
            "Спочатку покажіть жест."
        )

    else:

        try:

            # -------------------------------------------------
            # Прибираємо emoji для gTTS
            # -------------------------------------------------

            clean_text = ''.join(
                c for c in text_to_speak
                if c.isalnum() or c.isspace()
            )

            # -------------------------------------------------
            # Створюємо українську озвучку
            # -------------------------------------------------

            tts = gTTS(
                text=clean_text,
                lang="uk",
                slow=False
            )

            audio_buffer = io.BytesIO()

            tts.write_to_fp(
                audio_buffer
            )

            audio_buffer.seek(0)

            # -------------------------------------------------
            # АУДІО
            # -------------------------------------------------

            audio_bytes = audio_buffer.read()

            audio_b64 = base64.b64encode(
                audio_bytes
            ).decode()

            audio_html = f"""
            <div style="
                display:flex;
                flex-direction:column;
                align-items:center;
                gap:10px;
                margin-top:15px;
            ">

                <audio
                    id="gestureAudio"
                    controls
                    style="width:100%;"
                >
                    <source
                        src="data:audio/mp3;base64,{audio_b64}"
                        type="audio/mpeg"
                    >
                </audio>

                <button
                    onclick="
                        document
                        .getElementById('gestureAudio')
                        .play();
                    "
                    style="
                        background:#FF4B4B;
                        color:white;
                        border:none;
                        border-radius:12px;
                        padding:10px 25px;
                        font-size:16px;
                        cursor:pointer;
                    "
                >
                    ▶️ Прослухати
                </button>

            </div>
            """

            st.components.v1.html(
                audio_html,
                height=100
            )

        except Exception as e:

            st.error(
                f"Помилка озвучення: {e}"
            )

if ctx.state.playing:

    st.success(
        "🟢 Камера увімкнена"
    )

else:

    st.info(
        "🔵 Натисніть START, щоб увімкнути камеру."
    )
