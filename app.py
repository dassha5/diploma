import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
import io
import threading
from gtts import gTTS

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoHTMLAttributes,
)


# ============================================================
# НАЛАШТУВАННЯ СТОРІНКИ
# ============================================================

st.set_page_config(
    page_title="Розпізнавання жестів",
    page_icon="🤟",
    layout="wide",
)


# ============================================================
# НАЗВИ ЖЕСТІВ
# ============================================================

ukr_labels = [
    "Привіт",
    "До побачення",
    "Дякую",
    "Будь ласка",
    "Так",
    "Ні",
    "Я",
    "Ти",
    "Ми",
    "Добре",
    "Погано",
    "Любов",
    "Допомога",
    "Стоп",
    "Перемога",
]


# ============================================================
# СТИЛІ
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 30px 20px;
        border-radius: 15px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        text-align: center;
        margin-top: 20px;
    }

    .result-title {
        font-size: 20px;
        margin-bottom: 10px;
    }

    .result-text {
        font-size: 36px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ЗАГОЛОВОК
# ============================================================

st.markdown(
    '<div class="main-title">🤟 Автоматичне розпізнавання жестів</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Розпізнавання статичних жестів у реальному часі</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("ℹ️ Інструкція")

    st.markdown(
        """
        **Як користуватися системою:**

        1. Натисніть **START**.
        2. Дозвольте браузеру доступ до камери.
        3. Покажіть руку в камеру.
        4. Утримуйте жест декілька кадрів.
        5. Система визначить жест.
        6. За бажанням натисніть
           **«🔊 Озвучити результат»**.

        ---

        **Технології:**

        - OpenCV
        - MediaPipe Hands
        - Random Forest
        - Streamlit
        - Streamlit-WebRTC
        - gTTS
        """
    )

    st.divider()

    st.markdown("### ⚙️ Параметри")

    st.write("📷 Роздільність: **640 × 480**")
    st.write("🎞️ Частота: **15–20 FPS**")
    st.write("🖐️ Максимум рук: **1**")
    st.write("🎯 Стабілізація: **5 кадрів**")


# ============================================================
# ЗАВАНТАЖЕННЯ МОДЕЛІ
# ============================================================

@st.cache_resource
def load_resources():

    with open("gesture_model.pkl", "rb") as file:
        model_data = pickle.load(file)

    # Підтримка словника з model/scaler
    if isinstance(model_data, dict):

        model = model_data.get("model")
        scaler = model_data.get("scaler")

    else:

        model = model_data
        scaler = None

    if model is None:
        raise ValueError("Модель не знайдена у gesture_model.pkl")

    # --------------------------------------------------------
    # MediaPipe Hands
    # --------------------------------------------------------

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0,
    )

    return model, scaler, hands, mp_hands, mp_drawing


# ============================================================
# ЗАВАНТАЖУЄМО РЕСУРСИ
# ============================================================

try:

    model, scaler, hands, mp_hands, mp_drawing = load_resources()

except Exception as e:

    st.error(f"❌ Помилка завантаження моделі: {e}")
    st.stop()


# ============================================================
# СТАН РОЗПІЗНАВАННЯ
# ============================================================

state_lock = threading.Lock()

recognition_state = {
    "result": "Руку не знайдено",
    "last_prediction": None,
    "prediction_count": 0,
}


# Кількість однакових прогнозів для підтвердження жесту
FRAME_THRESHOLD = 5


# ============================================================
# ОБРОБКА КАДРУ
# ============================================================

def video_frame_callback(frame):

    # --------------------------------------------------------
    # Отримуємо кадр
    # --------------------------------------------------------

    image = frame.to_ndarray(format="bgr24")

    # Дзеркальне відображення
    image = cv2.flip(image, 1)

    # --------------------------------------------------------
    # BGR → RGB для MediaPipe
    # --------------------------------------------------------

    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    results = hands.process(frame_rgb)

    # --------------------------------------------------------
    # Якщо рука знайдена
    # --------------------------------------------------------

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        # Малюємо точки та з'єднання руки
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
        )

        # ----------------------------------------------------
        # Отримуємо 21 landmark
        # ----------------------------------------------------

        landmarks = []

        for landmark in hand_landmarks.landmark:

            landmarks.append([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        landmarks = np.array(landmarks, dtype=np.float32)

        # ----------------------------------------------------
        # Нормалізація відносно зап'ястя
        #
        # Точка 0 = wrist
        # ----------------------------------------------------

        wrist = landmarks[0].copy()

        normalized_landmarks = landmarks - wrist

        # ----------------------------------------------------
        # 21 × 3 = 63 ознаки
        # ----------------------------------------------------

        features = normalized_landmarks.flatten().reshape(1, -1)

        # ----------------------------------------------------
        # Масштабування
        # ----------------------------------------------------

        try:

            if scaler is not None:
                features_scaled = scaler.transform(features)

            else:
                features_scaled = features

            # ------------------------------------------------
            # Передбачення моделі
            # ------------------------------------------------

            prediction = model.predict(features_scaled)[0]

            # ------------------------------------------------
            # Отримуємо назву жесту
            # ------------------------------------------------

            try:

                prediction_index = int(prediction)

                if 0 <= prediction_index < len(ukr_labels):
                    predicted_label = ukr_labels[prediction_index]

                else:
                    predicted_label = str(prediction)

            except (ValueError, TypeError):

                predicted_label = str(prediction)

            # ------------------------------------------------
            # СТАБІЛІЗАЦІЯ
            # ------------------------------------------------

            with state_lock:

                if (
                    recognition_state["last_prediction"]
                    == predicted_label
                ):

                    recognition_state["prediction_count"] += 1

                else:

                    recognition_state["last_prediction"] = (
                        predicted_label
                    )

                    recognition_state["prediction_count"] = 1

                # Підтверджуємо жест після FRAME_THRESHOLD
                # однакових прогнозів

                if (
                    recognition_state["prediction_count"]
                    >= FRAME_THRESHOLD
                ):

                    recognition_state["result"] = (
                        predicted_label
                    )

        except Exception as e:

            with state_lock:
                recognition_state["result"] = (
                    f"Помилка: {e}"
                )

    else:

        # ----------------------------------------------------
        # Руки немає
        # ----------------------------------------------------

        with state_lock:

            recognition_state["result"] = (
                "Руку не знайдено"
            )

            recognition_state["last_prediction"] = None
            recognition_state["prediction_count"] = 0

    # --------------------------------------------------------
    # Повертаємо кадр
    # --------------------------------------------------------

    return frame.from_ndarray(
        image,
        format="bgr24",
    )


# ============================================================
# ОСНОВНА ЧАСТИНА
# ============================================================

col1, col2 = st.columns([2.2, 1])


# ============================================================
# ЛІВА КОЛОНКА — КАМЕРА
# ============================================================

with col1:

    st.subheader("📷 Камера")

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
                },
            },
            "audio": False,
        },

        media_toggle_controls=False,

        video_html_attrs=VideoHTMLAttributes(
            autoPlay=True,
            controls=False,
            muted=True,
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

        # Важливо:
        # кожен отриманий кадр обробляється послідовно.
        # Старі кадри не накопичуються у великій черзі.
        async_processing=False,
    )


# ============================================================
# ПРАВА КОЛОНКА — РЕЗУЛЬТАТ
# ============================================================

with col2:

    st.subheader("🎯 Результат")

    result_placeholder = st.empty()


    # --------------------------------------------------------
    # Оновлення результату
    # --------------------------------------------------------

    @st.fragment(run_every=0.2)
    def show_result():

        with state_lock:

            result = recognition_state["result"]

        result_placeholder.markdown(
            f"""
            <div class="result-box">

                <div class="result-title">
                    Розпізнаний жест:
                </div>

                <div class="result-text">
                    {result}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    show_result()


    st.write("")


    # ========================================================
    # ОЗВУЧЕННЯ
    # ========================================================

    st.subheader("🔊 Озвучення")

    speak_button = st.button(
        "🔊 Озвучити результат",
        use_container_width=True,
    )


    if speak_button:

        with state_lock:
            current_result = recognition_state["result"]

        if (
            current_result != "Руку не знайдено"
            and not current_result.startswith("Помилка")
        ):

            try:

                # Генеруємо українську озвучку
                tts = gTTS(
                    text=current_result,
                    lang="uk",
                )

                audio_buffer = io.BytesIO()

                tts.write_to_fp(audio_buffer)

                audio_buffer.seek(0)

                # Показуємо аудіоплеєр
                st.audio(
                    audio_buffer,
                    format="audio/mp3",
                )

            except Exception as e:

                st.error(
                    f"Не вдалося озвучити результат: {e}"
                )

        else:

            st.info(
                "Спочатку потрібно розпізнати жест."
            )


# ============================================================
# СТАН СИСТЕМИ
# ============================================================

st.divider()

if ctx.state.playing:

    st.success(
        "🟢 Камера працює. Покажіть жест."
    )

else:

    st.info(
        "🔵 Натисніть START, щоб увімкнути камеру."
    )
