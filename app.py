import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
import io
import threading
import av

from gtts import gTTS

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoHTMLAttributes,
)


# ============================================================
# STREAMLIT
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
# ЗАГОЛОВОК
# ============================================================

st.title("🤟 Автоматичне розпізнавання жестів")

st.write(
    "Розпізнавання статичних жестів у реальному часі"
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
        4. Утримуйте жест.
        5. Система визначить жест.
        6. Для озвучення натисніть
           кнопку нижче результату.

        ---

        **Технології:**

        - OpenCV
        - MediaPipe Hands
        - Random Forest
        - Streamlit
        - WebRTC
        - gTTS
        """
    )

    st.divider()

    st.write("📷 Камера: 480 × 360")
    st.write("🎞️ Частота: 15 FPS")
    st.write("🖐️ Максимум рук: 1")
    st.write("🎯 Стабілізація: 3 кадри")


# ============================================================
# ЗАВАНТАЖЕННЯ МОДЕЛІ
# ============================================================

@st.cache_resource
def load_model():

    with open(
        "gesture_model.pkl",
        "rb"
    ) as file:

        model_data = pickle.load(file)

    if isinstance(model_data, dict):

        model = model_data.get("model")
        scaler = model_data.get("scaler")

    else:

        model = model_data
        scaler = None

    if model is None:

        raise ValueError(
            "Модель не знайдена у gesture_model.pkl"
        )

    return model, scaler


# ============================================================
# MEDIAPIPE
# ============================================================

@st.cache_resource
def load_mediapipe():

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0,
    )

    return (
        hands,
        mp_hands,
        mp_drawing,
    )


# ============================================================
# ІНІЦІАЛІЗАЦІЯ
# ============================================================

try:

    model, scaler = load_model()

    (
        hands,
        mp_hands,
        mp_drawing,
    ) = load_mediapipe()

except Exception as e:

    st.error(
        f"❌ Помилка завантаження: {e}"
    )

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


# ============================================================
# СТАБІЛІЗАЦІЯ
# ============================================================

FRAME_THRESHOLD = 3


# ============================================================
# CALLBACK
# ============================================================

def video_frame_callback(
    frame: av.VideoFrame
) -> av.VideoFrame:

    # --------------------------------------------------------
    # Кадр → NumPy
    # --------------------------------------------------------

    image = frame.to_ndarray(
        format="bgr24"
    )

    # --------------------------------------------------------
    # Дзеркальне відображення
    # --------------------------------------------------------

    image = cv2.flip(
        image,
        1
    )

    # --------------------------------------------------------
    # BGR → RGB
    # --------------------------------------------------------

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # MEDIAPIPE
    #
    # Кожен кадр, який потрапив у callback,
    # обробляється MediaPipe.
    # --------------------------------------------------------

    results = hands.process(
        rgb_image
    )

    # ========================================================
    # РУКА ЗНАЙДЕНА
    # ========================================================

    if results.multi_hand_landmarks:

        hand_landmarks = (
            results.multi_hand_landmarks[0]
        )

        # ----------------------------------------------------
        # Малюємо landmarks
        # ----------------------------------------------------

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
        )

        # ----------------------------------------------------
        # 21 точки
        # ----------------------------------------------------

        landmarks = np.array(
            [
                [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                ]
                for landmark
                in hand_landmarks.landmark
            ],
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Нормалізація відносно зап'ястя
        # ----------------------------------------------------

        wrist = landmarks[0]

        normalized_landmarks = (
            landmarks - wrist
        )

        # ----------------------------------------------------
        # 21 × 3 = 63 ознаки
        # ----------------------------------------------------

        features = (
            normalized_landmarks
            .flatten()
            .reshape(1, -1)
        )

        try:

            # ------------------------------------------------
            # SCALER
            # ------------------------------------------------

            if scaler is not None:

                features_scaled = (
                    scaler.transform(features)
                )

            else:

                features_scaled = features

            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            prediction = model.predict(
                features_scaled
            )[0]

            # ------------------------------------------------
            # ІНДЕКС → НАЗВА ЖЕСТУ
            # ------------------------------------------------

            prediction_index = int(
                prediction
            )

            if (
                0
                <= prediction_index
                < len(ukr_labels)
            ):

                predicted_label = (
                    ukr_labels[
                        prediction_index
                    ]
                )

            else:

                predicted_label = str(
                    prediction
                )

            # ------------------------------------------------
            # СТАБІЛІЗАЦІЯ
            # ------------------------------------------------

            with state_lock:

                if (
                    recognition_state[
                        "last_prediction"
                    ]
                    == predicted_label
                ):

                    recognition_state[
                        "prediction_count"
                    ] += 1

                else:

                    recognition_state[
                        "last_prediction"
                    ] = predicted_label

                    recognition_state[
                        "prediction_count"
                    ] = 1

                # --------------------------------------------
                # ПІДТВЕРДЖЕННЯ
                # --------------------------------------------

                if (
                    recognition_state[
                        "prediction_count"
                    ]
                    >= FRAME_THRESHOLD
                ):

                    recognition_state[
                        "result"
                    ] = predicted_label

        except Exception as e:

            with state_lock:

                recognition_state[
                    "result"
                ] = f"Помилка: {e}"

    # ========================================================
    # РУКИ НЕМАЄ
    # ========================================================

    else:

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

    # ========================================================
    # ПОВЕРТАЄМО КАДР
    # ========================================================

    return av.VideoFrame.from_ndarray(
        image,
        format="bgr24",
    )


# ============================================================
# КОЛОНКИ
# ============================================================

camera_col, result_col = st.columns(
    [2.2, 1]
)


# ============================================================
# КАМЕРА
# ============================================================

with camera_col:

    st.subheader("📷 Камера")

    ctx = webrtc_streamer(

        key="gesture-recognition",

        mode=WebRtcMode.SENDRECV,

        video_frame_callback=(
            video_frame_callback
        ),

        # ----------------------------------------------------
        # КАМЕРА
        # ----------------------------------------------------

        media_stream_constraints={
            "video": {
                "width": {
                    "ideal": 480
                },
                "height": {
                    "ideal": 360
                },
                "frameRate": {
                    "ideal": 15,
                    "max": 15
                },
            },
            "audio": False,
        },

        # ----------------------------------------------------
        # Кнопки камери
        # ----------------------------------------------------

        media_toggle_controls=False,

        # ----------------------------------------------------
        # Відео без controls
        # ----------------------------------------------------

        video_html_attrs=VideoHTMLAttributes(
            autoPlay=True,
            controls=False,
            muted=True,
        ),

        # ----------------------------------------------------
        # STUN
        # ----------------------------------------------------

        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        },

        # ----------------------------------------------------
        # REAL-TIME PROCESSING
        # ----------------------------------------------------

        async_processing=True,
    )


# ============================================================
# РЕЗУЛЬТАТ
# ============================================================

with result_col:

    st.subheader("🎯 Результат")

    result_box = st.empty()


    # --------------------------------------------------------
    # Відображення результату
    # --------------------------------------------------------

    with state_lock:

        current_result = (
            recognition_state["result"]
        )

    result_box.success(
        f"Розпізнаний жест:\n\n"
        f"### {current_result}"
    )


    # --------------------------------------------------------
    # КНОПКА ОЗВУЧЕННЯ
    # --------------------------------------------------------

    st.write("")

    speak_button = st.button(
        "🔊 Озвучити результат",
        use_container_width=True,
    )


    if speak_button:

        with state_lock:

            current_result = (
                recognition_state["result"]
            )

        if (
            current_result
            != "Руку не знайдено"
            and not current_result.startswith(
                "Помилка"
            )
        ):

            try:

                tts = gTTS(
                    text=current_result,
                    lang="uk",
                )

                audio_buffer = io.BytesIO()

                tts.write_to_fp(
                    audio_buffer
                )

                audio_buffer.seek(0)

                st.audio(
                    audio_buffer,
                    format="audio/mp3",
                )

            except Exception as e:

                st.error(
                    f"❌ Помилка озвучення: {e}"
                )

        else:

            st.info(
                "Спочатку покажіть жест."
            )


# ============================================================
# СТАН КАМЕРИ
# ============================================================

st.divider()

if ctx.state.playing:

    st.success(
        "🟢 Камера працює"
    )

else:

    st.info(
        "🔵 Натисніть START, щоб увімкнути камеру"
    )
