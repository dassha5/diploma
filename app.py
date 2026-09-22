import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
import io
import base64
import threading
import time
import av

from gtts import gTTS

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoHTMLAttributes,
    VideoProcessorBase,
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
        2. Дозвольте доступ до камери.
        3. Покажіть руку в камеру.
        4. Утримуйте жест.
        5. Система визначить жест.
        6. Натисніть **🔊 Озвучити результат**.

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
    st.write("🎞️ Камера: 15 FPS")
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
# ЗАВАНТАЖЕННЯ МОДЕЛІ
# ============================================================

try:

    model, scaler = load_model()

except Exception as e:

    st.error(
        f"❌ Помилка завантаження моделі: {e}"
    )

    st.stop()


# ============================================================
# VIDEO PROCESSOR
#
# ВАЖЛИВО:
#
# recv() НЕ запускає MediaPipe.
#
# recv() тільки:
#
# 1. отримує кадр;
# 2. зберігає найсвіжіший кадр;
# 3. одразу повертає кадр у браузер.
#
# MediaPipe працює в окремому worker.
# ============================================================

class GestureProcessor(VideoProcessorBase):

    def __init__(self):

        # ----------------------------------------------------
        # LOCK
        # ----------------------------------------------------

        self.lock = threading.Lock()

        # ----------------------------------------------------
        # Найсвіжіший кадр
        # ----------------------------------------------------

        self.latest_frame = None

        # ----------------------------------------------------
        # Стан
        # ----------------------------------------------------

        self.result = "Руку не знайдено"

        self.last_prediction = None

        self.prediction_count = 0

        # ----------------------------------------------------
        # Сигнал завершення
        # ----------------------------------------------------

        self.stop_event = threading.Event()

        # ----------------------------------------------------
        # MediaPipe
        #
        # Створюємо окремий екземпляр для цього processor.
        # ----------------------------------------------------

        self.mp_hands = mp.solutions.hands

        self.mp_drawing = (
            mp.solutions.drawing_utils
        )

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,
        )

        # ----------------------------------------------------
        # Worker
        # ----------------------------------------------------

        self.worker = threading.Thread(
            target=self.process_frames,
            daemon=True,
        )

        self.worker.start()


    # ========================================================
    # RECEIVE FRAME
    # ========================================================

    def recv(self, frame):

        # ----------------------------------------------------
        # Отримуємо кадр
        # ----------------------------------------------------

        image = frame.to_ndarray(
            format="bgr24"
        )

        # ----------------------------------------------------
        # Дзеркальне відображення
        # ----------------------------------------------------

        image = cv2.flip(
            image,
            1
        )

        # ----------------------------------------------------
        # Передаємо НАЙСВІЖІШИЙ кадр worker-у
        #
        # Тут НЕ запускається MediaPipe.
        # ----------------------------------------------------

        with self.lock:

            self.latest_frame = image.copy()

        # ----------------------------------------------------
        # ДУЖЕ ВАЖЛИВО:
        #
        # Повертаємо кадр одразу.
        #
        # Відео НЕ чекає MediaPipe.
        # ----------------------------------------------------

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )


    # ========================================================
    # WORKER
    # ========================================================

    def process_frames(self):

        while not self.stop_event.is_set():

            # ------------------------------------------------
            # Беремо найсвіжіший кадр
            # ------------------------------------------------

            with self.lock:

                if self.latest_frame is None:

                    frame = None

                else:

                    frame = self.latest_frame.copy()

                    # ------------------------------------------------
                    # Важливо:
                    #
                    # після отримання кадру очищаємо його.
                    #
                    # Якщо MediaPipe працює довше,
                    # старі кадри не накопичуються.
                    # ------------------------------------------------

                    self.latest_frame = None

            # ------------------------------------------------
            # Немає нового кадру
            # ------------------------------------------------

            if frame is None:

                time.sleep(0.005)

                continue

            # ------------------------------------------------
            # BGR → RGB
            # ------------------------------------------------

            rgb_image = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            # ------------------------------------------------
            # MEDIAPIPE
            # ------------------------------------------------

            results = self.hands.process(
                rgb_image
            )

            # =================================================
            # РУКА ЗНАЙДЕНА
            # =================================================

            if results.multi_hand_landmarks:

                hand_landmarks = (
                    results.multi_hand_landmarks[0]
                )

                # ------------------------------------------------
                # 21 landmark
                # ------------------------------------------------

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

                # ------------------------------------------------
                # Нормалізація відносно wrist
                # ------------------------------------------------

                wrist = landmarks[0].copy()

                normalized_landmarks = (
                    landmarks - wrist
                )

                # ------------------------------------------------
                # 21 × 3 = 63 features
                # ------------------------------------------------

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
                            scaler.transform(
                                features
                            )
                        )

                    else:

                        features_scaled = features

                    # ------------------------------------------------
                    # RANDOM FOREST
                    # ------------------------------------------------

                    prediction = model.predict(
                        features_scaled
                    )[0]

                    # ------------------------------------------------
                    # INDEX → LABEL
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

                    with self.lock:

                        if (
                            self.last_prediction
                            == predicted_label
                        ):

                            self.prediction_count += 1

                        else:

                            self.last_prediction = (
                                predicted_label
                            )

                            self.prediction_count = 1

                        # ------------------------------------------------
                        # 3 однакові прогнози
                        # ------------------------------------------------

                        if (
                            self.prediction_count
                            >= 3
                        ):

                            self.result = (
                                predicted_label
                            )

                except Exception as e:

                    with self.lock:

                        self.result = (
                            f"Помилка: {e}"
                        )

            # =================================================
            # РУКИ НЕМАЄ
            # =================================================

            else:

                with self.lock:

                    self.result = (
                        "Руку не знайдено"
                    )

                    self.last_prediction = None

                    self.prediction_count = 0


    # ========================================================
    # ОТРИМАННЯ РЕЗУЛЬТАТУ
    # ========================================================

    def get_result(self):

        with self.lock:

            return self.result


    # ========================================================
    # ЗАВЕРШЕННЯ
    # ========================================================

    def on_ended(self):

        self.stop_event.set()

        if self.worker.is_alive():

            self.worker.join(
                timeout=1.0
            )

        self.hands.close()


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

        # ----------------------------------------------------
        # НАША НОВА АРХІТЕКТУРА
        # ----------------------------------------------------

        video_processor_factory=GestureProcessor,

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
        # Без зайвих кнопок
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
        # Тут callback НЕ блокує відео,
        # тому async_processing не потрібен.
        # ----------------------------------------------------

        async_processing=False,
    )


# ============================================================
# РЕЗУЛЬТАТ
# ============================================================

with result_col:

    st.subheader("🎯 Результат")

    result_placeholder = st.empty()


    # --------------------------------------------------------
    # ОНОВЛЕННЯ РЕЗУЛЬТАТУ
    # --------------------------------------------------------

    @st.fragment(
        run_every=0.2
    )
    def update_result():

        processor = ctx.video_processor

        if processor is None:

            result_placeholder.info(
                "Натисніть START"
            )

            return

        current_result = (
            processor.get_result()
        )

        if current_result == "Руку не знайдено":

            result_placeholder.info(
                "🖐️ Руку не знайдено"
            )

        elif current_result.startswith(
            "Помилка"
        ):

            result_placeholder.error(
                current_result
            )

        else:

            result_placeholder.success(
                f"🎯 **{current_result}**"
            )


    update_result()


    # ========================================================
    # ОЗВУЧЕННЯ
    # ========================================================

    st.write("")

    speak_button = st.button(
        "🔊 Озвучити результат",
        use_container_width=True,
    )


    if speak_button:

        processor = ctx.video_processor

        if processor is None:

            st.warning(
                "Спочатку запустіть камеру."
            )

        else:

            current_result = (
                processor.get_result()
            )

            if (
                current_result
                != "Руку не знайдено"
                and not current_result.startswith(
                    "Помилка"
                )
            ):

                try:

                    # ----------------------------------------
                    # Генеруємо українську озвучку
                    # ----------------------------------------

                    tts = gTTS(
                        text=current_result,
                        lang="uk",
                    )

                    audio_buffer = (
                        io.BytesIO()
                    )

                    tts.write_to_fp(
                        audio_buffer
                    )

                    audio_buffer.seek(0)

                    audio_bytes = (
                        audio_buffer.read()
                    )

                    # ----------------------------------------
                    # Base64
                    # ----------------------------------------

                    audio_base64 = (
                        base64.b64encode(
                            audio_bytes
                        ).decode()
                    )

                    # ----------------------------------------
                    # AUDIO
                    # ----------------------------------------

                    st.markdown(
                        f"""
                        <audio controls autoplay>
                            <source
                                src="data:audio/mp3;base64,{audio_base64}"
                                type="audio/mpeg"
                            >
                        </audio>
                        """,
                        unsafe_allow_html=True,
                    )

                except Exception as e:

                    st.error(
                        f"❌ Помилка озвучення: {e}"
                    )

            else:

                st.warning(
                    "Спочатку покажіть жест."
                )



st.divider()

if ctx.state.playing:

    st.success(
        "🟢 Камера працює"
    )

else:

    st.info(
        "🔵 Натисніть START, щоб увімкнути камеру"
    )
