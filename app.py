import json
import os
import tempfile
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

from src.audio_features import (
    extract_mfcc,
    SAMPLE_RATE,
    DURATION_SECONDS,
)

# 1. Page configuration
st.set_page_config(
    page_title="GENREation",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. Custom styling
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .brand-subtitle {
        color: #A78BFA;
        font-size: 1rem;
        font-weight: 500;
        margin-top: -12px;
        margin-bottom: 12px;
    }
    .brand-caption {
        color: #9CA3AF;
        font-size: 0.85rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(139, 92, 246, 0.08);
        border: 1px solid rgba(139, 92, 246, 0.25);
        padding: 16px;
        border-radius: 12px;
    }
    div[data-testid="stFileUploader"] {
        border-radius: 12px;
    }
    .footer {
        color: #9CA3AF;
        text-align: center;
        font-size: 0.8rem;
        padding-top: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3. App branding
st.title("🎵 GENREation")
st.markdown(
    '<p class="brand-subtitle">'
    "Intelligent Music Genre Recognition"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <p style="font-size: 16px; margin-top: 12px; margin-bottom: 0;">
        Discover the sound. Identify the genre.
    </p>
    <p style="font-size: 16px; margin-top: 4px;">
        Upload a music clip and let AI classify its musical style.
    </p>
    """,
    unsafe_allow_html=True
)
st.divider()

# 4. Model paths
MODEL_PATH = Path("artifacts/music_genre_cnn.keras")
LABELS_PATH = Path("artifacts/labels.json")
SUPPORTED_FORMATS = ["wav", "mp3", "ogg", "flac", "au"]

# 5. Load trained model
@st.cache_resource
def load_classifier():
    """Load the trained CNN and its genre labels."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run train.py before starting the application."
        )
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Genre labels not found: {LABELS_PATH}."
        )
    
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        genres = json.load(file)["genres"]
    if len(genres) != int(model.output_shape[-1]):
        raise ValueError(
            "The number of genre labels does not match "
            "the model's output classes."
        )
    return model, genres

try:
    model, genres = load_classifier()
except Exception as exc:
    st.error(f"Unable to load the classifier: {exc}")
    st.info(
        "Check that the trained model and labels.json "
        "are available inside the artifacts folder."
    )
    st.stop()

# 6. Audio upload
st.subheader("Analyze Your Music")
uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=SUPPORTED_FORMATS,
    help="Supported formats: WAV, MP3, OGG, FLAC and AU.",
)
if uploaded_file is not None:
    st.success(f"File selected: {uploaded_file.name}")
    st.audio(uploaded_file)
show_spectrogram = st.checkbox(
    "Show log-mel spectrogram",
    value=True,
)

# 7. Prediction pipeline
if uploaded_file is not None:
    st.audio(uploaded_file)
    suffix = Path(uploaded_file.name).suffix or ".wav"
    temp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name
        with st.spinner("Analyzing audio with GENREation..."):
            features = extract_mfcc(temp_path)
            model_input = np.expand_dims(features, axis=0)
            probabilities = model.predict(
                model_input,
                verbose=0,
            )[0]
        if not np.all(np.isfinite(probabilities)):
            raise ValueError(
                "The model returned invalid prediction values."
            )
        if len(probabilities) != len(genres):
            raise ValueError(
                "Prediction output does not match genre labels."
            )
        ranked_indices = np.argsort(probabilities)[::-1]
        best_index = int(ranked_indices[0])
        predicted_genre = genres[best_index].replace(
            "_", " "
        ).title()
        confidence = float(probabilities[best_index])

        # 8. Display prediction
        st.divider()
        prediction_col, chart_col = st.columns(
            [1, 1],
            gap="large",
        )
        with prediction_col:
            st.subheader("Prediction")
            st.metric(
                label="Predicted Genre",
                value=predicted_genre,
            )
            st.metric(
                label="Model Confidence",
                value=f"{confidence:.1%}",
            )
            st.caption(
                "Confidence is the model's estimated probability, "
                "not a guarantee that the prediction is correct."
            )

        # 9. Top-three predictions
        with chart_col:
            st.subheader("Top 3 Predictions")
            top_indices = ranked_indices[:min(3, len(genres))]
            chart_labels = [
                genres[int(index)].replace("_", " ").title()
                for index in top_indices
            ]
            chart_values = [
                float(probabilities[index]) * 100
                for index in top_indices
            ]
            st.bar_chart(
                {
                    "Confidence (%)": dict(
                        zip(chart_labels, chart_values)
                    )
                },
                y_label="Confidence (%)",
            )
            for rank, index in enumerate(top_indices, start=1):
                genre_name = genres[int(index)].replace(
                    "_", " "
                ).title()
                st.write(
                    f"**{rank}. {genre_name}** — "
                    f"{probabilities[index]:.1%}"
                )

        # 10. Spectrogram visualization
        if show_spectrogram:
            st.divider()
            st.subheader("Log-Mel Spectrogram")
            y, sr = librosa.load(
                temp_path,
                sr=SAMPLE_RATE,
                mono=True,
                duration=DURATION_SECONDS,
            )
            if y.size == 0:
                st.warning(
                    "The audio could not be visualized."
                )
            else:
                mel_spectrogram = librosa.feature.melspectrogram(
                    y=y,
                    sr=sr,
                    n_fft=2048,
                    hop_length=512,
                    n_mels=128,
                )
                mel_db = librosa.power_to_db(
                    mel_spectrogram,
                    ref=np.max,
                )
                fig, ax = plt.subplots(figsize=(12, 4))
                image = librosa.display.specshow(
                    mel_db,
                    sr=sr,
                    hop_length=512,
                    x_axis="time",
                    y_axis="mel",
                    ax=ax,
                    cmap="magma",
                )
                ax.set_title("Log-Mel Spectrogram")
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Frequency (Hz)")
                fig.colorbar(
                    image,
                    ax=ax,
                    format="%+2.0f dB",
                )
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # 11. Prediction interpretation
        with st.expander("About this prediction"):
            st.write(
                f"The model classified **{predicted_genre}** "
                f"as the most likely genre for "
                f"**{uploaded_file.name}**."
            )
            st.write(
                "The CNN analyzes MFCC features extracted from "
                "the audio. Predictions may be less reliable "
                "for mixed genres, noisy recordings, short clips, "
                "or music unlike the training dataset."
            )
            
    except Exception as exc:
        st.error(
            "Unable to analyze this audio file. "
            "Please try another supported audio file."
        )
        st.caption(f"Technical details: {exc}")

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
            
else:
    st.info(
        "Upload an audio file above to discover its predicted genre."
    )
    st.markdown("### What you can explore")
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    with feature_col1:
        st.markdown("**Genre Recognition**")
        st.caption(
            "Predict one of ten music genres using a trained CNN."
        )
    with feature_col2:
        st.markdown("**Confidence Analysis**")
        st.caption(
            "Compare the model's top genre predictions."
        )
    with feature_col3:
        st.markdown("**Audio Visualization**")
        st.caption(
            "Explore the frequency patterns in your music."
        )

# 12. Footer
st.divider()
st.markdown(
    """
    <div class="footer">
        GENREation · CNN-Based Music Genre Classification<br>
        Built with Python, TensorFlow, Librosa and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)