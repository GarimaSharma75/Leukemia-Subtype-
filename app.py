import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- Page Config ---
st.set_page_config(page_title="Leukemia Subtype Detector", layout="centered")

# --- HEADER Section ---
st.markdown("""
    <div style='background-color:#57068c;padding:20px;border-radius:10px'>
    <h2 style='color:white;text-align:center'>🧬 Leukemia Subtype Detection</h2>
    <p style='color:white;text-align:center'>
        Using an ensemble of deep learning models: <b>DenseNet121</b>, <b>MobileNetV2</b>, <b>VGG16</b>, and <b>Custom CNN</b>.<br>
        Predicts the sub-type : <i>Benign</i>, <i>Pre</i>, <i>Pro</i>, <i>Early</i>.
    </p>
    </div>
""", unsafe_allow_html=True)

# --- Constants ---
IMG_HEIGHT, IMG_WIDTH = 224, 224
CLASS_NAMES = ['Benign', 'Pre', 'Pro', 'Early']
SAVE_DIR = 'saved_leukemia_ensemble'
MODEL_PATHS = {
    "DenseNet121": os.path.join(SAVE_DIR, "DenseNet121_model.keras"),
    "MobileNetV2": os.path.join(SAVE_DIR, "MobileNetV2_model.keras"),
    "VGG16": os.path.join(SAVE_DIR, "VGG16_model.keras"),
    "CustomCNN": os.path.join(SAVE_DIR, "CustomCNN_model.keras")
}
HISTORY_PATHS = {
    "DenseNet121": os.path.join(SAVE_DIR, "DenseNet121_history.pkl"),
    "MobileNetV2": os.path.join(SAVE_DIR, "MobileNetV2_history.pkl"),
    "VGG16": os.path.join(SAVE_DIR, "VGG16_history.pkl"),
    "CustomCNN": os.path.join(SAVE_DIR, "CustomCNN_history.pkl")
}
ENSEMBLE_WEIGHTS = {
    "DenseNet121": 0.28,
    "MobileNetV2": 0.30,
    "VGG16": 0.22,
    "CustomCNN": 0.20
}

# --- Load Models ---
@st.cache_resource
def load_all_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            models[name] = load_model(path, compile=False)
        else:
            st.warning(f"⚠️ {name} model not found at: {path}")
    return models

models = load_all_models()

# --- Upload & Predict Section ---
uploaded_file = st.file_uploader("📁 Upload a blood smear image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    if st.button("🔍 Enter"):
        with st.spinner("⏳ Please wait while results are being computed..."):
            try:
                img = image.load_img(uploaded_file, target_size=(IMG_HEIGHT, IMG_WIDTH))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                st.markdown("### 🧪 Model Predictions (individual):")

                # Columns for side-by-side display
                col1, col2 = st.columns(2)
                individual_preds = {}
                for i, (name, model) in enumerate(models.items()):
                    pred = model.predict(img_array)
                    individual_preds[name] = pred
                    pred_class = CLASS_NAMES[np.argmax(pred)]
                    confidence = float(pred[0][np.argmax(pred)])
                    with [col1, col2][i % 2]:
                        st.info(f"**{name}**: `{pred_class}` ({confidence:.2%})")

                # Ensemble Prediction
                ensemble_pred = sum(
                    ENSEMBLE_WEIGHTS[name] * pred for name, pred in individual_preds.items()
                )
                final_class = CLASS_NAMES[np.argmax(ensemble_pred)]
                final_conf = float(np.max(ensemble_pred))

                st.markdown("---")
                st.markdown(f"""
                    <div style="background-color:#c6f6d5;padding:15px;border-radius:10px">
                    <h4 style="color:#2f855a">✅ Final Ensemble Prediction: <b>{final_class}</b></h4>
                    <p style="font-size:16px;color:#22543d">Confidence: <b>{final_conf:.2%}</b></p>
                    </div>
                """, unsafe_allow_html=True)

                # Confidence Chart
                st.bar_chart({CLASS_NAMES[i]: float(ensemble_pred[0][i]) for i in range(4)})

            except Exception as e:
                st.error(f"⚠️ Error during prediction: {e}")
else:
    st.info("👈 Upload an image to begin prediction.")

# --- Optional Training History ---
st.markdown("---")
st.subheader("📈 Model Training History")

if st.checkbox("Show training curves for each model"):
    for name, path in HISTORY_PATHS.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                history = pickle.load(f)
                acc = history['accuracy']
                val_acc = history['val_accuracy']
                loss = history['loss']
                val_loss = history['val_loss']

                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                ax[0].plot(acc, label='Train Acc')
                ax[0].plot(val_acc, label='Val Acc')
                ax[0].set_title(f'{name} Accuracy')
                ax[0].legend()

                ax[1].plot(loss, label='Train Loss')
                ax[1].plot(val_loss, label='Val Loss')
                ax[1].set_title(f'{name} Loss')
                ax[1].legend()

                st.pyplot(fig)
        else:
            st.warning(f"No history found for {name}.")
