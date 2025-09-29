import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# ============ CONFIG ==============
st.set_page_config(page_title="Flower Recognition", page_icon="🌸", layout="wide")
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
model = load_model(r'C:\Users\tuyet\Downloads\Flower_recog_Model (1)\Flower_recog_Model\Flower_Recog_Model.h5')

# ============ FUNCTIONS ============
def classify_image(uploaded_file):
    input_image = Image.open(uploaded_file).resize((180,180))
    input_array = tf.keras.utils.img_to_array(input_image)
    input_array = tf.expand_dims(input_array, 0)

    predictions = model.predict(input_array)
    result = tf.nn.softmax(predictions[0])

    # Tạo DataFrame cho dễ hiển thị
    df = pd.DataFrame({
        "Flower": flower_names,
        "Confidence (%)": (result.numpy() * 100).round(2)
    }).sort_values(by="Confidence (%)", ascending=False)

    return df, df.iloc[0]

# ============ SIDEBAR ==============
st.sidebar.header("🌼 Upload your image")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ============ MAIN LAYOUT ============
st.title("🌸 Flower Classification App")
st.markdown("This app uses a **CNN model** to classify images of flowers into 5 categories.")

if uploaded_file is not None:
    # Chia layout 2 cột
    col1, col2 = st.columns([1,2])

    with col1:
        st.image(uploaded_file, caption="Uploaded Image", width=300)

    with col2:
        df, best = classify_image(uploaded_file)

        st.subheader("🔎 Prediction Result")
        st.success(f"The image is **{best['Flower']}** with confidence **{best['Confidence (%)']}%**")

        st.subheader("📊 Confidence Scores")
        st.dataframe(df, use_container_width=True)

        # Bar chart
        st.bar_chart(data=df.set_index("Flower"))
else:
    st.info("⬅️ Please upload an image from the sidebar to start classification.")
