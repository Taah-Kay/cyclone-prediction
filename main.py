import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Path to the model file
#model_path = os.path.join(os.getcwd(), "Trained_model.keras")
model_path = "Trained_model.keras"

# Load the trained model
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    st.error("Model file not found. Please upload the file.")
    st.stop()

def predict_intensity(model, image_array):
    pred = model.predict(image_array)
    return pred

# Streamlit app layout
st.header("Predict Cyclone Satellite Image Windspeed")

st.markdown('''<p style="font-family:sans-serif; color:white; font-size: 20px;">Upload an image </p>''', unsafe_allow_html=True)
file = st.file_uploader("Image", type=["png", "jpg", "jpeg"])

if file is not None:
    
    file_bytes = np.asarray (bytearray(file.read()), dtype = np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Displaying the image
    st.image(opencv_image, channels="BGR")
    st.write(opencv_image.shape)
        
        # Resizing the image
    opencv_image = cv2.resize(opencv_image, (256, 256))
        
        # Convert image to 4 Dimension
    opencv_image.shape = (1, 256, 256, 3)

    if st.button('Compute Intensity'):
        intensity = predict_intensity(model, opencv_image)
        st.markdown("The intensity of your image in KNOTS is 👇")
        st.success(f"{intensity[0][0]} KNOTS")

st.title("Conclusion")

if 'intensity' in locals():
    i = intensity[0][0]
    if i >= 10 and i <= 40:
        st.subheader("Damage: Minimal")
        st.text("No significant structural damage; can uproot trees and cause some flooding in coastal areas.")
    elif i > 40 and i <= 70:
        st.subheader("Damage: Moderate")
        st.text("No major destruction to buildings; can uproot trees and signs. Coastal flooding can occur. Secondary effects include water and electricity shortages.")
    elif i > 70 and i <= 100:
        st.subheader("Damage: Extensive")
        st.text("Structural damage to small buildings and serious coastal flooding to those on low-lying land. Evacuation may be needed.")
    elif i > 100 and i <= 140:
        st.subheader("Damage: Extreme")
        st.text("All signs and trees blown down with extensive damage to roofs. Flooding likely inland. Evacuation probable.")
    else:
        st.subheader("Damage: Catastrophic")
        st.text("Buildings destroyed; all trees and signs blown down. Evacuation of up to 10 miles inland.")
