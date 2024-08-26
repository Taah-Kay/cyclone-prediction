import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model (assuming it's saved in the recommended .keras format)
model = tf.keras.models.load_model("trained_model.keras")

def predict_intensity(model, image_array):
    # Make a prediction
    pred = model.predict(image_array)
    return pred

# Streamlit app layout
st.header("Predict Cyclone Satellite Image Windspeed")

# Upload image
st.markdown('''<p style="font-family:sans-serif; color:white; font-size: 20px;">Upload an image </p>''', unsafe_allow_html=True)
file = st.file_uploader("Image", type=["png", "jpg", "jpeg"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption="Your uploaded image", use_column_width=True)

    img_array = np.array(image)
    img = tf.image.resize(img_array, size=(256, 256))
    img = tf.expand_dims(img, axis=0)
    img = img / 255.0

    if st.button('Compute Intensity'):
        intensity = predict_intensity(model, img)
        st.markdown("The intensity of your image in KNOTS is 👇")
        st.success(f"{intensity[0][0]} KNOTS")

        # Display the conclusion based on the predicted intensity
        st.title("Conclusion")
        i = intensity[0][0]  # Assuming intensity is a single value

        if i >= 10 and i <= 40:
            st.subheader("Damage: Minimal")
            st.text("No significant structural damage. Can uproot trees and cause some flooding in coastal areas.")

        elif i > 40 and i <= 70:
            st.subheader("Damage: Moderate")
            st.text("No major destruction to buildings, but can uproot trees and signs. Coastal flooding can occur. Secondary effects can include the shortage of water and electricity.")

        elif i > 70 and i <= 100:
            st.subheader("Damage: Extensive")
            st.text("Structural damage to small buildings and serious coastal flooding to those on low-lying land. Evacuation may be needed.")

        elif i > 100 and i <= 140:
            st.subheader("Damage: Extreme")
            st.text("All signs and trees blown down with extensive damage to roofs. Flat land inland may become flooded. Evacuation probable.")

        else:
            st.subheader("Damage: Catastrophic")
            st.text("Buildings destroyed with small buildings being overturned. All trees and signs blown down. Evacuation of up to 10 miles inland.")

