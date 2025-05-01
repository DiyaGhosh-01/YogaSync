import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
model = load_model("majorityVoting.h5")

# Yoga pose class names
CLASS_NAMES = [
    'Adho Mukha Svanasana', 'Adho Mukha Vrksasana', 'Alanasana', 'Anjaneyasana',
    'Ardha Chandrasana', 'Ardha Matsyendrasana', 'Ardha Navasana', 'Ardha Pincha Mayurasana',
    'Ashta Chandrasana', 'Baddha Konasana', 'Bakasana', 'Balasana', 'Bitilasana', 'Camatkarasana',
    'Dhanurasana', 'Eka Pada Rajakapotasana', 'Garudasana', 'Halasana', 'Hanumanasana',
    'Malasana', 'Marjaryasana', 'Navasana', 'Padmasana', 'Parsva Virabhadrasana',
    'Parsvottanasana', 'Paschimottanasana', 'Phalakasana', 'Pincha Mayurasana',
    'Salamba Bhujangasana', 'Salamba Sarvangasana', 'Setu Bandha Sarvangasana', 'Sivasana',
    'Supta Kapotasana', 'Trikonasana', 'Upavistha Konasana', 'Urdhva Dhanurasana',
    'Urdhva Mukha Svsnssana', 'Ustrasana', 'Utkatasana', 'Uttanasana',
    'Utthita Hasta Padangusthasana', 'Utthita Parsvakonasana', 'Vasisthasana',
    'Virabhadrasana One', 'Virabhadrasana Three', 'Virabhadrasana Two', 'Vrksasana'
]

# Streamlit App UI
st.set_page_config(page_title="Yoga Pose Classifier", page_icon="🧘", layout="centered")

st.title("🧘 Yoga Pose Classifier")
st.markdown("""
Upload a clear image of a person performing a **yoga pose**,
and the model will predict the pose.
""")

# Uploading image
image_file = st.file_uploader("📷 Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if image_file:
    with st.spinner('Analyzing the pose...'):
        
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess the image
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)

        # Display prediction
        st.success(f"🧘‍♂️ **Predicted Pose:** {CLASS_NAMES[predicted_index]}")
        

else:
    st.info("👆 Upload an image to get started!")

# Footer
st.markdown("---")

