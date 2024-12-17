import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

# load models
@st.cache_resource
def load_rnn_model():
    model = load_model(r"C:\Users\dess7\OneDrive\Робочий стіл\module16hw\models\fashion_mnist_cnn.h5")  
    return model

@st.cache_resource
def load_vgg16_model():
    model = load_model(r"C:\Users\dess7\OneDrive\Робочий стіл\module16hw\models\model_vgg16_last.h5")  
    return model

# Function for preporccess
def preprocess_image_rnn(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(28, 28), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalisation
    img_array = np.expand_dims(img_array, axis=0)  # extension for model
    return img, img_array


def preprocess_image_vgg16(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(32, 32))  # load image
    img_array = image.img_to_array(img)  # to NumPy
    if img_array.shape[-1] == 1:  # If pic in gray shapes
        img_array = np.concatenate([img_array] * 3, axis=-1)  # Double channels for RGB
    img_array = np.expand_dims(img_array, axis=0)  # Add meausure
    img_array = img_array / 255.0  # Normalisation
    return img, img_array


class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# model_load
rnn_model = load_rnn_model()
vgg16_model = load_vgg16_model()

# Frontend Streamlit
st.title("Image Classification App")
st.sidebar.header("Settings")

# Model pick
model_option = st.sidebar.selectbox("Select Model", ["RNN", "VGG16"])
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    if model_option == "RNN":
        input_image, processed_image = preprocess_image_rnn(uploaded_file)
        model = rnn_model
    else:
        input_image, processed_image = preprocess_image_vgg16(uploaded_file)
        model = vgg16_model
    
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]

    st.subheader("Input Image")
    st.image(input_image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Prediction Results")
    st.write(f"**Predicted Class:** {class_names[predicted_class]}")

    st.write("### Class Probabilities")
    fig, ax = plt.subplots()
    ax.bar(class_names, predictions[0])
    plt.xticks(rotation=45)
    st.pyplot(fig)
