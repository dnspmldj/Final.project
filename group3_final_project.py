import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model

@st.cache(allow_output_mutation=True)
def load_fashion_model():
    model = tf.keras.models.load_model('saved_fashion.h5')
    return model

def import_and_predict(image_data, model):
    size = (64, 64)
    image = ImageOps.fit(image_data, size)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

def load_image(filename):
    img = Image.open(filename).resize((64, 64))
    img = img_to_array(img)
    img = img / 255.0
    img = np.reshape(img, (1, 64, 64, 3))
    return img

model = load_fashion_model()

st.write("""# Fashion Dataset by Group 3""")
file = st.file_uploader("Choose photo from computer", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    image_array = load_image(file)
    prediction = import_and_predict(image_array, model)
    
    class_names = ['T-shirt', 'Top', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    result_class = np.argmax(prediction)
    result_label = class_names[result_class]
    string = f"Prediction: {result_label} ({prediction[0][result_class]:.2%} confidence)"
    st.success(string)
