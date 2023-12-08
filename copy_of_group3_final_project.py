import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def predict(image):
    # Load the model inside the function to avoid caching issues
    model = tf.keras.models.load_model('saved_fashion.h5')

    size = (64, 64)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

def main():
    st.write("""# Fashion Dataset by Group 3""")
    file = st.file_uploader("Choose photo from computer", type=["jpg", "png"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)

        # Predictions are not cached to prevent UserHashError
        prediction = predict(image)

        class_names = ['Tshirt', 'Top', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
        result_string = "OUTPUT: " + class_names[np.argmax(prediction)]
        st.success(result_string)

if __name__ == "__main__":
    main()
