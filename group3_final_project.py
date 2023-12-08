import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
from keras.utils import to_categorical
from keras.models import Sequential
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image,ImageOps
import numpy as np

@st.cache_resource
def load_model():
  model=tf.keras.models.load_model('saved_fashion.h5')
  return model
model = load_model()
st.write("""# Fashion Dataset by Group 3""")
file=st.file_uploader("Choose photo from computer",type=["jpg","png"])

def import_and_predict(image_data,model):
    size=(64,64)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Tshirt','Top)','Pullover','Dress','Coat',
                 'Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
