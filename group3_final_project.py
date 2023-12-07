import streamlit as st
import time
import sys
import pandas as pd
from PIL import Image


@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('saved_fashion.h5')
  return model
  
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(64,64)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction

st.write("""# Fasion""")

file=st.file_uploader("Choose clothes photo from computer",type=["jpg","png"])

st.image(image,use_column_width=True)

st.success(string)



from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
  img = Image.open(filename).resize((224, 224))
  plt.imshow(img)
  plt.show()
  img = load_img(filename, target_size=(28, 28))
  img = img_to_array(img)
  img = img[:,:,0]
  img = img.reshape(1,28, 28, 1)
  img = img.astype('float32')
  img = img / 255.0
  return img

def run_example(filename):
  img = load_image(filename)
  model=tf.keras.models.load_model('saved_fashion.h5')
  result = np.argmax(model.predict(img), axis=1)
  if result == 0:
    print('Tshirt')
  elif result == 1:
    print('Top')
  elif result == 2:
    print('Pullover')
  elif result == 3:
    print('Dress')
  elif result == 4:
    print('Coat')
  elif result == 5:
    print('Sandal')
  elif result == 6:
    print('Shirt')
  elif result == 7:
    print('Snicker')
  elif result == 8:
    print('Bag')
  else:
    print('Ankle Boot')



