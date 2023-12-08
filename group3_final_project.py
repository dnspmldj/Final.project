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

# make a prediction for a new image.
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model

def import_and_predict(image_data,model):
    size=(64,64)
    image=ImageOps.fit(image_data,size)
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
  model = load_model('/content/drive/MyDrive/Models/saved_fashion.h5')
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

    string="OUTPUT : "+result[np.argmax(prediction)]
    st.success(string)
