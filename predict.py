import numpy as np
import tensorflow as tf
from keras.models import load_model

model = tf.keras.models.load_model('animals_classification.pb')
print("Model loaded")
image_path = 'v_data/test/cats/image_0_2018.jpeg'

img = tf.keras.utils.load_img(
    image_path, target_size=(224, 224)
)
print('image fetched')
img = tf.keras.utils.load_img(
    image_path, target_size=(224, 224)
)
print('image fetched')

img = np.array(img)
img = img / 255.0
img = img.reshape(1,224,224,3)
label = model.predict(img)
print("Predicted Class (0 - Cats , 1- Elephants): ", label[0][0])
