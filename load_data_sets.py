import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import create_model as cm
import matplotlib.pyplot as plt


nb_train_samples =100
nb_validation_samples = 40
epochs = 10

img_width, img_height = 224, 224
batch_size = 16

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory = 'v_data/train',
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(224, 224),
)

print('training set created')

# for data, labels in train_ds.take(1):
#     print(data.shape)
#     print(labels.shape)

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory = 'v_data/test',
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(224, 224),
)
print('test set created')

def process(image, label):
    image = tf.cast(image/255., tf.float32)
    return image, label

train_ds = train_ds.map(process)
test_ds = test_ds.map(process)
 

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

print('test')

model = cm.build_model()
print(model.summary())
history = model.fit_generator(
    train_ds,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=test_ds,
    validation_steps=nb_validation_samples // batch_size)

model.save('animals_classification.pb')

plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='test')
plt.legend
plt.show()
plt.waitforbuttonpress()