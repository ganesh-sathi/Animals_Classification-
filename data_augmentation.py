from keras.preprocessing.image import ImageDataGenerator
 
import tensorflow as tf


datagen = ImageDataGenerator(
        rotation_range = 40,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        brightness_range = (0.5, 1.5))

image_path = r"v_data/temp/tiger1.jpg"
save_image_path = r"v_data/test/cats/"

print(save_image_path)

img = tf.keras.utils.load_img(
    image_path, target_size=(224, 224)
)

# Converting the input sample image to an array
x = tf.keras.utils.img_to_array(img)
#x = img_to_array(img)
# Reshaping the input image
x = x.reshape((1, ) + x.shape) 
   
# Generating and saving 5 augmented samples 
# using the above defined parameters. 
i = 0
for batch in datagen.flow(x, batch_size = 1,
                          save_to_dir = save_image_path, 
                          save_prefix ='image', save_format ='jpeg'):
    i += 1
    if i > 20:
        break
print('images created')
 