import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np

my_model = load_model('model to predict animals.h5')


image = Image.open('test1.jpeg')
new_image = image.resize((224, 224))
new_image.save('test1.jpeg')
x=cv2.imread('test1.jpeg')
x=np.array(x)
print(my_model.predict_classes(x.reshape(1, 224, 224, 3)))

image = Image.open('test2.jpeg')
new_image = image.resize((224, 224))
new_image.save('test2.jpeg')
x=cv2.imread('test2.jpeg')
x=np.array(x)
print(my_model.predict_classes(x.reshape(1, 224, 224, 3)))

image = Image.open('test3.jpeg')
new_image = image.resize((224, 224))
new_image.save('test3.jpeg')
x=cv2.imread('test3.jpeg')
x=np.array(x)
print(my_model.predict_classes(x.reshape(1, 224, 224, 3)))

image = Image.open('test4.jpeg')
new_image = image.resize((224, 224))
new_image.save('test4.jpeg')
x=cv2.imread('test4.jpeg')
x=np.array(x)
print(my_model.predict_classes(x.reshape(1, 224, 224, 3)))