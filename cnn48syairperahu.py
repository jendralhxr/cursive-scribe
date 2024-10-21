import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras import layers, models


# Load CSV file
data = pd.read_csv('catet.csv')

# Define image size and path prefix

IMG_WIDTH= 48
IMG_HEIGHT= IMG_WIDTH

img_size = (IMG_WIDTH, IMG_HEIGHT)  # Update as per your image size
img_dir = 'images/'  # Directory containing images

# Load and preprocess images
def load_image(file_path):
    img = load_img(img_dir + file_path, target_size=img_size, color_mode='grayscale')
    img = img_to_array(img) / 255.0  # Normalize to [0, 1]
    return img

# Load the images and labels into numpy arrays
images = np.array([load_image(f) for f in data['filename']])
labels = np.array(data['label'])

# Reshape images if necessary (if you have grayscale images)
images = images.reshape((-1, IMG_WIDTH, IMG_HEIGHT, 1))

# Split into training and test sets
from sklearn.model_selection import train_test_split
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.1)

# Prepare the dataset using tf.data
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

# Normalize and reshape images
train_images = train_images.reshape((train_images.shape[0], IMG_WIDTH, IMG_HEIGHT, 1)) / 255.0
test_images = test_images.reshape((test_images.shape[0], IMG_WIDTH, IMG_HEIGHT, 1)) / 255.0


# Create the CNN model
model = models.Sequential()

# Convolutional Layer 1
model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # Grayscale input
model.add(layers.MaxPooling2D((2, 2)))

# Convolutional Layer 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Convolutional Layer 3
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flattening the output
model.add(layers.Flatten())

# Fully connected (dense) layers
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(40, activation='softmax'))  # 40 distinct classes

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# train the model
model.fit(train_dataset, epochs=10, validation_data=test_dataset)


## prediction

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load and preprocess the image
def preprocess_image(image_path, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    img = load_img(image_path, target_size=img_size, color_mode='grayscale')  # Adjust color_mode if needed
    img = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)
    return img

# Load the image
input_image = preprocess_image('images/p01-lineimg0_n0003_label01.png', img_size)

# Make a prediction
prediction = model.predict(input_image)

# Get the predicted class
predicted_class = np.argmax(prediction, axis=-1)
print(f'Predicted class: {predicted_class[0]}')
