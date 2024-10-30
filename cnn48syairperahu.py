import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras import layers, models

# Load CSV file
data = pd.read_csv('syairperahucnn/catet.csv')

# Define image size and path prefix

IMG_WIDTH= 48
IMG_HEIGHT= IMG_WIDTH
img_size = (IMG_WIDTH, IMG_HEIGHT)  # Update as per your image size
img_dir = 'syairperahucnn/images/'  # Directory containing images

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
model.add(layers.Dropout(0.20))  # so as to avoid overfitting

# Convolutional Layer 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Convolutional Layer 3
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(0.20))  # so as to avoid overfitting

# Flattening the output
model.add(layers.Flatten())

# Fully connected (dense) layers
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.40))  # says this makes stronger 'regularization'

# output layer
model.add(layers.Dense(40, activation='softmax'))  # 40 distinct classes

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# train the model
history= model.fit(train_dataset, epochs=100, validation_data=test_dataset)
model.save('syairperahu.keras')
# Epoch 100/100
# 113/113 ━━━━━━━━━━━━━━━━━━━━ 2s 15ms/step - accuracy: 0.9864 - loss: 0.0373 - val_accuracy: 0.8030 - val_loss: 1.3989

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4), dpi=300)

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# reevaluating the model accuracy
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy}")
# 13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.0587 - loss: 10.6775 
# Test accuracy: 0.05236907675862312
# may prone to overfitting, but works just okay atm

from keras.models import load_model
model = load_model('syairperahucnn/syairperahu.keras')


#### prediction
# Load and preprocess the image
def preprocess_image(image_path, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    img = load_img(image_path, target_size=img_size, color_mode='grayscale')  # Adjust color_mode if needed
    img = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)
    return img

# make prediction from image
#input_image = preprocess_image('syairperahucnn/images/p01-lineimg0_n0003_label01.png', img_size)
input_image = preprocess_image('mekaten/mekaten_n0000_label03.png', img_size)
input_image = preprocess_image('mekaten/mekaten_n0005_label03.png', img_size)
prediction = model.predict(input_image)
predicted_class = np.argmax(prediction, axis=-1)
print(f'Predicted class: {predicted_class[0]}')

# make random prediction
import random
random_index= random.randint(0, train_images.shape[0])
# draw(train_images[random_index] * 200) 
prediction = model.predict(np.expand_dims(train_images[random_index], axis=0)) # why this call doesn't bode well?
predicted_class = np.argmax(prediction, axis=-1)
print(f'Predicted class: {predicted_class[0]}, label was {train_labels[random_index]}')

