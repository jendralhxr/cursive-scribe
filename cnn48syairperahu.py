import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras import layers, models

# Load CSV file
data = pd.read_csv('syairperahucnn/catet.csv')

# Define image size and path prefix

CLASS_NUM=40
IMG_WIDTH= 48
IMG_HEIGHT= IMG_WIDTH
img_size = (IMG_WIDTH, IMG_HEIGHT)  # Update as per your image size
img_dir = 'syairperahucnn/images/'  # Directory containing images

# Load and preprocess images
def load_image(file_path):
    img = load_img(img_dir + file_path, target_size=img_size, color_mode='grayscale')
    img = img_to_array(img) / np.max(img)  # Normalize to [0, 1]
    return img

# Load the images and labels into numpy arrays
images = np.array([load_image(f) for f in data['filename']])
labels = np.array(data['label'])

# Reshape images if necessary (if you have grayscale images)
images = images.reshape((-1, IMG_WIDTH, IMG_HEIGHT, 1))

# Split into training and test sets
# some would suggest to add more data or do augmentation
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
model.add(layers.Dropout(0.3))  # so as to avoid overfitting, some would suggest to be more aggressive

# Convolutional Layer 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))  # so as to avoid overfitting

# Convolutional Layer 3
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(0.3))  # so as to avoid overfitting

# Flattening the output
model.add(layers.Flatten())

# Fully connected (dense) layers
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.6))  # says this makes stronger 'regularization'

# output layer
model.add(layers.Dense(CLASS_NUM, activation='softmax'))  # 40 distinct classes

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# train the model
history= model.fit(train_dataset, epochs=200, validation_data=test_dataset)
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
# random call to test_images doesn't bode well, but it is okay with images
# why keras, why?
random_index= random.randint(0, images.shape[0])
draw(images[random_index] * 200) 
prediction = model.predict(np.expand_dims(images[random_index], axis=0)) 
predicted_class = np.argmax(prediction, axis=-1)
print(f'Predicted class: {predicted_class[0]}, label was {labels[random_index]}')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

true_labels = []
predictions = []

# Iterate through the test dataset
for images, labels in train_dataset:
    preds = model.predict(images)  # Predict probabilities for each class

    # Append true labels and predicted class indices
    true_labels.extend(labels.numpy())  # Directly use integer labels
    predictions.extend(np.argmax(preds, axis=1))  # Get predicted class indices

# Convert lists to numpy arrays for the confusion matrix
true_labels = np.array(true_labels)
predictions = np.array(predictions)

# Generate the confusion matrix
num_classes = 40
cm = np.zeros((num_classes, num_classes), dtype=int)

# Fill the confusion matrix manually
for true, pred in zip(true_labels, predictions):
    cm[true, pred] += 1

# Ensure all classes are represented in the confusion matrix
# The following line is optional; it makes sure that classes with no predictions are displayed
for i in range(num_classes):
    if i not in true_labels:  # If a class is not present in true labels
        cm[i] = np.zeros(num_classes)  # Set that row to zero

# Plot the confusion matrix
plt.figure(figsize=(12, 10), dpi=300)  # Increase figure size for better readability
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=hurf)
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())  # `values_format='d'` to display integer counts
# Adjust tick parameters for better readability
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(fontsize=12)
plt.title('Confusion Matrix', fontsize=16)


