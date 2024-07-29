import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd

# Generate random data
np.random.seed(42)

# Parameters
num_samples = 1000
max_length = 10  # Max length of the strings
num_classes = 40  # Number of classes

# data
source = pd.read_csv('coba.csv')
random_strings=pd.concat([source['2bfs'], source['2alpha']])
random_labels=pd.concat([source['label'], source['label']])

# Tokenize the strings
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(random_strings)
sequences = tokenizer.texts_to_sequences(random_strings)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

# Convert labels to one-hot encoding
labels = tf.keras.utils.to_categorical(random_labels, num_classes=num_classes)

# Split data into training and testing sets
train_sequences, test_sequences, train_labels, test_labels = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42)

# Parameters for the model
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token
embedding_dim = 50  # Dimension of the embedding vector

# Check shapes
print(f'Train sequences shape: {train_sequences.shape}')
print(f'Test sequences shape: {test_sequences.shape}')
print(f'Train labels shape: {train_labels.shape}')
print(f'Test labels shape: {test_labels.shape}')

# Define the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
epochs = 4000
batch_size = 32
model.fit(train_sequences, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_sequences, test_labels)
print(f'Accuracy on test data: {accuracy}')

pickle.dump(model, open('rasm-lstm.model', 'wb'))
#model= pickle.load(open('rasm-lstm.model', 'rb'))


# Example prediction
def predict(string):
    sequence = tokenizer.texts_to_sequences([string])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length, padding='post')
    predicted_index = np.argmax(model.predict(padded_sequence))
    print(f'Predicted class: {predicted_index}')

# Test the prediction function
predict("222223") # alif



