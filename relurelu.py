import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate more random data for demonstration
np.random.seed(42)

# Generate 1000 random 4-digit number strings
random_numbers = np.random.randint(1000, 10000, size=1000).astype(str)

# Assign random letters from A to Z as labels
random_labels = np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), size=1000)

# Convert characters to numeric values (optional preprocessing step)
# No need to convert to float32 if directly processing as strings
numbers = np.array([list(map(int, list(num))) for num in random_numbers])

# Convert labels to numeric indices
label_to_index = {label: i for i, label in enumerate(np.unique(random_labels))}
indices = np.array([label_to_index[label] for label in random_labels])

# Check shapes and types
print(f'Numbers shape: {numbers.shape}, Numbers type: {numbers.dtype}')
print(f'Indices shape: {indices.shape}, Indices type: {indices.dtype}')

# Create TensorFlow Dataset and batch it
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices((numbers, indices)).batch(batch_size)

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),  # Input shape corrected to (4,)
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(label_to_index), activation='softmax')  # Output layer with softmax activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
epochs = 800
model.fit(dataset, epochs=epochs)

# Evaluate the model
loss, accuracy = model.evaluate(dataset)
print(f'Accuracy: {accuracy}')

def predict(num):
    nums= list(map(int, list(num)))
    test_number = np.array([[nums]]).astype(np.float32) / 9999.0  # Normalize test input
    predicted_index = np.argmax(model.predict(test_number))
    predicted_label = list(label_to_index.keys())[predicted_index]
    print(f'Predicted label: {predicted_label}')


# Example prediction
test_number = np.array([[0, 2, 7, 0]]).astype(np.float32) / 9999.0  # Normalize test input
predicted_index = np.argmax(model.predict(test_number))
predicted_label = list(label_to_index.keys())[predicted_index]
print(f'Predicted label: {predicted_label}')


