import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np

# vane sequence
words = [
'222',
'642',
'642',
'741',
'642',
'474',
'474',
'474',
'474',
'7744',
'7744',
'655',
'655',
'5353',
'5353',
'1645',
'1645',
'1642',
'1642',
'4605',
'4605',
'4605',
'1644',
'1644',
'1644',
'4674',
'5574',
'6643',
'7466',
'642',
'4175',
'4175',
'471',
'505',
'1054',
'1054',
'642'
     ]

# Labels (huruf)
labels=[
  'ا'
  , 'ب'
  , 'ت'
  , 'ة'
  , 'ث'
  , 'ج'
  , 'چ'
  , 'ح'
  , 'خ'
  , 'د'
  , 'ذ'
  , 'ر'
  , 'ز'
  , 'س'
  , 'ش'
  , 'ص'
  , 'ض'
  , 'ط'
  , 'ظ'
  , 'ع'
  , 'غ'
  , 'ڠ'
  , 'ف'
  , 'ڤ'
  , 'ق'
  , 'ک'
        , 'ݢ'
  , 'ل'
  , 'م'
  , 'ن'
  , 'و'
  , 'ۏ'
  , 'ه'
  , 'ء'
  , 'ي'
  , 'ی'
  , 'ڽ'
  ]

# Convert labels to numerical format
label_to_id = {label: idx for idx, label in enumerate(set(labels))}
labels = [label_to_id[label] for label in labels]

# Convert words to numerical representation 
(for simplicity, using character indices)
max_len = max(len(word) for word in words)
X = np.zeros((len(words), max_len))
for i, word in enumerate(words):222
    for j, char in enumerate(word):
        X[i, j] = ord(char)

# Convert labels to numpy array
y = np.array(labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(max_len,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(set(labels)), activation='softmax')  # Number of classes is the length of unique labels
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Example prediction
test_word = '222'
# Convert test_word to numerical representation
X_pred = np.zeros((1, max_len))
for j, char in enumerate(test_word):
    X_pred[0, j] = ord(char)
# Predict
predictions = model.predict(X_pred)
predicted_class = np.argmax(predictions)
# Map predicted class index back to label
id_to_label = {v: k for k, v in label_to_id.items()}
predicted_label = id_to_label[predicted_class]
print(f'The word "{test_word}" is classified as "{predicted_label}".')