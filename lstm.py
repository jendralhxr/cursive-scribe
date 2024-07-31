import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd

PHI= 1.6180339887498948482 # ppl says this is a beautiful number :)

# Generate random data
np.random.seed(42)

# Parameters
num_samples = 1000
min_length= 2
max_length = 12  # Max length of the strings
num_classes = 40  # Number of classes
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)

# data
source = pd.read_csv('coba.csv')
random_strings=pd.concat([source['bfs'], source['bfsdfs']])
random_labels=pd.concat([source['val'], source['val']])

# Tokenize the strings
tokenizer.fit_on_texts(random_strings)
sequences = tokenizer.texts_to_sequences(random_strings)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

# Convert labels to one-hot encoding
labels = tf.keras.utils.to_categorical(random_labels, num_classes=num_classes)

# Split data into training and testing sets
train_sequences, test_sequences, train_labels, test_labels = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=38)

# Parameters for the model
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token
embedding_dim = 50  # Dimension of the embedding vector

# Define the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    LSTM(64),
    Dense(64, activation='relu'),
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
#model.fit(train_sequences, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)

def save_variables(filename, *args):
    with open(filename, 'wb') as f:
        pickle.dump(args, f)
def load_variables(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

save_variables('rasm-lstm.model', model, train_sequences, test_sequences, train_labels, test_labels)
model, train_sequences, test_sequences, train_labels, test_labels= load_variables('rasm-lstm.model')

loss, accuracy = model.evaluate(test_sequences, test_labels)
print(f'Accuracy on test data: {accuracy}')

# Example prediction
def predict(string):
    sequence = tokenizer.texts_to_sequences([string])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length, padding='post')
    evals = model.predict(padded_sequence)
    predicted_index = np.argmax(evals)
    return( predicted_index, max(max(evals)) )

# Test the prediction function
predict("222") # alif

# the hurf lookup
hurf= [''] * 40
hurf[1]= 'ا'
hurf[2]= 'ب'
hurf[3]= 'ت'
hurf[4]= 'ة'
hurf[5]= 'ث'
hurf[6]= 'ج'
hurf[7]= 'چ'
hurf[8]= 'ح'
hurf[9]= 'خ'
hurf[10]= 'د'
hurf[11]= 'ذ'
hurf[12]= 'ر'
hurf[13]= 'ز'
hurf[14]= 'س'
hurf[15]= 'ش'
hurf[16]= 'ص'
hurf[17]= 'ض'
hurf[18]= 'ط'
hurf[19]= 'ظ'
hurf[20]= 'ع'
hurf[21]= 'غ'
hurf[22]= 'ڠ'
hurf[23]= 'ف'
hurf[24]= 'ڤ'
hurf[25]= 'ق'
hurf[26]= 'ک'
hurf[27]= 'ݢ'
hurf[28]= 'ل'
hurf[29]= 'م'
hurf[30]= 'ن'
hurf[31]= 'و'
hurf[32]= 'ۏ'
hurf[33]= 'ه'
hurf[34]= 'ء'
hurf[35]= 'ي'
hurf[36]= 'ی'
hurf[37]= 'ڽ'


def stringtorasm(strokeorder):
    remainder_stroke= strokeorder
    rasm=''
    while len(remainder_stroke)>=2 and remainder_stroke!='':
        label_best=''
        eval_best=0
        len_best=-1
        len_current=len(remainder_stroke)
        for n in range(min_length, max_length+1):
            if n>len_current:
                break
            tee_string= remainder_stroke[0:n]
            tee_label, tee_eval= predict(tee_string)
            tee_eval *= pow(PHI, n)
            if tee_eval> eval_best:
                eval_best= tee_eval
                label_best= tee_label
                len_best= n
        hurf_best= hurf[label_best]
        rasm+= hurf_best
        if hurf_best=='ا' or hurf_best=='د' or hurf_best=='ذ' or hurf_best=='ر' or hurf_best=='ز' or hurf_best=='و':
            remainder_stroke=''
        else:
            remainder_stroke= remainder_stroke[len_best:]
        if remainder_stroke=='':
            break
    return(rasm)
        
        
        
    



