import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# todo
# and-or graph network (based on tabulated LCS?)
# or perhaps a leaf node is a set of path with +/-1 variation

PHI= 1.6180339887498948482 # ppl says this is a beautiful number :)

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
hurf[32]= 'ۏ'
hurf[33]= 'ه'
hurf[34]= 'ء'
hurf[35]= 'ي'
hurf[36]= 'ی'
hurf[37]= 'ڽ'

# Generate random data
np.random.seed(42)

# Parameters
LENGTH_MIN= 2
LENGTH_MAX = 12  # Max length of the strings
NUM_CLASSES = 40  # Number of classes

# data
source = pd.read_csv('olah.csv')
#random_strings=pd.concat([source['2bfs'], source['2alpha-bfsdfs']])
#random_labels=pd.concat([source['label'], source['label']])
random_strings=pd.concat([source['rasm']])
random_labels=pd.concat([source['val']])
#source['rasm'].str.len().mean()
#source['rasm'].str.len().min()
#source['rasm'].str.len().max()



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Reshape, Lambda
from sklearn.model_selection import train_test_split

tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
hurf_mapping = source.set_index('val')['hurf'].to_dict()
val_distribution = source['val'].value_counts()
val_distribution.index = val_distribution.index.map(hurf_mapping)
# Plotting the distribution using matplotlib
plt.figure(figsize=(10, 6))
val_distribution.plot(kind='bar')
plt.xlabel('Hurf Labels')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Tokenize the strings
tokenizer.fit_on_texts(random_strings)
sequences = tokenizer.texts_to_sequences(random_strings)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

# Convert labels to one-hot encoding
labels = tf.keras.utils.to_categorical(random_labels, NUM_CLASSES=NUM_CLASSES)

# Split data into training and testing sets
train_sequences, test_sequences, train_labels, test_labels = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=38)

# Parameters for the model
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token
embedding_dim = 50  # Dimension of the embedding vector

# Define the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=LENGTH_MAX),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(LENGTH_MAX, activation='softmax'),
    Dense(NUM_CLASSES, activation='sigmoid'),
    Lambda(lambda x: x * 8)            # Scale the output to range [0, 8]
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
epochs = 24000
batch_size = 32
#model.fit(train_sequences, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)
history = model.fit(train_sequences, train_labels, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    validation_split=0.1)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

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

def predict(string):
    sequence = tokenizer.texts_to_sequences([string])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=LENGTH_MAX, padding='post')
    evals = model.predict(padded_sequence)
    predicted_index = np.argmax(evals)
    return( predicted_index, max(max(evals)) )

# Test the prediction function
# predict("222") # alif

# checking the weights of the final Dense layer
weights, biases = model.layers[-2].get_weights()
for i in range(0,NUM_CLASSES):
    plt.figure()
    plt.plot(weights[:,i])
    if i==0:
        label= 'space'
    else:
        label= hurf[i]
    plt.text(np.pi, 0, label, fontsize=32, ha='left', va='bottom', color='red')

def stringtorasm_LSTM(strokeorder):
    remainder_stroke= strokeorder
    rasm=''
    while len(remainder_stroke)>=2 and remainder_stroke!='':
        label_best=''
        eval_best=0
        len_best=-1
        len_current=len(remainder_stroke)
        for n in range(LENGTH_MIN, LENGTH_MAX+1):
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

        

#### LCS 

import seaborn as sns
from collections import defaultdict

LCS_FREQ= 32
LCS_MIN= 2

score = {f'{i}': [] for i in range(0, NUM_CLASSES)}
appearance = np.zeros(40, dtype=float)

def update_rasm_score(hurf_class, rasm_seq):
    if hurf_class in score:
        for chaincode in score[hurf_class]:
            if chaincode['seq'] == rasm_seq:
                chaincode['score'] += 1  
                return True  # Successfully updated
        score[hurf_class].append({'seq': rasm_seq, 'score': 1})
        return False  
    return False  


def lcs_tabulate(val, string):
    appearance[val] += 1
    length= len(string)
    for i in range(length):
        for j in range(i + 1, length + 1):
            substring = string[i:j]
            if len(substring) > 2:
                update_rasm_score(str(val), substring)

fieldstring= 'rasm'
fieldval= 'val'
for i in range(0,source.shape[0]):
    #print(f"{i} {source[fieldstring][i]} {source[fieldval][i]}")
    lcs_tabulate(int(source[fieldval][i]), str(source[fieldstring][i]).replace(' ', ''))

top_LCS = {}
for hurf_class, rasm_seq in score.items():
    sorted_token = sorted(rasm_seq, key=lambda x: x['score'], reverse=True)
    top_LCS[hurf_class] = [token for token in sorted_token[:LCS_FREQ] if token['score'] >= LCS_MIN]


# check for duplicates
seq_indices = defaultdict(list)
for key, entries in top_LCS.items():
    for i, entry in enumerate(entries):
        #seq_indices[entry['seq']].append(key)
        seq_indices[entry['seq']].append(hurf[int(key)])
duplicates_with_indices = {seq: indices for seq, indices in seq_indices.items() if len(indices) > 1}


llcs = np.zeros((NUM_CLASSES, LCS_FREQ))
slcs = np.zeros((NUM_CLASSES, LCS_FREQ))
alcs = [["" for i in range(LCS_FREQ)] for j in range(NUM_CLASSES)]

for j in range(0, NUM_CLASSES):
    if top_LCS[str(j)] is not None:
        for i in range(0, len(top_LCS[str(j)]) ):
            llcs[j][i] = len(top_LCS[str(j)][i]['seq']) # length of each substring
            slcs[j][i] = top_LCS[str(j)][i]['score']/appearance[j] # apperance frequency of each substring
            alcs[j][i] = top_LCS[str(j)][i]['seq'] # the substring itself

# important to plot these graphs
plt.figure(dpi=300)
sns.set_theme(rc={
    'figure.figsize': (8, 8),
    'font.family': ['Noto Naskh Arabic', 'Noto Sans'],
    'font.size': 6,  # Adjust font size if necessary
    'xtick.labelsize': 6,
    'ytick.labelsize': 6
})
#sns.heatmap(llcs, cmap='nipy_spectral', annot=True, cbar=True, fmt='g', annot_kws={"size": 4})
sns.heatmap(slcs, cmap='nipy_spectral', annot=alcs, cbar=True, fmt='', annot_kws={"size": 4}, cbar_kws={"ticks": np.arange(0, 1.01, 0.1), "format": "%.1f"})
plt.imshow(llcs, cmap='nipy_spectral', interpolation='nearest')
plt.yticks(ticks=range(40), labels=hurf, rotation=0, fontsize=6)
plt.xticks(fontsize=6, rotation=45)
plt.savefig("/shm/heatmap-lcs.png")
# plt.xticks(ticks=range(len(y_labels)), labels=y_labels)
#ax = plt.gca()
#for tick in ax.get_yticklabels():
#    tick.set_y(tick.get_position()[1] + 400)  # Move tick labels down
#plt.show()

counts, bins = np.histogram(slcs.flatten(), bins=LCS_FREQ)
adjusted_counts = counts / np.count_nonzero(slcs)
plt.bar(bins[:-1], adjusted_counts, width=np.diff(bins), edgecolor='black', align='edge')
# Adding labels
plt.xlabel("Probability of subsequence appearance for each hurf")
plt.ylabel("Adjusted Probability of subsequence from the kitab")
distribution_threshold = 1-1/PHI
plt.axvline(x=distribution_threshold, color='red', linestyle='--', linewidth=2, label='x=0.381966')
# Annotate the vertical line
plt.annotate("1-(1/φ)", 
             xy=(distribution_threshold, 0),  # Position of the annotation (x, y)
             xytext=(distribution_threshold + 0.05, 0.3),  # Position of the text (adjust as needed)
             #arrowprops=dict(facecolor='black', arrowstyle='->'),  # Arrow pointing to the line
             fontsize=12, color='black')


def myjaro(s1,s2):
    prefix_weight: float = 0.1
    s1_len = len(s1)
    s2_len = len(s2)

    if not s1_len or not s2_len:
        #result= 0
        return 0.0

    min_len = min(s1_len, s2_len)
    search_range = max(s1_len, s2_len)
    search_range = (search_range // 2) - 1
    if search_range < 0:
        search_range = 0

    s1_flags = [False] * s1_len
    s2_flags = [False] * s2_len

    # looking only within search range, count & flag matched pairs
    common_chars = 0
    almost_common_chars= 0
    for i, s1_ch in enumerate(s1):
        low = max(0, i - search_range)
        hi = min(i + search_range, s2_len - 1)
        for j in range(low, hi + 1):
            #print(f"eval s1[{i}]:{s1[i]} s2[{j}].replace(' ', ''):{s2[j]}")
            if not s2_flags[j] and s2[j] == s1_ch:
                s1_flags[i] = s2_flags[j] = True
                common_chars += 1
                #print(f"com {i}-{j}: {s1_ch}")
                break
            if abs(ord(s1[i])-ord(s2[j]))==1 or abs(ord(s1[i])-ord(s2[j]))==4 or abs(ord(s1[i])-ord(s2[j]))==7:
                almost_common_chars += 1
                #print(f"almostcom {i}-{j}: {s1[i]}")
                break
            
    if almost_common_chars and not common_chars:
        common_chars=1e-12
    if not common_chars and not almost_common_chars:
        return 0.0
        #result=0

    # count transpositions
    k = trans_count = 0
    for i, s1_f in enumerate(s1_flags):
        if s1_f:
            for j in range(k, s2_len):
                if s2_flags[j]:
                    k = j + 1
                    break
            if s1[i] != s2[j]:
                trans_count += 1
    trans_count //= 2

    # adjust for similarities in nonmatched characters
    weight = common_chars / s1_len + common_chars / s2_len
    weight += (common_chars - trans_count) / common_chars
    weight += (almost_common_chars) / (s1_len+s2_len) /4
    weight /= 3

    # # stop to boost if strings are not similar
    # if not self.winklerize:
    #     return weight
    # if weight <= 0.7:
    #     return weight

    # winkler modification
    # adjust for up to first 4 chars in common
    j = min(min_len, 3)
    i = 0
    while i < j and s1[i] == s2[i]:
        i += 1
    if i:
        weight += i * prefix_weight * (1.0 - weight)

    # optionally adjust for long strings
    # after agreeing beginning chars, at least two or more must agree and
    # agreed characters must be > half of remaining characters
    # if not self.long_tolerance or min_len <= 4:
    #     return weight
    if common_chars <= i + 1 or 2 * common_chars < min_len + i:
        return weight
    tmp = (common_chars - i - 1) / (s1_len + s2_len - i * 2 + 2)
    weight += (1.0 - weight) * tmp
    return weight

MC_RETRY_MAX= 1e4


appearance = np.zeros(40, dtype=float)

def stringtorasm_MC(chaincode):
    remainder_stroke= chaincode
    rasm=''
    while len(remainder_stroke)>=2 and remainder_stroke!='':
        tee_best=''
        lookup_best=''
        class_best=-1
        score_best=-1
        len_best=-1
        len_current=len(remainder_stroke)
        mc_retry= 0
        for n in range(LENGTH_MIN, int(LENGTH_MAX*PHI)):
            if n>len_current:
                break
            tee_string= remainder_stroke[0:n]
        while(mc_retry < MC_RETRY_MAX):
            _ = ''
            while _=='': # avoid empty hurf class
                mc_class= int(np.random.rand()*NUM_CLASSES)
                _=alcs[mc_class][0]
                lcs_lookup=''
                lcs_prob=1
            
            appearance[mc_class] += 1
            for m in range(LENGTH_MIN, len(tee_string), 1):
                tee_tmp= tee_string[0:m]
                
                while len(lcs_lookup) < LENGTH_MAX*PHI*PHI:
                    mc_index= int(np.random.rand()*LCS_FREQ)
                    if alcs[mc_class][mc_index] != '':
                        lcs_lookup += alcs[mc_class][mc_index]
                        lcs_prob *= slcs[mc_class][mc_index]
                    # similarity eval, HOPE for the best
                    score= myjaro(tee_tmp.replace(' ', '').replace('+', '').replace('-', ''), \
                                  lcs_lookup.replace(' ', '').replace('+', '').replace('-', '')) \
                            *pow(PHI, len(tee_tmp)) #\ # penalty for shorter chain should should be applied?
                                #* lcs_prob # may not be needed since will skew to those rare hurfs
                            # SHALL WE FACTOR IN THE SUBSTRING PROBABILITY TOO? i.e. slcs
                    if score==0:
                        break
                    #print(f"ret{mc_retry}\tclass{mc_class}\tscore{score:.2f}\t{tee_tmp}\t{lcs_lookup} @{lcs_prob:.2f}")
                    elif score>score_best:
                        score_best= score
                        len_best= m
                        class_best= mc_class
                        tee_best=tee_tmp
                        lookup_best=lcs_lookup
                        print(f"ret {mc_retry}\tclass {class_best} ({hurf[class_best]})\tscore {score_best:.2f}\t{tee_best}")
            mc_retry= mc_retry+1 # up can be incremented anywhere in the nesting
        
        hurf_best= hurf[class_best]
        print(f"BEST class{class_best} ({hurf_best})\tscore{score_best:.2f}\t{tee_best}\t{lookup_best}")
        
        # MAY allow to skip of evaluation if we are unsure
        # if eval_best > 0.5 and len_current<LENGTH_MIN*PHI:
        #     rasm+= hurf_best
        rasm+= hurf_best
        if hurf_best=='ا' or hurf_best=='د' or hurf_best=='ذ' or hurf_best=='ر' or hurf_best=='ز' or hurf_best=='و':
            remainder_stroke=''
        else:
            remainder_stroke= remainder_stroke[len_best:]
        if remainder_stroke=='':
            break
    return(rasm)


import sys
from contextlib import contextmanager

class DualOutput:
    def __init__(self, file):
        self.file = file

    def write(self, message):
        self.file.write(message)
        # Also write to standard output
        sys.__stdout__.write(message)

    def flush(self):
        self.file.flush()
        sys.__stdout__.flush()

@contextmanager
def redirect_stdout_to_file_and_console(file_path):
    with open(file_path, 'w') as f:
        original_stdout = sys.stdout  # Save the original stdout
        sys.stdout = DualOutput(f)    # Redirect to DualOutput
        yield  # Control goes to the block of code that uses this context manager
        sys.stdout = original_stdout  # Restore original stdout after the block

# Use the context manager
with redirect_stdout_to_file_and_console('/shm/markicob3.txt'):
    stringtorasm_MC("564304+4053+434675440")
    



import textdistance
def stringtorasm_LCS(strokeorder):
    remainder_stroke= strokeorder
    rasm=''
    while len(remainder_stroke)>=2 and remainder_stroke!='':
        label_best=''
        eval_best=-1
        len_best=-1
        len_current=len(remainder_stroke)
        for n in range(LENGTH_MIN, LENGTH_MAX+1):
            if n>len_current:
                break
            tee_string= remainder_stroke[0:n]
            for j in range(0, NUM_CLASSES):
                if top_LCS[str(j)] is not None:
                    for i in range(0, len(top_LCS[str(j)]) ):
                        #tee_eval= fuzz.ratio(tee_string, LCS[i][j]) *pow(PHI, len(tee_string))
                        tee_eval= textdistance.jaro.similarity(tee_string, top_LCS[str(j)][i]['seq'], 0.1) *pow(PHI, len(tee_string))
                        if tee_eval> eval_best:
                            #print(f"length:{n} class:{label_best} score:{eval_best}")
                            eval_best= tee_eval
                            label_best= i
                            len_best= n
        hurf_best= hurf[label_best]
        #print(f"BEST length:{n} class:{label_best} score:{eval_best}")
        rasm+= hurf_best
        if hurf_best=='ا' or hurf_best=='د' or hurf_best=='ذ' or hurf_best=='ر' or hurf_best=='ز' or hurf_best=='و':
            remainder_stroke=''
        else:
            remainder_stroke= remainder_stroke[len_best:]
        if remainder_stroke=='':
            break
    return(rasm)



#-------

from itertools import product
from collections import defaultdict, Counter

def lcs_multiple(strings, LENGTH_MIN=3):
    if not strings:
        return []

    num_strings = len(strings)
    lengths = [len(s) for s in strings]

    # Create a multi-dimensional array to store lengths of LCS and sets of LCS substrings with counts
    dp = {}
    for indices in product(*(range(length + 1) for length in lengths)):
        dp[indices] = (0, defaultdict(int))

    # Build the dp array in bottom-up fashion
    for indices in product(*(range(length + 1) for length in lengths)):
        if all(index == 0 for index in indices):
            continue

        LENGTH_MAX = 0
        max_subseqs = defaultdict(int)
        for i in range(num_strings):
            if indices[i] > 0:
                prev_indices = indices[:i] + (indices[i] - 1,) + indices[i + 1:]
                if dp[prev_indices][0] > LENGTH_MAX:
                    LENGTH_MAX = dp[prev_indices][0]
                    max_subseqs = dp[prev_indices][1].copy()

                elif dp[prev_indices][0] == LENGTH_MAX:
                    for subseq, count in dp[prev_indices][1].items():
                        max_subseqs[subseq] += count

        current_chars = [strings[i][indices[i] - 1] for i in range(num_strings) if indices[i] > 0]
        if len(current_chars) == num_strings and all(char == current_chars[0] for char in current_chars):
            prev_indices = tuple(index - 1 for index in indices)
            new_subseqs = defaultdict(int)
            for subseq, count in dp[prev_indices][1].items() or {("", 1)}:
                new_subseqs[subseq + current_chars[0]] = count

            if dp[prev_indices][0] + 1 > LENGTH_MAX:
                LENGTH_MAX = dp[prev_indices][0] + 1
                max_subseqs = new_subseqs

            elif dp[prev_indices][0] + 1 == LENGTH_MAX:
                for subseq, count in new_subseqs.items():
                    max_subseqs[subseq] += count

        dp[indices] = (LENGTH_MAX, max_subseqs)

    # Combine all substrings and their counts
    all_subseqs = Counter()
    for indices in dp:
        for subseq, count in dp[indices][1].items():
            if len(subseq) >= LENGTH_MIN:
                all_subseqs[subseq] += count

    # Filter out the common substrings (appear in all strings)
    common_subseqs = {subseq: count for subseq, count in all_subseqs.items() if count >= num_strings}

    if len(common_subseqs):
        # If there are common substrings, return them sorted by count (descending) and lexicographically
        return sorted(common_subseqs.items(), key=lambda x: (-x[1], x[0]))
    else:
    # If no common substring, return top 6 substrings sorted by count (descending) and lexicographically
        return sorted(all_subseqs.items(), key=lambda x: (-x[1], x[0]))[:6]


def damerau_levenshtein_freeman_distance(s1, s2):
    d = {}
    len_str1 = len(s1)
    len_str2 = len(s2)

    for i in range(-1, len_str1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len_str2 + 1):
        d[(-1, j)] = j + 1

    for i in range(len_str1):
        for j in range(len_str2):
            #cost = 0 if s1[i] == s2[j] else 1
            diff= abs( ord(s1[i-1])%8-ord(s2[i-1])%8 )
            if diff>4:
                diff= 8-diff
            if s1[i-1] == s2[j-1]:
                cost = 0
            elif diff==1:
                cost = 0.5
            elif diff==4:
                cost = 0.125
            else:
                cost = 1
            
            # levenshtein part            
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,  # deletion
                d[(i, j - 1)] + 1,  # insertion
                d[(i - 1, j - 1)] + cost,  # substitution
            )
            # damerau part
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition, 

    return d[len_str1 - 1, len_str2 - 1]


def fuzzy_substring_matching(template, long_string):
    if long_string!='':
        min_distance = float('inf')
        best_match = None
        best_start_index = -1
        len_template = len(template)
        len_long_string = len(long_string)
        
        for i in range(len_long_string - len_template + 1):
            substring = long_string[i:i + len_template]
            distance = damerau_levenshtein_freeman_distance(template, substring) / len_template
            if distance < min_distance:
                min_distance = distance
                best_match = substring
                #best_index = i
        
        #print(f"match: {long_string[:best_index]}")
        #print(f"remainder: {long_string[best_start_index + len_template:]}")
        if best_match is not None: # and perhahps min_distance threshold too
            remainder = long_string[best_start_index + len_template:]
        else:
            remainder = ''
        return best_match, min_distance, remainder


def stringtorasm_LEV(remainder_stroke):
    rasm=''
    while len(remainder_stroke)>=3 and remainder_stroke!='':
        # find the substring with smalesst edit distance
        lev_dist_min=1e9
        hurf_min=''
        template_min=''
        remainder_min=''
        for template, data in hurf.nodes(data=True):
            if len(remainder_stroke)>=3: 
                hurf_temp, lev_dist_temp, remainder_temp= fuzzy_substring_matching(template, remainder_stroke)
                if lev_dist_temp<lev_dist_min:
                    template_min= template
                    hurf_min=data['label']
                    lev_dist_min= lev_dist_temp
                    remainder_min= remainder_temp
                    # adding rules for terminal hurf
            else:
                remainder_stroke=''
                break
        # found the best possible match
        # distance selection can be applied here
        if template_min!='':
            rasm+=hurf_min
        if hurf_min=='ا' or hurf_min=='د' or hurf_min=='ذ' or hurf_min=='ر' or hurf_min=='ز' or hurf_min=='و':
            remainder_stroke=''
        else:
            remainder_stroke= remainder_min
        #print(f"current match: {hurf_min} ({template_min}) from dist {lev_dist_min}, rasm is {rasm}, remainder is {remainder_stroke}")    



# Define the file path (replace 'your_file.txt' with your actual file name)
import sys
file_path = sys.argv[1]

with open(file_path, 'r') as file:
    # Iterate over each line in the file
    for line in file:
        #print(line, end='')  # The 'end' argument avoids adding extra newlines
        print(f"{stringtorasm_LCS(line)} ")
        

